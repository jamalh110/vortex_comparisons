from typing import Any, Dict, List
from ray import serve
import ray
from flmr import FLMRConfig, FLMRQueryEncoderTokenizer
from transformers import BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from torch import Tensor, nn
import torch
from utils import make_logger, logfunc
import os
import logging
from collections import deque

_MAX_BATCH_SIZE = 32
DATA_DIR = os.environ.get("DATA_ROOT", "/mydata")
LOG_DIR = os.environ.get("LOG_ROOT", "/users/jamalh11/raylogs")


def _get_nixl_payloads(items: List[Dict[str, Any]]) -> List[Dict[str, Tensor]]:
    """Fetch nested RDT refs directly into this replica's GPU."""
    return ray.get([item["rdt_ref"] for item in items])


@serve.deployment
class StepD:
    def __init__(self, loglevel):
        self._rdt_refs = deque(maxlen=512)
        self.logger = make_logger(os.getpid(), "StepD", LOG_DIR)
        if loglevel == "INFO":
            self.logger.setLevel(logging.INFO)
        elif loglevel == "CRITICAL":
            self.logger.setLevel(logging.CRITICAL)
        else:
            raise Exception("Invalid log level")
        self.flmr_config = None
        self.skiplist = []
        self.query_tokenizer = None

        self.transformer_mapping_cross_attention_length = 32
        self.vision_encoder_embedding_size = 1024
        self.late_interaction_embedding_size = 128
        self.checkpoint_path = f"{DATA_DIR}/PreFLMR_ViT-L"
        self.transformer_mapping_config_base = "bert-base-uncased"
        self.local_tf_mapping_path = (
            f"{DATA_DIR}/EVQA/models/models_step_D_transformer_mapping.pt"
        )
        self.local_tf_mapping_output_path = (
            f"{DATA_DIR}/EVQA/models/models_step_D_transformer_mapping_output.pt"
        )
        self.local_model_path_stepc = (
            f"{DATA_DIR}/EVQA/models/models_step_C_transformer_mapping_input_linear.pt"
        )
        self.transformer_mapping_network = None
        self.transformer_mapping_output_linear = None
        self.transformer_mapping_input_linear = None
        self.mask_instruction = False

        self.collected_intermediate_results = {}
        self.load_model_cpu()
        self.load_model_gpu()

    def load_model_cpu(self):
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        transformer_mapping_config = BertConfig.from_pretrained(
            self.transformer_mapping_config_base
        )
        transformer_mapping_config.is_decoder = True
        transformer_mapping_config.add_cross_attention = True
        transformer_mapping_config.num_hidden_layers = 1

        self.transformer_mapping_network = BertEncoder(transformer_mapping_config)
        self.transformer_mapping_network.load_state_dict(
            torch.load(self.local_tf_mapping_path, weights_only=True)
        )
        self.transformer_mapping_output_linear = nn.Linear(
            transformer_mapping_config.hidden_size, self.late_interaction_embedding_size
        )
        self.transformer_mapping_output_linear.load_state_dict(
            torch.load(self.local_tf_mapping_output_path, weights_only=True)
        )
        self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
            self.checkpoint_path,
            text_config=self.flmr_config.text_config,
            subfolder="query_tokenizer",
        )
        if self.flmr_config.mask_instruction_token is not None:
            self.mask_instruction = True
            self.instruction_token_id = self.query_tokenizer.encode(
                self.flmr_config.mask_instruction_token, add_special_tokens=False
            )[0]
        else:
            self.mask_instruction = False

        transformer_mapping_config_base_stepc = (
            self.flmr_config.transformer_mapping_config_base
        )
        transformer_mapping_config_stepc = BertConfig.from_pretrained(
            transformer_mapping_config_base_stepc
        )
        transformer_mapping_config_stepc.num_hidden_layers = (
            self.flmr_config.transformer_mapping_num_hidden_layers
        )
        transformer_mapping_config_stepc.is_decoder = True
        transformer_mapping_config_stepc.add_cross_attention = True

        self.transformer_mapping_input_linear = nn.Linear(
            self.flmr_config.vision_config.hidden_size,
            transformer_mapping_config_stepc.hidden_size,
        )
        self.transformer_mapping_input_linear.load_state_dict(
            torch.load(self.local_model_path_stepc, weights_only=True)
        )

    def load_model_gpu(self):
        self.transformer_mapping_network.cuda()
        self.transformer_mapping_output_linear.cuda()
        self.transformer_mapping_input_linear.cuda()

    def mask(self, input_ids, skiplist):
        return [
            [(x not in skiplist) and (x != 0) for x in d]
            for d in input_ids.detach().cpu().tolist()
        ]

    def query_mask(self, input_ids, skiplist):
        if not self.mask_instruction:
            return self.mask(input_ids, skiplist)

        sep_id = self.instruction_token_id
        sep_positions = torch.argmax((input_ids == sep_id).int(), dim=1).tolist()
        for i, x in enumerate(sep_positions):
            if x < 1:
                sep_positions[i] = 1
        mask = [
            [
                (x not in skiplist)
                and (x != 0)
                and (index > sep_positions[seq_index] or index < 2)
                for index, x in enumerate(d)
            ]
            for seq_index, d in enumerate(input_ids.detach().cpu().tolist())
        ]
        return mask

    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(
            dtype=encoder_attention_mask.dtype
        )
        encoder_extended_attention_mask = (
            1.0 - encoder_extended_attention_mask
        ) * torch.finfo(encoder_attention_mask.dtype).min
        return encoder_extended_attention_mask

    def proces_queries(
        self,
        input_ids,
        text_embeddings,
        text_encoder_hidden_states,
        vision_embeddings,
        transformer_mapping_input_features,
    ):
        if self.transformer_mapping_network is None:
            self.load_model_cpu()
            self.load_model_gpu()

        mask = (
            torch.tensor(self.query_mask(input_ids, skiplist=self.skiplist))
            .unsqueeze(2)
            .float()
            .cuda()
        )
        text_embeddings = text_embeddings.to(mask.device) * mask
        encoder_mask = torch.ones_like(mask).to(mask.device, dtype=mask.dtype)
        if text_encoder_hidden_states.shape[1] > self.transformer_mapping_cross_attention_length:
            text_encoder_hidden_states = text_encoder_hidden_states[
                :, : self.transformer_mapping_cross_attention_length
            ]
            encoder_mask = encoder_mask[:, : self.transformer_mapping_cross_attention_length]
        encoder_extended_attention_mask = self.invert_attention_mask(
            encoder_mask.squeeze(-1)
        )

        transformer_mapping_outputs = self.transformer_mapping_network(
            transformer_mapping_input_features.to(mask.device),
            encoder_hidden_states=text_encoder_hidden_states.to(mask.device),
            encoder_attention_mask=encoder_extended_attention_mask.to(mask.device),
        )
        transformer_mapping_output_features = transformer_mapping_outputs.last_hidden_state
        transformer_mapping_output_features = self.transformer_mapping_output_linear(
            transformer_mapping_output_features
        )

        vision_embeddings = torch.cat(
            [vision_embeddings.to(mask.device), transformer_mapping_output_features],
            dim=1,
        )

        Q = torch.cat([text_embeddings, vision_embeddings], dim=1)
        query_embeddings = torch.nn.functional.normalize(Q, p=2, dim=2).detach()
        return query_embeddings

    @serve.batch(max_batch_size=_MAX_BATCH_SIZE)
    async def __call__(
        self, stepa: List[Dict[str, Any]], stepb: List[Dict[str, Any]], requestid: List[str]
    ):
        logfunc(self.logger, [{"requestid": i} for i in requestid], "StepD_Enter")

        stepa_payloads = _get_nixl_payloads(stepa)
        stepb_payloads = _get_nixl_payloads(stepb)

        vision_second_last_layer_hidden_states = torch.stack(
            [
                x["vision_second_last_layer_hidden_states"]
                for x in stepb_payloads
            ],
            dim=0,
        )

        transformer_mapping_input_features = self.transformer_mapping_input_linear(
            vision_second_last_layer_hidden_states
        )

        input_ids = torch.stack([x["input_ids"] for x in stepa_payloads], dim=0)
        text_embeddings = torch.stack(
            [x["text_embeddings"] for x in stepa_payloads], dim=0
        )
        text_encoder_hidden_states = torch.stack(
            [x["text_encoder_hidden_states"] for x in stepa_payloads], dim=0
        )
        vision_embeddings = torch.stack(
            [x["vision_embeddings"] for x in stepb_payloads], dim=0
        )

        query_embeddings = self.proces_queries(
            input_ids,
            text_embeddings,
            text_encoder_hidden_states,
            vision_embeddings,
            transformer_mapping_input_features,
        )
        torch.cuda.synchronize()
        results = []
        for i in range(query_embeddings.shape[0]):
            ref = ray.put(
                query_embeddings[i].contiguous(),
                _tensor_transport="nixl",
            )
            self._rdt_refs.append(ref)
            results.append({"rdt_ref": ref})
        logfunc(self.logger, [{"requestid": i} for i in requestid], "StepD_Exit")
        return results
