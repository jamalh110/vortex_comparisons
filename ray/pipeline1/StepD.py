from typing import Any, Dict, List
from ray import serve
from flmr import FLMRConfig, FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval, FLMRTextModel, FLMRVisionModel
from transformers import AutoImageProcessor, BertConfig
from transformers.models.bert.modeling_bert import BertEncoder
from torch import Tensor, nn
import torch

_MAX_BATCH_SIZE = 32

@serve.deployment
class StepD:
    def __init__(self):
        self.flmr_config = None
        self.skiplist = []
        self.query_tokenizer = None
       
        self.transformer_mapping_cross_attention_length = 32
        self.vision_encoder_embedding_size = 1024
        self.late_interaction_embedding_size = 128
        self.checkpoint_path = '/mydata/PreFLMR_ViT-L'
        self.transformer_mapping_config_base = 'bert-base-uncased'
        self.local_tf_mapping_path = '/mydata/EVQA/models/models_step_D_transformer_mapping.pt'
        self.local_tf_mapping_output_path = '/mydata/EVQA/models/models_step_D_transformer_mapping_output.pt'
        self.transformer_mapping_network = None
        self.transformer_mapping_output_linear = None
        
        self.mask_instruction = False
        
        # Kep track of collected intermediate results: {query_id0: IntermediateResult, query_id2:{} ...}
        self.collected_intermediate_results = {}
        self.load_model_cpu()
        self.load_model_gpu()

    def load_model_cpu(self):
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        transformer_mapping_config = BertConfig.from_pretrained(self.transformer_mapping_config_base)
        transformer_mapping_config.is_decoder = True
        transformer_mapping_config.add_cross_attention = True
        transformer_mapping_config.num_hidden_layers = 1

        self.transformer_mapping_network = BertEncoder(transformer_mapping_config)
        self.transformer_mapping_network.load_state_dict(torch.load(self.local_tf_mapping_path, weights_only=True))
        self.transformer_mapping_output_linear = nn.Linear(
            transformer_mapping_config.hidden_size, self.late_interaction_embedding_size
        )
        self.transformer_mapping_output_linear.load_state_dict(torch.load(self.local_tf_mapping_output_path, weights_only=True))
        self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(
                    self.checkpoint_path, 
                    text_config=self.flmr_config.text_config, 
                    subfolder="query_tokenizer")
        if self.flmr_config.mask_instruction_token is not None:
            self.mask_instruction = True
            # obtain the token id of the instruction token
            self.instruction_token_id = self.query_tokenizer.encode(
                self.flmr_config.mask_instruction_token, add_special_tokens=False
            )[0]
        else:
            self.mask_instruction = False

    def load_model_gpu(self):
        self.transformer_mapping_network.cuda()
        self.transformer_mapping_output_linear.cuda()

    def mask(self, input_ids, skiplist):
        return [[(x not in skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
    
    
    def query_mask(self, input_ids, skiplist):
        if not self.mask_instruction:
            return self.mask(input_ids, skiplist)

        # find the position of end of instruction in input_ids
        # mask the tokens before the position
        sep_id = self.instruction_token_id
        sep_positions = torch.argmax((input_ids == sep_id).int(), dim=1).tolist()
        # if any of the positions is lower than 1, set to 1
        for i, x in enumerate(sep_positions):
            if x < 1:
                sep_positions[i] = 1
        mask = [
            [
                (x not in skiplist) and (x != 0) and (index > sep_positions[seq_index] or index < 2)
                for index, x in enumerate(d)
            ]
            for seq_index, d in enumerate(input_ids.cpu().tolist())
        ]
        return mask
        
    def invert_attention_mask(self, encoder_attention_mask: Tensor) -> Tensor:
        """
        Invert an attention mask (e.g., switches 0. and 1.).

        Args:
            encoder_attention_mask (`torch.Tensor`): An attention mask.

        Returns:
            `torch.Tensor`: The inverted attention mask.
        """
        if encoder_attention_mask.dim() == 3:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
        if encoder_attention_mask.dim() == 2:
            encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
        # T5 has a mask that can compare sequence ids, we can simulate this here with this transposition
        # Cf. https://github.com/tensorflow/mesh/blob/8d2465e9bc93129b913b5ccc6a59aa97abd96ec6/mesh_tensorflow
        # /transformer/transformer_layers.py#L270
        # encoder_extended_attention_mask = (encoder_extended_attention_mask ==
        # encoder_extended_attention_mask.transpose(-1, -2))
        encoder_extended_attention_mask = encoder_extended_attention_mask.to(dtype=encoder_attention_mask.dtype)  # fp16 compatibility
        encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * torch.finfo(encoder_attention_mask.dtype).min

        return encoder_extended_attention_mask
    def proces_queries(self,
                       input_ids,
                       text_embeddings,
                       text_encoder_hidden_states,
                       vision_embeddings,
                       transformer_mapping_input_features,
                       ):
        if self.transformer_mapping_network == None:
            # print('==========start loading model cpu==========')
            self.load_model_cpu()
            
            # print('==========start loading model gpu==========')
            self.load_model_gpu()
            
        
        mask = torch.tensor(self.query_mask(input_ids, skiplist=self.skiplist)).unsqueeze(2).float().cuda()
        text_embeddings = text_embeddings.to(mask.device) * mask
        encoder_mask = torch.ones_like(mask).to(mask.device, dtype=mask.dtype)
        if text_encoder_hidden_states.shape[1] > self.transformer_mapping_cross_attention_length:
            text_encoder_hidden_states = text_encoder_hidden_states[:, :self.transformer_mapping_cross_attention_length]
            encoder_mask = encoder_mask[:, :self.transformer_mapping_cross_attention_length]
        # Obtain cross attention mask
        encoder_extended_attention_mask = self.invert_attention_mask(encoder_mask.squeeze(-1))
        # Pass through the transformer mapping
        
        
        # ENCODER hidden states: Encoder_bsize, Encoder_seqLen, _
        # ENCODER attention mask: ones_like(encoder_hidden_states)

        transformer_mapping_outputs = self.transformer_mapping_network(
            transformer_mapping_input_features.to(mask.device),
            encoder_hidden_states=text_encoder_hidden_states.to(mask.device),
            encoder_attention_mask=encoder_extended_attention_mask.to(mask.device),
        )
        transformer_mapping_output_features = transformer_mapping_outputs.last_hidden_state
        # Convert the dimension to FLMR dim
        transformer_mapping_output_features = self.transformer_mapping_output_linear(
            transformer_mapping_output_features
        )
        # Merge with the vision embeddings
        
        vision_embeddings = torch.cat([vision_embeddings.to(mask.device), transformer_mapping_output_features], dim=1)
        
        Q = torch.cat([text_embeddings, vision_embeddings], dim=1)
        query_embeddings = torch.nn.functional.normalize(Q, p=2, dim=2).detach().cpu()
        return query_embeddings
    
    @serve.batch(max_batch_size=_MAX_BATCH_SIZE)
    async def __call__(self, inputs: List[Dict[str, Any]]):
        print("BATCH SIZE: ",len(inputs))
        #print("stepD inputs\n\n\n", inputs[0]['input_ids'])
        input_ids = torch.stack([x["input_ids"] for x in inputs], dim=0).cuda()
        text_embeddings = torch.stack([x["text_embeddings"] for x in inputs], dim=0).cuda()
        text_encoder_hidden_states = torch.stack([x["text_encoder_hidden_states"] for x in inputs], dim=0).cuda()
        vision_embeddings = torch.stack([x["vision_embeddings"] for x in inputs], dim=0).cuda()
        transformer_mapping_input_features = torch.stack([x["transformer_mapping_input_features"] for x in inputs], dim=0).cuda()
        print("stepD shape\n\n\n", text_embeddings.shape, vision_embeddings.shape, transformer_mapping_input_features.shape)
        query_embeddings = self.proces_queries(input_ids, text_embeddings, text_encoder_hidden_states, vision_embeddings, transformer_mapping_input_features).cpu()
        print("requestid", inputs[0]['requestid'])
        return query_embeddings