#!/usr/bin/env python3
"""
Ray Serve pipeline1 with Ray Direct Transport (NIXL/RDMA).

Based on main_directsend.py, but keeps torch.Tensors on GPU across
StepA/B/D/E and marks producer methods with tensor_transport=\"nixl\" so
Ray transfers them via NIXL (UCX/RDMA) instead of the CPU object store.
"""
import asyncio
from datetime import datetime
import time
from ray import serve, cloudpickle
import json
import ray
from starlette.requests import Request
from ray.serve.handle import DeploymentHandle
from ray.serve import Application
from ray.serve.schema import LoggingConfig
from typing import Any, Dict, List
import os
import torch
from torch import Tensor, nn
from flmr import (
    FLMRConfig,
    FLMRQueryEncoderTokenizer,
    FLMRContextEncoderTokenizer,
    FLMRModelForRetrieval,
    FLMRTextModel,
    FLMRVisionModel,
    search_custom_collection,
    create_searcher,
)
from transformers import AutoImageProcessor, BertConfig
import io
from PIL import Image
from easydict import EasyDict
from transformers.models.bert.modeling_bert import BertEncoder
from StepD_rdma import StepD
import random
import string
import numpy as np
import faulthandler
import traceback
import signal
import pickle
import logging
from collections import deque

from utils import make_logger, logfunc

_MAX_BATCH_SIZE = 32
_STEP_E_BATCH_SIZE = 32
DATA_DIR = os.environ.get("DATA_ROOT", "/mydata")
LOG_DIR = os.environ.get("LOG_ROOT", "/users/jamalh11/raylogs")

def _put_nixl(value: Any, keepalive: deque) -> ray.ObjectRef:
    """Keep tensors on GPU and publish them through Ray Direct Transport."""
    ref = ray.put(value, _tensor_transport="nixl")
    # Serve serializes this ObjectRef inside a normal response. Keep the owner-side
    # reference alive until downstream has had ample time to fetch it.
    keepalive.append(ref)
    return ref


def _get_nixl_payloads(items: List[Dict[str, Any]]) -> List[Any]:
    """Fetch nested RDT refs inside the receiving Serve replica."""
    return ray.get([item["rdt_ref"] for item in items])


@serve.deployment
class StepA:
    def __init__(self, loglevel):
        self._rdt_refs = deque(maxlen=512)
        self.logger = make_logger(os.getpid(), "StepA", LOG_DIR)
        if loglevel == "INFO":
            self.logger.setLevel(logging.INFO)
        elif loglevel == "CRITICAL":
            self.logger.setLevel(logging.CRITICAL)
        self.checkpoint_path = f"{DATA_DIR}/PreFLMR_ViT-L"
        self.local_encoder_path = f"{DATA_DIR}/EVQA/models/models_step_A_query_text_encoder.pt"
        self.local_projection_path = f"{DATA_DIR}/EVQA/models/models_step_A_query_text_linear.pt"
        self.flmr_config = None
        self.query_tokenizer = None
        self.context_tokenizer = None
        self.query_text_encoder = None
        self.query_text_encoder_linear = None
        self.device = "cpu"
        self.skiplist = []

        self.load_model_cpu()
        self.load_model_gpu()

    def load_model_cpu(self):
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
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

        self.query_text_encoder = FLMRTextModel(self.flmr_config.text_config)
        self.query_text_encoder_linear = nn.Linear(
            self.flmr_config.text_config.hidden_size, self.flmr_config.dim, bias=False
        )

        try:
            self.query_text_encoder.load_state_dict(
                torch.load(self.local_encoder_path, weights_only=True)
            )
            self.query_text_encoder_linear.load_state_dict(
                torch.load(self.local_projection_path, weights_only=True)
            )
        except Exception:
            print(
                f"Failed to load models checkpoint!!! \n Please check {self.local_encoder_path} or {self.local_projection_path}"
            )

    def load_model_gpu(self):
        self.device = "cuda"
        self.query_text_encoder_linear.to(self.device)
        self.query_text_encoder.to(self.device)

    def prepare_inputs(self, sample):
        sample = EasyDict(sample)

        module = EasyDict(
            {
                "type": "QuestionInput",
                "option": "default",
                "separation_tokens": {"start": "", "end": ""},
            }
        )

        instruction = sample.instruction.strip()
        if instruction[-1] != ":":
            instruction = instruction + ":"
        instruction = instruction.replace(":", self.flmr_config.mask_instruction_token)
        text_sequence = " ".join(
            [instruction]
            + [module.separation_tokens.start]
            + [sample.question]
            + [module.separation_tokens.end]
        )

        sample["text_sequence"] = text_sequence
        return sample

    def stepA_output(self, inputs: List[Dict[str, Any]]):
        input_ids = torch.stack(
            [torch.from_numpy(i["input_ids"]) for i in inputs], dim=0
        ).to(self.device)
        attention_mask = torch.stack(
            [torch.from_numpy(i["attention_mask"]) for i in inputs], dim=0
        ).to(self.device)

        text_encoder_outputs = self.query_text_encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        text_encoder_hidden_states = text_encoder_outputs[0]
        text_embeddings = self.query_text_encoder_linear(text_encoder_hidden_states)

        return {
            "text_embeddings": text_embeddings.detach(),
            "input_ids": input_ids.detach(),
            "text_encoder_hidden_states": text_encoder_hidden_states.detach(),
        }

    @serve.batch(max_batch_size=_MAX_BATCH_SIZE)
    async def __call__(self, inputs: List[Dict[str, Any]]):
        logfunc(self.logger, inputs, "StepA_Enter")
        output = self.stepA_output(inputs)
        text_embeddings = output["text_embeddings"]
        input_ids = output["input_ids"]
        text_encoder_hidden_states = output["text_encoder_hidden_states"]

        torch.cuda.synchronize()
        results = [
            {
                "rdt_ref": _put_nixl(
                    {
                        "text_embeddings": text_embeddings[i].contiguous(),
                        "input_ids": input_ids[i].contiguous(),
                        "text_encoder_hidden_states": (
                            text_encoder_hidden_states[i].contiguous()
                        ),
                    },
                    self._rdt_refs,
                )
            }
            for i in range(text_embeddings.shape[0])
        ]
        logfunc(self.logger, inputs, "StepA_Exit")
        return results


class FLMRMultiLayerPerceptron(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def __init__(self, sizes, bias=True, act=nn.Tanh):
        super(FLMRMultiLayerPerceptron, self).__init__()
        layers = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))
            if i < len(sizes) - 2:
                layers.append(act())
        self.model = nn.Sequential(*layers)


@serve.deployment
class StepB:
    def __init__(self, loglevel):
        self._rdt_refs = deque(maxlen=512)
        self.logger = make_logger(os.getpid(), "StepB", LOG_DIR)
        if loglevel == "INFO":
            self.logger.setLevel(logging.INFO)
        elif loglevel == "CRITICAL":
            self.logger.setLevel(logging.CRITICAL)
        self.checkpoint_path = f"{DATA_DIR}/PreFLMR_ViT-L"
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        self.local_encoder_path = f"{DATA_DIR}/EVQA/models/models_step_B_vision_encoder.pt"
        self.local_projection_path = (
            f"{DATA_DIR}/EVQA/models/models_step_B_vision_projection.pt"
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            f"{DATA_DIR}/clip-vit-large-patch14"
        )
        self.device = "cuda"

        self.query_vision_encoder = FLMRVisionModel(self.flmr_config.vision_config)
        self.query_vision_projection = FLMRMultiLayerPerceptron(
            (
                self.flmr_config.vision_config.hidden_size,
                (self.flmr_config.dim * self.flmr_config.mapping_network_prefix_length)
                // 2,
                self.flmr_config.dim * self.flmr_config.mapping_network_prefix_length,
            )
        )
        self.query_vision_encoder.load_state_dict(
            torch.load(self.local_encoder_path, weights_only=False)
        )
        self.query_vision_projection.load_state_dict(
            torch.load(self.local_projection_path, weights_only=False)
        )
        self.query_vision_projection.cuda()
        self.query_vision_encoder.cuda()

    def StepB_output(self, pixel_values, requestid):
        batch_size = pixel_values.shape[0]
        pixel_values = pixel_values.to(self.device)
        if len(pixel_values.shape) == 5:
            pixel_values = pixel_values.reshape(
                -1, pixel_values.shape[2], pixel_values.shape[3], pixel_values.shape[4]
            )

        with torch.no_grad():
            vision_encoder_outputs = self.query_vision_encoder(
                pixel_values, output_hidden_states=True
            )
            vision_embeddings = vision_encoder_outputs.last_hidden_state[:, 0]
            vision_embeddings = self.query_vision_projection(vision_embeddings)
            vision_embeddings = vision_embeddings.view(
                batch_size, -1, self.flmr_config.dim
            )

        vision_second_last_layer_hidden_states = vision_encoder_outputs.hidden_states[
            -2
        ][:, 1:]

        return {
            "vision_embeddings": vision_embeddings.detach(),
            "vision_second_last_layer_hidden_states": (
                vision_second_last_layer_hidden_states.detach()
            ),
        }

    @serve.batch(max_batch_size=_MAX_BATCH_SIZE)
    async def __call__(self, input: List[Dict[str, Any]]):
        logfunc(self.logger, input, "StepB_Enter")
        pixel_values = []
        for i in input:
            pv = i["pixel_values"]
            if isinstance(pv, torch.Tensor):
                pixel_values.append(pv)
            else:
                pixel_values.append(torch.from_numpy(pv))
        output = self.StepB_output(torch.stack(pixel_values, dim=0), requestid="abcd")
        vision_embeddings = output["vision_embeddings"]
        vision_second_last_layer_hidden_states = output[
            "vision_second_last_layer_hidden_states"
        ]
        torch.cuda.synchronize()
        results = [
            {
                "rdt_ref": _put_nixl(
                    {
                        "vision_embeddings": vision_embeddings[i].contiguous(),
                        "vision_second_last_layer_hidden_states": (
                            vision_second_last_layer_hidden_states[i].contiguous()
                        ),
                    },
                    self._rdt_refs,
                )
            }
            for i in range(vision_embeddings.shape[0])
        ]
        logfunc(self.logger, input, "StepB_Exit")
        return results


@serve.deployment
class StepE:
    def __init__(self, experiment_name: str, index_name: str, loglevel):
        self.logger = make_logger(os.getpid(), "StepE", LOG_DIR)
        if loglevel == "INFO":
            self.logger.setLevel(logging.INFO)
        elif loglevel == "CRITICAL":
            self.logger.setLevel(logging.CRITICAL)
        self.searcher = None
        self.index_root_path = f"{DATA_DIR}/EVQA/index/"
        self.index_experiment_name = experiment_name
        self.index_name = index_name
        self.load_searcher_gpu()

    def load_searcher_gpu(self):
        self.searcher = create_searcher(
            index_root_path=self.index_root_path,
            index_experiment_name=self.index_experiment_name,
            index_name=self.index_name,
            nbits=8,
            use_gpu=True,
        )

    def process_search(self, queries, query_embeddings, bsize):
        ranking = search_custom_collection(
            searcher=self.searcher,
            queries=queries,
            query_embeddings=query_embeddings,
            num_document_to_retrieve=5,
            centroid_search_batch_size=bsize,
        )
        return ranking.todict()

    @serve.batch(max_batch_size=_STEP_E_BATCH_SIZE)
    async def __call__(self, input1: List[Dict[str, Any]], input2: Any):
        logfunc(self.logger, input1, "StepE_Enter")
        bsize = None
        queries = {}
        query_embeddings_list = _get_nixl_payloads(input2)

        for i in input1:
            queries[i["question_id"]] = i["text_sequence"]

        query_embeddings = torch.stack(query_embeddings_list, dim=0)

        tim = time.time()
        output = self.process_search(queries, query_embeddings, bsize)
        print(f"Search took {time.time() - tim} seconds")

        ret = []
        for i in input1:
            ret.append(output[i["question_id"]])

        logfunc(self.logger, input1, "StepE_Exit")
        return ret


@serve.deployment
class Ingress:
    def __init__(
        self,
        stepA: DeploymentHandle,
        stepB: DeploymentHandle,
        stepD: DeploymentHandle,
        stepE: DeploymentHandle,
        loglevel,
    ):
        self.logger = make_logger(os.getpid(), "Ingress", LOG_DIR)
        if loglevel == "INFO":
            self.logger.setLevel(logging.INFO)
        elif loglevel == "CRITICAL":
            self.logger.setLevel(logging.CRITICAL)
        self.stepA = stepA
        self.stepB = stepB
        self.stepD = stepD
        self.stepE = stepE

    async def __call__(self, http_request: Request):
        try:
            requestid = http_request.headers.get("x-requestid")
            if not requestid:
                requestid = "".join(
                    random.choice(string.ascii_uppercase + string.digits) for _ in range(8)
                )
            logfunc(self.logger, [{"requestid": requestid}], "Ingress_Enter")
            input = await http_request.body()
            input = pickle.loads(input)
            input["requestid"] = requestid

            image = {"pixel_values": input["pixel_values"]}
            del input["pixel_values"]
            inputref = ray.put(input)

            image["requestid"] = requestid

            stepA_output = self.stepA.remote(inputref)
            stepB_output = self.stepB.remote(image)
            stepD_output = self.stepD.remote(stepA_output, stepB_output, requestid)
            stepE_output = self.stepE.remote(inputref, stepD_output)
            output = await stepE_output
            logfunc(self.logger, [{"requestid": requestid}], "Ingress_Exit")
            return output
        except Exception as e:
            print("\n\n\n\n ERROR:", e, "\n\n\n\n")
            traceback.print_exc()
            return ["error"]


def app(args: Dict[str, str]) -> Application:
    experiment_name = args.get("experiment_name", "EVQA_train_split/")
    index_name = args.get("index_name", "EVQA_PreFLMR_ViT-L")
    loglevel = args.get("loglevel", "CRITICAL")
    stepA = StepA.bind(loglevel=loglevel)
    stepB = StepB.bind(loglevel=loglevel)
    stepD = StepD.bind(loglevel=loglevel)
    stepE = StepE.bind(
        experiment_name=experiment_name, index_name=index_name, loglevel=loglevel
    )
    return Ingress.bind(
        stepA=stepA, stepB=stepB, stepD=stepD, stepE=stepE, loglevel=loglevel
    )
