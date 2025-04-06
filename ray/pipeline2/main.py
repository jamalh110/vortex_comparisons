import asyncio
from datetime import datetime
import time
from ray import serve
import ray
from starlette.requests import Request
from ray.serve.handle import DeploymentHandle
from ray.serve import Application
from typing import Any, Dict, List
import os 
import torch
from torch import Tensor, nn
from transformers import AutoImageProcessor, BertConfig
import io
import random
import string
import numpy as np
import faulthandler
import traceback
import signal
import pickle
import logging
from myutils import make_logger, logfunc

from funasr.utils.postprocess_utils import rich_transcription_postprocess
from funasr.utils.load_utils import extract_fbank
import sys
from FlagEmbedding import BGEM3FlagModel, FlagModel
import faiss

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SenseVoice"))
from SenseVoice.utils.frontend import WavFrontend, WavFrontendOnline
from SenseVoice.model import SenseVoiceSmall


_MAX_BATCH_SIZE = 32
DATA_DIR="/mydata/pipeline2"
LOG_DIR = "/users/jamalh11/raylogs"
LOG_LEVEL = logging.INFO


@serve.deployment
class StepAudio:
    def __init__(self):
        self.logger = make_logger(os.getpid(), "StepAudio", LOG_DIR)
        self.logger.setLevel(LOG_LEVEL)
        self.language = 'en'
        self.model_dir = "iic/SenseVoiceSmall"
        self.model = None
        self.device_name = "cuda:0"
        self.kwargs = None
        self.frontend = None
        self.load_model()

    def load_model(self):
        self.model, self.kwargs = SenseVoiceSmall.from_pretrained(model=self.model_dir, device=self.device_name)
        self.model.eval()
        self.kwargs["data_type"] = "fbank"
        self.kwargs["sound"] = "fbank"
        self.frontend = self.kwargs["frontend"]
        print("Speech to Text model loaded")

    def exec_model(self, batch_audios):
        speech, speech_lengths = extract_fbank(
            batch_audios, data_type=self.kwargs.get("data_type", "sound"), frontend=self.frontend
        )
        res = self.model.inference(
            data_in=speech,
            data_lengths=speech_lengths,
            language=self.language, 
            use_itn=False,
            ban_emo_unk=True,
            **self.kwargs,
        )
        text_list = []
        for idx in range(len(res[0])):
            text_list.append(rich_transcription_postprocess(res[0][idx]["text"]))
            #print(type(text_list[idx]))
        return text_list

    @serve.batch(max_batch_size=_MAX_BATCH_SIZE)
    async def __call__(self, inputs: List, requestIds: List):
        logfunc(self.logger, requestIds, "StepAudio_Enter")
        ret = self.exec_model(inputs)
        logfunc(self.logger, requestIds, "StepAudio_Exit")
        return ret

@serve.deployment
class StepEncode:
    def __init__(self):
        self.logger = make_logger(os.getpid(), "StepEncode", LOG_DIR)
        self.logger.setLevel(LOG_LEVEL)
        self.encoder = FlagModel(
               'BAAI/bge-small-en-v1.5',
               'cuda:0',
          )
        self.emb_dim = 384

    @serve.batch(max_batch_size=_MAX_BATCH_SIZE)
    async def __call__(self, inputs: List, requestIds: List):
        logfunc(self.logger, requestIds, "StepEncode_Enter")
        #is numpy array
        result =  self.encoder.encode(inputs)
        logfunc(self.logger, requestIds, "StepEncode_Exit")
        #TODO: test removing this and just returning result
        return [result[i] for i in range(len(result))]

@serve.deployment
class StepSearch:
    def __init__(self):
        self.logger = make_logger(os.getpid(), "StepSearch", LOG_DIR)
        self.logger.setLevel(LOG_LEVEL)
        self.index_dir = os.path.join(DATA_DIR, "msmarco_pq.index")
        self.cpu_index = faiss.read_index(self.index_dir)
        self.res = faiss.StandardGpuResources()
        self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)
        self.gpu_index.nprobe = 10
        self.topk = 5

    @serve.batch(max_batch_size=_MAX_BATCH_SIZE)
    async def __call__(self, inputs: List, requestIds: List):
        logfunc(self.logger, requestIds, "StepSearch_Enter")
        _, I = self.gpu_index.search(np.stack(inputs, axis=0), self.topk)
        #TODO: pull docs
        logfunc(self.logger, requestIds, "StepSearch_Exit")
        return I

@serve.deployment
class Ingress:
    def __init__(self, stepAudio: DeploymentHandle, stepEncode: DeploymentHandle, stepSearch: DeploymentHandle):
        self.logger = make_logger(os.getpid(), "Ingress", LOG_DIR)
        self.logger.setLevel(LOG_LEVEL)
        self.stepAudio = stepAudio
        self.stepEncode = stepEncode
        self.stepSearch = stepSearch

    async def __call__(self, http_request: Request):
        try:
            requestid = http_request.headers["x-requestid"]
            rid_obj = {"requestid": requestid}
            logfunc(self.logger, [rid_obj], "Ingress_Enter")
            #input = await http_request.body()
            #input = pickle.loads(input)
            input = await http_request.json()
            logfunc(self.logger, [rid_obj], "Ingress_Input_Decoded")
            inputArr = np.array(input)  
            logfunc(self.logger, [rid_obj], "Ingress_Numpy_Converted")
            stepaudio_output = self.stepAudio.remote(inputArr, rid_obj)
            stepencode_output = self.stepEncode.remote(stepaudio_output, rid_obj)
            stepsearch_output = self.stepSearch.remote(stepencode_output, rid_obj)
            
            return await stepsearch_output
        except Exception as e:
            print("\n\n\n\n ERROR:", e, "\n\n\n\n")
            traceback.print_exc()
            return ["error"]



def app(args: Dict[str, str]) -> Application:
    stepAudio = StepAudio.bind()
    stepEncode = StepEncode.bind()
    stepSearch = StepSearch.bind()
    return Ingress.bind(stepAudio=stepAudio, stepEncode=stepEncode, stepSearch=stepSearch)