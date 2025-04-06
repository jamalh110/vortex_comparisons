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
        return text_list

    @serve.batch(max_batch_size=_MAX_BATCH_SIZE)
    async def __call__(self, inputs: List, requestIds: List):
        logfunc(self.logger, requestIds, "StepAudio_Enter")
        ret = self.exec_model(inputs)
        logfunc(self.logger, requestIds, "StepAudio_Exit")
        return ret



@serve.deployment
class Ingress:
    def __init__(self, stepAudio: DeploymentHandle):
        self.logger = make_logger(os.getpid(), "Ingress", LOG_DIR)
        self.logger.setLevel(LOG_LEVEL)
        self.stepAudio = stepAudio
       


    async def __call__(self, http_request: Request):
        try:
            requestid = http_request.headers["x-requestid"]
            logfunc(self.logger, [{"requestid": requestid}], "Ingress_Enter")
            #input = await http_request.body()
            #input = pickle.loads(input)
            input = await http_request.json()
            logfunc(self.logger, [{"requestid": requestid}], "Ingress_Input_Decoded")
            inputArr = np.array(input)  
            logfunc(self.logger, [{"requestid": requestid}], "Ingress_Numpy_Converted")
            output = await self.stepAudio.remote(inputArr, {"requestid": requestid})
            
            return output
        except Exception as e:
            print("\n\n\n\n ERROR:", e, "\n\n\n\n")
            traceback.print_exc()
            return ["error"]



def app(args: Dict[str, str]) -> Application:
    stepAudio = StepAudio.bind()
    return Ingress.bind(stepAudio=stepAudio)