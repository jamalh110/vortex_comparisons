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
from transformers import AutoModelForSequenceClassification, AutoTokenizer
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
from nemo.collections.tts.models import FastPitchModel, HifiGanModel
from torch.nn.utils.rnn import pad_sequence
from transformers import BartTokenizer, BartForSequenceClassification

from funasr.utils.postprocess_utils import rich_transcription_postprocess
from funasr.utils.load_utils import extract_fbank
import sys
from FlagEmbedding import BGEM3FlagModel, FlagModel
import faiss

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "SenseVoice"))
from SenseVoice.utils.frontend import WavFrontend, WavFrontendOnline
from SenseVoice.model import SenseVoiceSmall


#TODO: make the TTS shapes between smart monoloth and monolith and micro the same (not actually important, but it's bothering me)
#_MAX_BATCH_SIZE = 4
DATA_DIR="/mydata/msmarco"
LOG_DIR = "/users/jamalh11/raylogs"
LOG_LEVEL = logging.INFO

class StepAudioModel:
    def __init__(self):
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

class StepEncodeModel:
    def __init__(self):
        self.encoder = FlagModel(
                'BAAI/bge-small-en-v1.5',
                'cuda:0',
            )
        self.emb_dim = 384

class StepSearchModel:
    def __init__(self):
        self.index_dir = os.path.join(DATA_DIR, "msmarco_pq.index")
        self.cpu_index = faiss.read_index(self.index_dir)
        self.res = faiss.StandardGpuResources()
        self.gpu_index = faiss.index_cpu_to_gpu(self.res, 0, self.cpu_index)
        self.gpu_index.nprobe = 10
        self.topk = 1
        self.docs = []
        with open(os.path.join(DATA_DIR, "msmarco_3_clusters/doc_list.pkl"), "rb") as f:
            self.docs = pickle.load(f)
        
        self.encodeModel = StepEncodeModel()
        print("Loaded documents and index")
    def search(self, inputs):
        encoded_inputs = self.encodeModel.encoder.encode(inputs)
        _, I = self.gpu_index.search(encoded_inputs, self.topk)
        #TODO: pull docs
        ret_docs = []
        for indexes in I:
            ret_docs.append([self.docs[int(i)] for i in indexes])
        return ret_docs
    
class StepTTSModel:
    def __init__(self):
        self.fastpitchname = "nvidia/tts_en_fastpitch"
        self.hifiganname = "nvidia/tts_hifigan"
        self.device = torch.device("cuda")
        self.fastpitch = FastPitchModel.from_pretrained(self.fastpitchname).to(self.device).eval()
        self.hifigan = HifiGanModel.from_pretrained(model_name=self.hifiganname).to(self.device).eval()


    def model_exec(self, texts: list[str]) -> np.ndarray:
        with torch.no_grad():
            token_list = [self.fastpitch.parse(text).squeeze(0) for text in texts]
            tokens = pad_sequence(token_list, batch_first=True).to(self.device)
            spectrograms = self.fastpitch.generate_spectrogram(tokens=tokens)
            audios = self.hifigan.convert_spectrogram_to_audio(spec=spectrograms)

        np_audios = audios.cpu().numpy()
        return np_audios
    
class StepToxCheckModel:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = "cuda:0"
        self.model_name = "facebook/bart-large-mnli"
        self.hypothesis = "harmful."
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForSequenceClassification.from_pretrained(self.model_name).to(self.device)

    def model_exec(self, batch_premise: list[str]) -> np.ndarray:
        '''
        batch_premise: list of text strings
        
        return: classified type ID
        0 hate speech, 1 offensive language, 2 neither
        '''
        inputs = self.tokenizer(batch_premise,
                       [self.hypothesis] * len(batch_premise),
                       return_tensors='pt', padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            result = self.model(**inputs)
        logits = result.logits
        entail_contradiction_logits = logits[:, [0, 2]]  # entailment = index 2
        probs = entail_contradiction_logits.softmax(dim=1)
        true_probs = probs[:, 1] * 100  # entailment probability
        return true_probs.tolist()

@serve.deployment
class Monolith:
    def __init__(self):
        self.logger = make_logger(os.getpid(), "Monolith", LOG_DIR)
        self.logger.setLevel(LOG_LEVEL)
        self.stepAudioModel = StepAudioModel()
        self.stepSearchModel = StepSearchModel()
        self.stepTTSModel = StepTTSModel()
        self.stepToxCheckModel = StepToxCheckModel()


    @serve.batch(max_batch_size=2)
    async def __call__(self, inputs: List, requestIds: List):
        logfunc(self.logger, requestIds, "Monolith_Enter")
        res = []
        stepAudioOutput = self.stepAudioModel.exec_model(inputs)
        stepSearchOutput = self.stepSearchModel.search(stepAudioOutput)

        flattened_search_output = [item for sublist in stepSearchOutput for item in sublist]
        stepTTSOutput = self.stepTTSModel.model_exec(flattened_search_output)
        stepToxCheckOutput = self.stepToxCheckModel.model_exec(flattened_search_output)
        # Reassemble outputs into the structure of stepSearchOutput
        tts_nested_output = []
        tox_nested_output = []
        index = 0
        for sublist in stepSearchOutput:
            n = len(sublist)
            tts_nested_output.append(stepTTSOutput[index:index+n])
            tox_nested_output.append(stepToxCheckOutput[index:index+n])
            index += n

        for i in range(len(stepSearchOutput)):
            res.append((stepAudioOutput[i], stepSearchOutput[i], tts_nested_output[i].shape, tox_nested_output[i]))

        logfunc(self.logger, requestIds, "Monolith_Exit")
        return res

@serve.deployment
class SmartMonolith:
    def __init__(self):
        self.logger = make_logger(os.getpid(), "SmartMonolith", LOG_DIR)
        self.logger.setLevel(LOG_LEVEL)
        self.stepAudioModel = StepAudioModel()
        self.stepSearchModel = StepSearchModel()
        self.stepTTSModel = StepTTSModel()
        self.stepToxCheckModel = StepToxCheckModel()
        self.batch_size_tts = 1
        self.batch_size_toxcheck = 4


    @serve.batch(max_batch_size=8)
    async def __call__(self, inputs: List, requestIds: List):
        logfunc(self.logger, requestIds, "SmartMonolith_Enter")
        res = []
        stepAudioOutput = self.stepAudioModel.exec_model(inputs)
        stepSearchOutput = self.stepSearchModel.search(stepAudioOutput)
        
        flattened_search_output = [item for sublist in stepSearchOutput for item in sublist]
        
        stepTTSOutput = []
        for i in range(0, len(flattened_search_output), self.batch_size_tts):
            batch = flattened_search_output[i:i+self.batch_size_tts]
            batch_result = self.stepTTSModel.model_exec(batch)
            stepTTSOutput.extend(batch_result)

        stepToxCheckOutput = []
        for i in range(0, len(flattened_search_output), self.batch_size_toxcheck):
            batch = flattened_search_output[i:i+self.batch_size_toxcheck]
            batch_result = self.stepToxCheckModel.model_exec(batch)
            stepToxCheckOutput.extend(batch_result)

        # Reassemble outputs into the structure of stepSearchOutput
        tts_nested_output = []
        tox_nested_output = []
        index = 0
        for sublist in stepSearchOutput:
            n = len(sublist)
            tts_nested_output.append(stepTTSOutput[index:index+n])
            tox_nested_output.append(stepToxCheckOutput[index:index+n])
            index += n

        for i in range(len(stepSearchOutput)):
            #not the same audio shapes but essentially the same
            res.append((stepAudioOutput[i], stepSearchOutput[i], [audio.shape for audio in tts_nested_output[i]], tox_nested_output[i]))

        logfunc(self.logger, requestIds, "SmartMonolith_Exit")
        return res
    
@serve.deployment
class StepAudio:
    def __init__(self):
        self.logger = make_logger(os.getpid(), "StepAudio", LOG_DIR)
        self.logger.setLevel(LOG_LEVEL)
        self.model = StepAudioModel()
        
    @serve.batch(max_batch_size=16)
    async def __call__(self, inputs: List, requestIds: List):
        logfunc(self.logger, requestIds, "StepAudio_Enter")
        #TODO: check if need to numpy stack?
        ret = self.model.exec_model(inputs)
        logfunc(self.logger, requestIds, "StepAudio_Exit")
        return ret

@serve.deployment
class StepSearch:
    def __init__(self):
        self.logger = make_logger(os.getpid(), "StepSearch", LOG_DIR)
        self.logger.setLevel(LOG_LEVEL)
        self.model = StepSearchModel()
    @serve.batch(max_batch_size=16)
    async def __call__(self, inputs: List, requestIds: List):
        logfunc(self.logger, requestIds, "StepSearch_Enter")
        res = self.model.search(inputs)
        logfunc(self.logger, requestIds, "StepSearch_Exit")
        return res
    
@serve.deployment
class StepTTS:
    def __init__(self):
        self.logger = make_logger(os.getpid(), "StepTTS", LOG_DIR)
        self.logger.setLevel(LOG_LEVEL)
        self.model = StepTTSModel()

    @serve.batch(max_batch_size=1)
    async def __call__(self, inputs: Any, requestId: Any):
        logfunc(self.logger, requestId, "StepTTS_Enter")
        inputs_flattened = [item for sublist in inputs for item in sublist]
        result =  self.model.model_exec(inputs_flattened)

        nested_output = []
        index = 0
        for sublist in inputs:
            n = len(sublist)
            nested_output.append(result[index:index+n])
            index += n

        logfunc(self.logger, requestId, "StepTTS_Exit")
        #don't transfer the whole thing, just return the shape
        return [item.shape for item in nested_output]

@serve.deployment
class StepToxCheck:
    def __init__(self):
        self.logger = make_logger(os.getpid(), "StepToxCheck", LOG_DIR)
        self.logger.setLevel(LOG_LEVEL)
        self.model = StepToxCheckModel()
    
    @serve.batch(max_batch_size=4)
    async def __call__(self, inputs: Any, requestId: Any):
        logfunc(self.logger, requestId, "StepToxCheck_Enter")
        #TODO: flatten and unflatten
        inputs_flattened = [item for sublist in inputs for item in sublist]
        result = self.model.model_exec(inputs_flattened)
        # Reassemble outputs into the structure of inputs
        nested_output = []
        index = 0
        for sublist in inputs:
            n = len(sublist)
            nested_output.append(result[index:index+n])
            index += n

        logfunc(self.logger, requestId, "StepToxCheck_Exit")
        return nested_output

@serve.deployment
class Ingress:
    def __init__(self, stepAudio: DeploymentHandle, stepSearch: DeploymentHandle, stepTTS: DeploymentHandle, stepToxCheck: DeploymentHandle):
        self.logger = make_logger(os.getpid(), "Ingress", LOG_DIR)
        self.logger.setLevel(LOG_LEVEL)
        self.stepAudio = stepAudio
        self.stepSearch = stepSearch
        self.stepTTS = stepTTS
        self.stepToxCheck = stepToxCheck

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
            #for some reason, the next steps hang without waiting for object ref here. 
            #TODO: figure out why. Test on a simplfied pipeline and see if sending a deploymentresponse to two steps in a row causes issues
            stepsearch_output = await self.stepSearch.remote(stepaudio_output, rid_obj)._to_object_ref()
            #stepsearch_output = self.stepSearch.remote(stepencode_output, rid_obj)
            stepTTS_output = self.stepTTS.remote(stepsearch_output, rid_obj)
            steptoxcheck_output = self.stepToxCheck.remote(stepsearch_output, rid_obj)
            res = await stepaudio_output, await stepsearch_output, await stepTTS_output, await steptoxcheck_output
            logfunc(self.logger, [rid_obj], "Ingress_Exit")
            return res
        except Exception as e:
            print("\n\n\n\n ERROR:", e, "\n\n\n\n")
            traceback.print_exc()
            return ["error"]

@serve.deployment
class Ingress_Mono:
    def __init__(self, monolith: DeploymentHandle):
        self.logger = make_logger(os.getpid(), "Ingress_Mono", LOG_DIR)
        self.logger.setLevel(LOG_LEVEL)
        self.monolith = monolith

    async def __call__(self, http_request: Request):
        try:
            requestid = http_request.headers["x-requestid"]
            rid_obj = {"requestid": requestid}
            logfunc(self.logger, [rid_obj], "Ingress_Mono_Enter")
            input = await http_request.json()
            logfunc(self.logger, [rid_obj], "Ingress_Mono_Input_Decoded")
            inputArr = np.array(input)  
            logfunc(self.logger, [rid_obj], "Ingress_Mono_Numpy_Converted")
            mono_output = self.monolith.remote(inputArr, rid_obj)
            res = await mono_output
            logfunc(self.logger, [rid_obj], "Ingress_Mono_Exit")
            return res
        except Exception as e:
            print("\n\n\n\n ERROR:", e, "\n\n\n\n")
            traceback.print_exc()
            return ["error"]



def app(args: Dict[str, str]) -> Application:
    stepAudio = StepAudio.bind()
    stepSearch = StepSearch.bind()
    stepTTS = StepTTS.bind()
    stepToxCheck = StepToxCheck.bind()
    return Ingress.bind(stepAudio=stepAudio, stepSearch=stepSearch, stepTTS=stepTTS, stepToxCheck=stepToxCheck)

def app_mono(args: Dict[str, str]) -> Application:
    monolith = Monolith.bind()
    return Ingress_Mono.bind(monolith=monolith)

def app_mono_smart(args: Dict[str, str]) -> Application:
    monolith = SmartMonolith.bind()
    return Ingress_Mono.bind(monolith=monolith)