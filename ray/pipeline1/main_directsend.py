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
from flmr import FLMRConfig, FLMRQueryEncoderTokenizer, FLMRContextEncoderTokenizer, FLMRModelForRetrieval, FLMRTextModel, FLMRVisionModel, search_custom_collection, create_searcher
from transformers import AutoImageProcessor, BertConfig
import io
from PIL import Image
from easydict import EasyDict
from transformers.models.bert.modeling_bert import BertEncoder
from StepD_directsend import StepD
import random
import string
import numpy as np
import faulthandler
import traceback
import signal
import pickle
import logging

from utils import make_logger, logfunc
#TODO: Keep tensors in GPU instead of copying them to cpu in __call__ of every step? 
#TODO: test keeping tensors in gpu across ray returns and see if it is faster
#TODO: test calling step c from step b and see if it is faster
#TODO: test if batch works properly
_MAX_BATCH_SIZE = 32
_STEP_E_BATCH_SIZE = 32
DATA_DIR="/mydata"
LOG_DIR = "/users/jamalh11/raylogs"
#LOG_LEVEL = logging.INFO


@serve.deployment
class StepA:
    def __init__(self, loglevel):

        
        self.logger = make_logger(os.getpid(), "StepA", LOG_DIR)
        if(loglevel == "INFO"):
            self.logger.setLevel(logging.INFO)
        elif(loglevel == "CRITICAL"):
            self.logger.setLevel(logging.CRITICAL)
        self.checkpoint_path = f"{DATA_DIR}/PreFLMR_ViT-L"
        self.local_encoder_path = f'{DATA_DIR}/EVQA/models/models_step_A_query_text_encoder.pt'
        self.local_projection_path = f'{DATA_DIR}/EVQA/models/models_step_A_query_text_linear.pt'
        self.flmr_config                = None
        self.query_tokenizer            = None
        self.context_tokenizer          = None   
        self.query_text_encoder         = None 
        self.query_text_encoder_linear  = None
        self.device                     = 'cpu'
        self.skiplist = []

        self.load_model_cpu()
        #if(os.getenv('USE_GPU', 'False') == 'True'):
        self.load_model_gpu()
        

    def load_model_cpu(self):
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
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

        self.query_text_encoder = FLMRTextModel(self.flmr_config.text_config)
        self.query_text_encoder_linear = nn.Linear(self.flmr_config.text_config.hidden_size, self.flmr_config.dim, bias=False)

        try:
            self.query_text_encoder.load_state_dict(torch.load(self.local_encoder_path, weights_only=True))
            self.query_text_encoder_linear.load_state_dict(torch.load(self.local_projection_path, weights_only=True))
        except:
            print(f'Failed to load models checkpoint!!! \n Please check {self.local_encoder_path} or {self.local_projection_path}')

    def load_model_gpu(self):
        self.device = 'cuda'
        self.query_text_encoder_linear.to(self.device)
        self.query_text_encoder.to(self.device)

    def prepare_inputs(self, sample):
        sample = EasyDict(sample)

        module = EasyDict(
            {"type": "QuestionInput", "option": "default", "separation_tokens": {"start": "", "end": ""}}
        )

        instruction = sample.instruction.strip()
        if instruction[-1] != ":":
            instruction = instruction + ":"
        instruction = instruction.replace(":", self.flmr_config.mask_instruction_token)
        #random_instruction = random.choice(instructions)
        text_sequence = " ".join(
            [instruction]
            + [module.separation_tokens.start]
            + [sample.question]
            + [module.separation_tokens.end]
        )

        sample["text_sequence"] = text_sequence

        return sample

    def stepA_output(
        self,
        inputs: List[Dict[str, Any]],
    ):
        #self.logger.info(f"StepA_Prepare_Start {inputs[0]['requestid']}")
        input_text_sequence = []
        for input in inputs:
            input_text_sequence.append(input["text_sequence"])
        #self.logger.info(f"StepA_Prepare_End {inputs[0]['requestid']}")
        # query sentences: bsize of sentences
        # encoded_inputs      = self.query_tokenizer(input_text_sequence)
        # input_ids           = encoded_inputs['input_ids'].to(self.query_text_encoder.device)
        # attention_mask      = encoded_inputs['attention_mask'].to(self.query_text_encoder.device)

        input_ids = torch.stack([torch.from_numpy(i["input_ids"]) for i in inputs], dim=0).to(self.device)
        attention_mask = torch.stack([torch.from_numpy(i["attention_mask"]) for i in inputs], dim=0).to(self.device)
        
        text_encoder_outputs = self.query_text_encoder(input_ids=input_ids,attention_mask=attention_mask)
        text_encoder_hidden_states = text_encoder_outputs[0]
        text_embeddings = self.query_text_encoder_linear(text_encoder_hidden_states)

        # note, text_embeddings not masked yet here!!!!
        #print(text_embeddings, input_ids, text_encoder_hidden_states)
        #print("\n\n\n\n")
        #print("shape A", text_embeddings.shape)
        #print("shape size A", text_embeddings.shape[0])
        #print("\n\n\n\n")
        
        return {
            "text_embeddings": text_embeddings,
            "input_ids": input_ids,
            "text_encoder_hidden_states": text_encoder_hidden_states
        }
    
    @serve.batch(max_batch_size=_MAX_BATCH_SIZE)
    async def __call__(self, inputs: List[Dict[str, Any]]):
        #print("BATCH SIZE: ",len(inputs))
        #time.sleep(10)
        logfunc(self.logger, inputs, "StepA_Enter")
        output = self.stepA_output(inputs)
        text_embeddings = output['text_embeddings'].detach().cpu().numpy()
        input_ids = output['input_ids'].detach().cpu().numpy()
        text_encoder_hidden_states = output['text_encoder_hidden_states'].detach().cpu().numpy()

        results = []
        for i in range(text_embeddings.shape[0]):
            results.append({
                "text_embeddings": text_embeddings[i],
                "input_ids": input_ids[i],
                "text_encoder_hidden_states": text_encoder_hidden_states[i]
            })
        logfunc(self.logger, inputs, "StepA_Exit")
        return results
        #return self.stepA_output(inputs)


class FLMRMultiLayerPerceptron(nn.Module):
    """
    A simple multi-layer perceptron with an activation function. This can be used as the mapping network in the FLMR model.
    """

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

#TODO: batch?
@serve.deployment
class StepB:
    def __init__(self, loglevel):
        self.logger = make_logger(os.getpid(), "StepB", LOG_DIR)
        if(loglevel == "INFO"):
            self.logger.setLevel(logging.INFO)
        elif(loglevel == "CRITICAL"):
            self.logger.setLevel(logging.CRITICAL)
        self.checkpoint_path = f'{DATA_DIR}/PreFLMR_ViT-L'
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        self.local_encoder_path = f'{DATA_DIR}/EVQA/models/models_step_B_vision_encoder.pt'
        self.local_projection_path = f'{DATA_DIR}/EVQA/models/models_step_B_vision_projection.pt'
        self.image_processor = AutoImageProcessor.from_pretrained(f'{DATA_DIR}/clip-vit-large-patch14')
        self.device = 'cuda'

        self.query_vision_encoder = FLMRVisionModel(self.flmr_config.vision_config)
        self.query_vision_projection = FLMRMultiLayerPerceptron(
                (
                    self.flmr_config.vision_config.hidden_size,
                    (self.flmr_config.dim * self.flmr_config.mapping_network_prefix_length) // 2,
                    self.flmr_config.dim * self.flmr_config.mapping_network_prefix_length,
                )
            )
        self.query_vision_encoder.load_state_dict(torch.load(self.local_encoder_path, weights_only=False))
        self.query_vision_projection.load_state_dict(torch.load(self.local_projection_path, weights_only=False))
        self.query_vision_projection.cuda()
        self.query_vision_encoder.cuda()

    def StepB_output(self, pixel_values, requestid):
        #self.logger.info(f"StepB_Prepare_Start {requestid}")
        #print("here1")
        batch_size = pixel_values.shape[0]
        #self.logger.info(f"StepB_Prepare_End {requestid}")
        # Forward the vision encoder
        pixel_values = pixel_values.to(self.device)
        if len(pixel_values.shape) == 5:
            # Multiple ROIs are provided
            # merge the first two dimensions
            pixel_values = pixel_values.reshape(
                -1, pixel_values.shape[2], pixel_values.shape[3], pixel_values.shape[4]
            )
        
        #print("here2")
        with torch.no_grad():
            vision_encoder_outputs = self.query_vision_encoder(pixel_values, output_hidden_states=True)
            vision_embeddings = vision_encoder_outputs.last_hidden_state[:, 0]
            vision_embeddings = self.query_vision_projection(vision_embeddings)
            vision_embeddings = vision_embeddings.view(batch_size, -1, self.flmr_config.dim)
    
        vision_second_last_layer_hidden_states = vision_encoder_outputs.hidden_states[-2][:, 1:]
        
        return {"vision_embeddings": vision_embeddings, "vision_second_last_layer_hidden_states": vision_second_last_layer_hidden_states}
    
    @serve.batch(max_batch_size=_MAX_BATCH_SIZE)
    async def __call__(self, input: List[Dict[str, Any]]):
        rid = input[0]['requestid']
        logfunc(self.logger, input, "StepB_Enter")
        #print("BATCH SIZE: ",len(list_of_images))
        #time.sleep(10)
        pixel_values = []
        for i in input:
            pixel_values.append(torch.from_numpy(i['pixel_values']))
        output = self.StepB_output(torch.stack(pixel_values, dim=0), requestid='abcd')
        vision_embeddings = output['vision_embeddings'].detach().cpu().numpy()
        vision_second_last_layer_hidden_states = output['vision_second_last_layer_hidden_states'].detach().cpu().numpy()
        results = []
        for i in range(vision_embeddings.shape[0]):
            results.append({
                "vision_embeddings": vision_embeddings[i],
                "vision_second_last_layer_hidden_states": vision_second_last_layer_hidden_states[i]
            })
        logfunc(self.logger, input, "StepB_Exit")
        return results
        #return self.StepB_output(list_of_images)

@serve.deployment
class StepE:
    def __init__(self, experiment_name: str, index_name: str, loglevel):
        self.logger = make_logger(os.getpid(), "StepE", LOG_DIR)
        if(loglevel == "INFO"):
            self.logger.setLevel(logging.INFO)
        elif(loglevel == "CRITICAL"):
            self.logger.setLevel(logging.CRITICAL)
        self.searcher = None
        self.index_root_path        = f'{DATA_DIR}/EVQA/index/'
        self.index_experiment_name  = experiment_name
        self.index_name             = index_name
        self.load_searcher_gpu()

    def load_searcher_gpu(self):
        self.searcher = create_searcher(
            index_root_path=self.index_root_path,
            index_experiment_name=self.index_experiment_name,
            index_name=self.index_name,
            nbits=8, # number of bits in compression
            use_gpu=True, # break if set to False, see doc: https://docs.google.com/document/d/1KuWGWZrxURkVxDjFRy1Qnwsy7jDQb-RhlbUzm_A-tOs/edit?tab=t.0
        )

    def process_search(self, queries, query_embeddings, bsize):
        ranking = search_custom_collection(
            searcher=self.searcher,
            queries=queries,
            #query_embeddings=torch.Tensor(query_embeddings),
            query_embeddings=query_embeddings,
            num_document_to_retrieve=5, # how many documents to retrieve for each query
            centroid_search_batch_size=bsize,
        )
        return ranking.todict()
    @serve.batch(max_batch_size=_STEP_E_BATCH_SIZE)
    async def __call__(self, input1: List[Dict[str, Any]], input2: Any):
        logfunc(self.logger, input1, "StepE_Enter")
        #bsize = len(input)
        #print("BATCH SIZE: ",len(input))
        bsize = None
        queries = {}
        query_embeddings_list = []

        for i, j in zip(input1, input2):
            queries[i['question_id']] = i['text_sequence']
            query_embeddings_list.append(torch.from_numpy(j))

        query_embeddings = torch.stack(query_embeddings_list, dim=0).cuda()

        tim = time.time()
        output = self.process_search(queries, query_embeddings, bsize)
        print(f"Search took {time.time() - tim} seconds")

        ret = []
        for i in input1:
            ret.append(output[i['question_id']])

        logfunc(self.logger, input1, "StepE_Exit")
        return ret


@serve.deployment
class Ingress:
    def __init__(self, stepA: DeploymentHandle, stepB: DeploymentHandle, stepD: DeploymentHandle, stepE: DeploymentHandle, loglevel):
        self.logger = make_logger(os.getpid(), "Ingress", LOG_DIR)
        if(loglevel == "INFO"):
            self.logger.setLevel(logging.INFO)
        elif(loglevel == "CRITICAL"):
            self.logger.setLevel(logging.CRITICAL)
        self.stepA = stepA
        self.stepB = stepB
        self.stepD = stepD
        self.stepE = stepE
        # faulthandler.enable()
    
        # def dump_stack(signum, frame):
        #     faulthandler.dump_traceback()

        #signal.signal(signal.SIGUSR1, dump_stack)


    async def __call__(self, http_request: Request):
        try:
            #requestid = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(6))
            requestid = http_request.headers["x-requestid"]
            logfunc(self.logger, [{"requestid": requestid}], "Ingress_Enter")
            #input = await http_request.json()
            #time1 = time.time()
            input = await http_request.body()
            #time2 = time.time()
            #print(f"Time to get input: {time2 - time1} seconds")
            #time3 = time.time()
            input = pickle.loads(input)
            #time4 = time.time()
            #print(f"Time to unpickle input: {time4 - time3} seconds")
            input['requestid'] = requestid
            
            

            image = {"pixel_values": input['pixel_values']}

            del input['pixel_values']
            inputref = ray.put(input)

            image['requestid'] = requestid

            #stepA_output = self.stepA.remote({"requestid": requestid, "text_sequence": input['text_sequence'], "input_ids": input['input_ids'], "attention_mask": input['attention_mask']})
            stepA_output = self.stepA.remote(inputref)
            stepB_output = self.stepB.remote(image)    
            #output_b_raw = await stepB_output
            #output_b_raw['requestid'] = requestid
            #stepC_output = self.stepC.remote(output_b_raw)
            #output_a_raw = await stepA_output
            #output_c_raw = await stepC_output
            #TODO: test if passing in the objectrefs is faster than awaitng and sending

            # stepD_output = self.stepD.remote({
            #     "input_ids": output_a_raw['input_ids'],
            #     "text_embeddings": output_a_raw['text_embeddings'],
            #     "text_encoder_hidden_states": output_a_raw['text_encoder_hidden_states'],
            #     "vision_embeddings": output_b_raw['vision_embeddings'],
            #     "vision_second_last_layer_hidden_states": output_b_raw['vision_second_last_layer_hidden_states'],
            #     'requestid': requestid
            # })
            stepD_output = self.stepD.remote(stepA_output, stepB_output, requestid)
            #output_d_raw = await stepD_output
            #stepE_output = self.stepE.remote({"question_id": input['question_id'], "question": input['text_sequence'], "query_embeddings": stepD_output, "requestid": requestid})
            #stepE_output = self.stepE.remote({"question_id": input['question_id'], "question": input['text_sequence'], "requestid": requestid}, stepD_output)
            stepE_output = self.stepE.remote(inputref, stepD_output)
            output = await stepE_output
            logfunc(self.logger, [{"requestid": requestid}], "Ingress_Exit")
            return output
        except Exception as e:
            print("\n\n\n\n ERROR:", e, "\n\n\n\n")
            traceback.print_exc()
            return ["error"]


# stepA = StepA.bind()
# stepB = StepB.bind()
# stepC = StepC.bind()
# stepD = StepD.bind()
# app = Ingress.bind(stepA=stepA, stepB=stepB, stepC=stepC, stepD=stepD)

def app(args: Dict[str, str]) -> Application:
    experiment_name = args.get("experiment_name", "EVQA_train_split/")
    index_name = args.get("index_name", "EVQA_PreFLMR_ViT-L")
    loglevel = args.get("loglevel", "CRITICAL")
    stepA = StepA.bind(loglevel=loglevel)
    stepB = StepB.bind(loglevel=loglevel)
    stepD = StepD.bind(loglevel=loglevel)
    stepE = StepE.bind(experiment_name=experiment_name, index_name=index_name, loglevel=loglevel)
    return Ingress.bind(stepA=stepA, stepB=stepB, stepD=stepD, stepE=stepE, loglevel=loglevel) 

# def app(args: Dict[str, str]) -> Application:
#     stepA = StepA.bind()
#     return Ingress.bind(stepA=stepA)
