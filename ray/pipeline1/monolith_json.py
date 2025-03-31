import asyncio
import logging
import pickle
import random
import string
import time
from typing import Any, Dict, List
from starlette.requests import Request
from ray.serve.handle import DeploymentHandle
from ray.serve import Application
from ray import serve
import ray

import torch
from easydict import EasyDict
import numpy as np
from transformers import AutoImageProcessor
from flmr import (
    FLMRModelForRetrieval,
    FLMRQueryEncoderTokenizer,
    FLMRContextEncoderTokenizer,
    FLMRConfig,
)
from PIL import Image
from flmr import create_searcher, search_custom_collection
from utils import make_logger, logfunc
import os
import multiprocessing as mp
from functools import partial

_MAX_BATCH_SIZE = 16
DATA_DIR="/mydata"
LOG_DIR = "/users/jamalh11/raylogs"

def process_input(input_data):
    return pickle.loads(input_data)

@serve.deployment
class Monolith:
    def __init__(self, experiment_name: str, index_name):
        self.logger = make_logger(os.getpid(), "Monolith", LOG_DIR)
        self.logger.setLevel(logging.INFO)

        self.index_root_path        = f'{DATA_DIR}/EVQA/index/'
        self.index_name             = index_name
        self.index_experiment_name  = experiment_name
        self.checkpoint_path        = f'{DATA_DIR}/PreFLMR_ViT-L'
        self.image_processor_name   = f'{DATA_DIR}/clip-vit-large-patch14'
        self.Ks                     = [1]
        self.use_gpu                = True
        self.nbits                  = 8
        self.query_batch_size       = 8
        self.flmr_config            = None
        self.query_tokenizer        = None
        self.context_tokenizer      = None
        self.flmr_model             = None
        self.image_processor        = None
        self.searcher               = None
        self.device                 = "cuda" if self.use_gpu else "cpu"

        self.load_model_cpu()
        self.load_model_gpu()

        self.process_pool = mp.Pool(processes=32)

    def load_model_cpu(self):
        self.flmr_config = FLMRConfig.from_pretrained(self.checkpoint_path)
        self.query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(self.checkpoint_path,
                                                                        text_config=self.flmr_config.text_config,
                                                                        subfolder="query_tokenizer")
        self.context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(self.checkpoint_path,
                                                                        text_config=self.flmr_config.text_config,
                                                                        subfolder="context_tokenizer")
        self.flmr_model = FLMRModelForRetrieval.from_pretrained(
            self.checkpoint_path,
            query_tokenizer=self.query_tokenizer,
            context_tokenizer=self.context_tokenizer,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(self.image_processor_name)
        
        
    def load_model_gpu(self):
        self.flmr_model = self.flmr_model.to("cuda")
        self.searcher = create_searcher(
            index_root_path=self.index_root_path,
            index_experiment_name=self.index_experiment_name,
            index_name=self.index_name,
            nbits=self.nbits, # number of bits in compression
            use_gpu=True, # break if set to False, see doc: https://docs.google.com/document/d/1KuWGWZrxURkVxDjFRy1Qnwsy7jDQb-RhlbUzm_A-tOs/edit?tab=t.0
        )

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
    
    def process_images(self, list_of_images):
        pixel_values = []
        # for imgbytes in list_of_images:
        #     img = Image.open(io.BytesIO(imgbytes)).convert("RGB")
        #     encoded = self.image_processor(img, return_tensors="pt")
        #     pixel_values.append(encoded.pixel_values)
    
        for img_path in list_of_images:
            if img_path is None:
                image = Image.new("RGB", (336, 336), color='black')
            else:
                image = Image.open(img_path).convert("RGB")
            encoded = self.image_processor(image, return_tensors="pt")
            pixel_values.append(encoded.pixel_values)
        pixel_values = torch.stack(pixel_values, dim=0)
        #print("here1")
        batch_size = pixel_values.shape[0]
        # Forward the vision encoder
        pixel_values = pixel_values.to(self.device)
        return pixel_values
    
    
    # async def process_all_inputs(self, http_request):
         
    #     # First gather all inputs asynchronously
    #     start = time.time()
    #     inputs = [await i.body() for i in http_request]
    #     print(f"Time to get bodies: {time.time()-start} batch {len(inputs)}")
    #     # Then process them in parallel
    #     # with mp.Pool(processes=32) as pool:
    #     #     input_jsons = pool.map(process_input, inputs)
        
    #     # return input_jsons

    #     loop = asyncio.get_event_loop()
    #     input_jsons = await loop.run_in_executor(
    #         None, 
    #         lambda: list(self.process_pool.map(process_input, inputs))
    #     )
    
    #     return input_jsons

    @serve.batch(max_batch_size=_MAX_BATCH_SIZE)
    async def __call__(self, input_jsons: Any):

        #input_jsons = [await i.json() for i in http_request]
        #start = time.time()
        #inputs = [await i.body() for i in http_request]
        #input_jsons = [pickle.loads(i) for i in inputs]
        #input_jsons = await self.process_all_inputs(http_request)

        #print(f"Time to load images: {time.time()-start} batch {len(input_jsons)}")
        #input_jsons = ray.get(input_jsons_refs)
        logfunc(self.logger, input_jsons, "Monolith_Enter")

        bsize               = None

        #image_paths = [i["img_path"] for i in input_jsons]
        question_ids = [i["question_id"] for i in input_jsons]
        questions = [i["question"] for i in input_jsons]

        #text_sequences = [self.prepare_inputs(i)["text_sequence"] for i in input_jsons]
        #pixel_values = self.process_images(image_paths)
        text_sequences = [i["text_sequence"] for i in input_jsons]

        start = time.time()
        #pixel_values = torch.stack([torch.tensor(i["pixel_values"]) for i in input_jsons], dim=0).to(self.device)
        pixel_values = torch.stack([torch.from_numpy(i["pixel_values"]) for i in input_jsons], dim=0).to(self.device)
        print(f"Time to load images: {time.time()-start} batch {len(input_jsons)}")

        encoding = self.query_tokenizer(text_sequences)
        input_ids = torch.LongTensor(encoding["input_ids"]).to(self.device)
        attention_mask = torch.LongTensor(encoding["attention_mask"]).to(self.device)
        query_input = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
        }

        query_embeddings = self.flmr_model.query(**query_input).late_interaction_output.to(self.device)
        #query_embeddings = query_embeddings.detach().cpu()

        queries = {
            question_id: question for question_id, question in zip(question_ids, questions)
        } 

        ranking = search_custom_collection(
            searcher=self.searcher,
            queries= queries,
            query_embeddings=query_embeddings,
            num_document_to_retrieve=5, # how many documents to retrieve for each query
            centroid_search_batch_size=bsize,
        )
        ranking = ranking.todict()
        ret = []
        for id in question_ids:
            ret.append(ranking[id])
        #print(time.time())
        logfunc(self.logger, input_jsons, "Monolith_Exit")
        #print(time.time())
        return ret

@serve.deployment
class Ingress:
    def __init__(self, monolith: DeploymentHandle):
        self.monolith = monolith
        self.logger = make_logger(os.getpid(), "Ingress", LOG_DIR)
        self.logger.setLevel(logging.INFO)

        
    async def __call__(self, http_request: Request):
        try:
            requestid = http_request.headers["x-requestid"]
            
            input = await http_request.json()
            input['requestid'] = requestid
            logfunc(self.logger, [{"requestid": requestid}], "Ingress_Enter")
            input['pixel_values'] = np.array(input['pixel_values'])

            ref = ray.put(input)
            ret = await self.monolith.remote(ref)
            del ref
            
            
            #ret = await self.monolith.remote(input)

            logfunc(self.logger, [{"requestid": requestid}], "Ingress_Exit")
            return ret
        
        except Exception as e:
            print("\n\n\n\n ERROR:", e, "\n\n\n\n")
            return ["error"]



def app(args: Dict[str, str]) -> Application:
    experiment_name = args.get("experiment_name", "EVQA_train_split/")
    index_name = args.get("index_name", "EVQA_PreFLMR_ViT-L")
    handle: DeploymentHandle =  Monolith.bind(experiment_name=experiment_name, index_name=index_name)
    ingress_handle: DeploymentHandle = Ingress.bind(monolith=handle)
    return ingress_handle