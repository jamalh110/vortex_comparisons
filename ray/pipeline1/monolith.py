import random
import string
from typing import Dict, List
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

_MAX_BATCH_SIZE = 32
DATA_DIR="/mydata"

@serve.deployment
class Monolith:
    def __init__(self, experiment_name: str, index_name):
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

    @serve.batch(max_batch_size=_MAX_BATCH_SIZE)
    async def __call__(self, http_request: List[Request]):
        
        input_jsons = [await i.json() for i in http_request]
        
        bsize               = None

        image_paths = [i["img_path"] for i in input_jsons]
        question_ids = [i["question_id"] for i in input_jsons]
        questions = [i["question"] for i in input_jsons]

        text_sequences = [self.prepare_inputs(i)["text_sequence"] for i in input_jsons]
        pixel_values = self.process_images(image_paths)


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
        return ret


def app(args: Dict[str, str]) -> Application:
    experiment_name = args.get("experiment_name", "EVQA_train_split/")
    index_name = args.get("index_name", "EVQA_PreFLMR_ViT-L")
    handle: DeploymentHandle =  Monolith.bind(experiment_name=experiment_name, index_name=index_name)
    return handle