import json
import pickle
import random
import string
import sys
import threading
import time
import requests
from datasets import load_dataset
import os 
from PIL import Image
from tqdm import tqdm
from utils import make_logger
from easydict import EasyDict
import torch
from flmr import (
    FLMRConfig,
    FLMRQueryEncoderTokenizer,
)
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
import aiohttp
import asyncio
import numpy as np
from types import ModuleType, FunctionType
from gc import get_referents

LOGGING_DIR = "/users/jamalh11/raylogs"
DATA_DIR = "/mydata"

hosts = [
        "http://10.10.1.13:8000",
        "http://10.10.1.12:8000",
        "http://10.10.1.11:8000",
        "http://10.10.1.10:8000",
        "http://10.10.1.9:8000",
        "http://10.10.1.8:8000"
    ]
session = None

def get_random_host():
        """Return a randomly selected host from the list."""
        return random.choice(hosts)

def add_path_prefix_in_img_path(example, prefix):
    if example["img_path"] != None:
        example["img_path"] = os.path.join(prefix, example["img_path"])
    return example

def process_image(example):
    img_path = example["img_path"]
    if img_path is None:
        image = Image.new("RGB", (336, 336), color='black')
    else:
        image = Image.open(img_path).convert("RGB")
    example["imagebytes"] = image.tobytes()
    return example
    

def prepare_inputs(sample, config):
    sample = EasyDict(sample)

    module = EasyDict(
        {"type": "QuestionInput", "option": "default", "separation_tokens": {"start": "", "end": ""}}
    )

    instruction = sample.instruction.strip()
    if instruction[-1] != ":":
        instruction = instruction + ":"
    instruction = instruction.replace(":", config.mask_instruction_token)
    #random_instruction = random.choice(instructions)
    text_sequence = " ".join(
        [instruction]
        + [module.separation_tokens.start]
        + [sample.question]
        + [module.separation_tokens.end]
    )

    sample["text_sequence"] = text_sequence

    return sample

def tokenize_inputs(examples, query_tokenizer, image_processor):
    encoding = query_tokenizer(examples["text_sequence"])
    examples["input_ids"] = encoding["input_ids"]
    examples["attention_mask"] = encoding["attention_mask"]

    pixel_values = []
    for img_path in examples["img_path"]:

        if img_path is None:
            image = Image.new("RGB", (336, 336), color='black')
        else:
            image = Image.open(img_path).convert("RGB")
        
        encoded = image_processor(image, return_tensors="pt")
        pixel_values.append(encoded.pixel_values)

    pixel_values = torch.stack(pixel_values, dim=0)
    examples["pixel_values"] = pixel_values
    return examples

def convert_to_numpy(example):
    #import torch  # ensure torch is imported in case it's not in the local scope
    #print("shape", example["pixel_values"].shape,example["attention_mask"].shape )
    if isinstance(example["input_ids"], torch.Tensor):
        example["input_ids"] = example["input_ids"][0].numpy().tolist()
    if isinstance(example["attention_mask"], torch.Tensor):
        example["attention_mask"] = example["attention_mask"][0].numpy().tolist()
    if isinstance(example["pixel_values"], torch.Tensor):
        example["pixel_values"] = example["pixel_values"][0].numpy().tolist()
    return example
def convert_to_numpy_map(example):
    #print(example['pixel_values'])
    example["input_ids"] = example["input_ids"].numpy().tolist()
    example["attention_mask"] = example["attention_mask"].numpy().tolist()
    example["pixel_values"] = example["pixel_values"].numpy().tolist()
    return example

def request_task(url, data_bytes, requestid):
    headers = {'Content-Type': 'application/octet-stream', "x-requestid": requestid}
    response = requests.post(url, data=data_bytes, headers=headers)

def request_task_sync(url, data_bytes, requestid):
    headers = {'Content-Type': 'application/octet-stream', "x-requestid": requestid}
    response = requests.post(url, data=data_bytes, headers=headers)

    return response

def fire_and_forget(url, data_bytes, requestid):
    threading.Thread(target=request_task, args=(url, data_bytes, requestid)).start()
    
async def async_request_task(url, data_bytes):
    """Asynchronous function to send a request"""
    global session
    if session is None:
        session = aiohttp.ClientSession()
    
    headers = {'Content-Type': 'application/octet-stream'}
    try:
        async with session.post(url, data=data_bytes, headers=headers) as response:
            # You can await the response if you need it
            # result = await response.json()
            pass
    except Exception as e:
        print(f"Error in async request: {e}")

def fire_and_forget2(url, data_bytes):
    """Non-blocking function to fire a request without waiting for response"""
    # Get the current event loop or create a new one
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no event loop exists in this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Schedule the coroutine to run soon and return immediately
    asyncio.create_task(async_request_task(url, data_bytes))

BLACKLIST = type, ModuleType, FunctionType
def getsize(obj):
    """sum size of object & members."""
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size

if __name__ == "__main__":
    image_root_dir = f"{DATA_DIR}/EVQA"
    use_split = "train"
    ds_dir = f"{DATA_DIR}/EVQA/EVQA_data"
    p_ds_dir = f"{DATA_DIR}/EVQA/EVQA_passages"

    logger = make_logger(os.getpid(), "Client", LOGGING_DIR)
    image_processor_name = '/mydata/clip-vit-large-patch14'
    checkpoint_path = '/mydata/PreFLMR_ViT-L'
    flmr_config = FLMRConfig.from_pretrained(checkpoint_path)
    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(checkpoint_path,
                                                                    text_config=flmr_config.text_config,
                                                                    subfolder="query_tokenizer")
    image_processor = AutoImageProcessor.from_pretrained(image_processor_name)

    ds = load_dataset('parquet', data_files ={  
                                                'train' : ds_dir + '/train-00000-of-00001.parquet',
                                                'test'  : ds_dir + '/test-00000-of-00001-2.parquet',
                                                })[use_split].select(range(0, 6000))



    ds = ds.map(add_path_prefix_in_img_path, fn_kwargs={"prefix": image_root_dir})
    ds = ds.map(prepare_inputs, fn_kwargs={"config": flmr_config})
    #print("tokenizing inputs")
    ds = ds.map(
        tokenize_inputs,
        fn_kwargs={"query_tokenizer": query_tokenizer, "image_processor": image_processor},
        batched=True,
        batch_size=16,
        num_proc=1,
    )
    


    #map ds to make input_ids, attention_mask, and pixel_values numpy arrays instead of tensors


    print("done tokenizing")
    #ds = ds.map(convert_to_numpy_map)
    # ds.set_format(
    #     type="torch", 
    #     columns=["input_ids", "attention_mask", "pixel_values", "text_sequence", "question_id", "question", "pos_item_ids"]
    # )


    # Create a DataLoader for sequential access with prefetching
    loader = DataLoader(
        ds, 
        batch_size=1, 
        shuffle=False, 
        num_workers=1,      # Use multiple workers to prefetch batches in parallel
        prefetch_factor=2,   # How many batches each worker preloads (can adjust based on your system)
        pin_memory=True      # Optionally, if you are transferring to GPU later
    )
    print("DS LEN", len(ds))
    #ds = ds.map(process_image)

    #print(ds[0])
    
    def onecall():
        data = ds[0]
        data_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        
        headers = {'Content-Type': 'application/octet-stream'}
        totaltimestart = time.time()
        response = requests.post("http://127.0.0.1:8000/", 
                            data=data_bytes,
                            headers=headers)
        output = response.json()
        print("TOTAL TIME:", time.time()-totaltimestart)
        print(output)
    
    #onecall()
    

    

    #exit(0)
    #print(ds['question_id'])
    #exit(0)
    nqueries = 6000
    max_retries = 3
    answers = []
    bytes_to_send = []
    file_path = "/mydata/ds_test.pkl"
    if not os.path.exists(file_path):
        for i in tqdm(range(nqueries), desc="Preparing bytes to send"):
            data = ds[i]
            requestid = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
            data['requestid'] = requestid
            datatosend = {
                #"question": data["question"],
                "question_id": data["question_id"],
                "text_sequence": data["text_sequence"],
                "pixel_values": (np.array(data["pixel_values"])),
                "input_ids": (np.array(data["input_ids"])),
                "attention_mask": (np.array(data["attention_mask"])),
                "requestid": requestid
            }
            data_bytes = pickle.dumps(datatosend, protocol=pickle.HIGHEST_PROTOCOL)
            bytes_to_send.append((data_bytes, requestid))

            # data = ds[i]
            # requestid = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
            # data['requestid'] = requestid
            # data_bytes = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
            # bytes_to_send.append((data_bytes, requestid))
        
        with open(file_path, "wb") as file:
            pickle.dump(bytes_to_send, file)
    else:
        with open(file_path, "rb") as file:
            bytes_to_send = pickle.load(file)
        
    totaltimestart = time.time()
    rate = 1/int(sys.argv[1])
    seconds = 100

    #est_queries = int(seconds/rate * 1.2)
    est_queries = 6000
    bytes_to_send_extended = []
    for i in range(est_queries):
        requestid = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
        bytes = bytes_to_send[i%len(bytes_to_send)][0]
        bytes_to_send_extended.append((bytes, requestid))

    bytes_to_send = bytes_to_send_extended
    test = "FF"
    #test = "SYNC"
    #for i in range(int(seconds/rate)):
    for i in range(est_queries):
        req = bytes_to_send[i]
        #print(getsize(req))
        #print(batch['text_sequence'], batch['question'])
        #data = convert_to_numpy(batch)
        # data = batch
        # data['pixel_values'] = data['pixel_values'][0]
        # data['attention_mask'] = data['attention_mask'][0]
        # data['input_ids'] = data['input_ids'][0]
        # data['text_sequence'] = data['text_sequence'][0]
        # data['question'] = data['question'][0]
        # data['question_id'] = data['question_id'][0]

        # requestid = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
        # data['requestid'] = requestid
        data = req[0]
        requestid = req[1]
        logger.info(f"Client_Send {requestid}")
        for attempt in range(1, max_retries + 1):
            try:
                if test == "FF":
                    fire_and_forget(get_random_host(), data, requestid=requestid)
                if test == "SYNC":
                    response = request_task_sync("http://127.0.0.1:8000/", data, requestid)
                    response.raise_for_status()  # Raise an exception for HTTP error codes
                    output = response.json()
                    if(output[0] == "error"):
                        print("Error in query", requestid)
                        raise Exception("Error in query")
                    logger.info(f"Client_Rec {requestid}")
                    print(output)
                    print("Request",requestid,"took", response.elapsed.total_seconds(), "seconds")
                    answers.append(output)
                break  # Exit the retry loop if the request was successful
            except (requests.RequestException, json.JSONDecodeError, Exception) as e:
                print(f"Attempt {attempt} failed for query {requestid}: {e}")
                if attempt == max_retries:
                    print("Max retries reached for query", requestid)
                    exit(1)
                else:
                    # Optionally add a delay before retrying
                    time.sleep(1)
        time.sleep(rate)


    print("TOTAL TIME:", time.time()-totaltimestart)
    if test == "FF":
        exit(0)
    passages_ds = load_dataset('parquet', data_files ={  
                                                'train' : p_ds_dir + '/train_passages-00000-of-00001.parquet',
                                                'test'  : p_ds_dir + '/test_passages-00000-of-00001.parquet',
                                                })[use_split]

    correct = 0
    for i in range(int(seconds/rate)):
        data = ds[i]
        correct_passage = data['pos_item_ids']
        if(len(correct_passage)!= 1):
            print("ERROR: More than one correct passage", i)
        
        correct_passage = correct_passage[0]
        for answer in answers:
            passage_num = answer[0]
            passage = passages_ds[passage_num]
            #print(correct_passage, passage['passage_id'])
            if correct_passage in passage['passage_id']:
                correct += 1
                break
    print(correct, "out of", int(seconds/rate), "correct")