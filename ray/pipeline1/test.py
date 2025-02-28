import json
import random
import string
import time
import requests
from datasets import load_dataset
import os 
from PIL import Image
from utils import make_logger

LOGGING_DIR = "/users/jamalh11/raylogs"
logger = make_logger(os.getpid(), "Client", LOGGING_DIR)

DATA_DIR = "/mydata"
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
    

image_root_dir = f"{DATA_DIR}/EVQA"
use_split = "train"
ds_dir = f"{DATA_DIR}/EVQA/EVQA_data/EVQA_data"
p_ds_dir = f"{DATA_DIR}/EVQA/EVQA_passages/EVQA_passages"

ds = load_dataset('parquet', data_files ={  
                                            'train' : ds_dir + '/train-00000-of-00001.parquet',
                                            'test'  : ds_dir + '/test-00000-of-00001-2.parquet',
                                            })[use_split]


ds = ds.map(add_path_prefix_in_img_path, fn_kwargs={"prefix": image_root_dir})
print("DS LEN", len(ds))
#ds = ds.map(process_image)

print(ds[0])

def onecall():
    data = ds[0]
    response = requests.post("http://127.0.0.1:8000/", json=data)
    output = response.json()
    print(output)
# onecall()
#exit(0)
#print(ds['question_id'])
#exit(0)
nqueries = 500
max_retries = 3
answers = []
totaltimestart = time.time()
for i in range(nqueries):
    data = ds[i]
    requestid = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    data['requestid'] = requestid
    logger.info(f"Client_Send {requestid}")
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post("http://127.0.0.1:8000/", json=data)
            response.raise_for_status()  # Raise an exception for HTTP error codes
            output = response.json()
            if(output[0] == "error"):
                print("Error in query", i)
                raise Exception("Error in query")
            logger.info(f"Client_Rec {requestid}")
            #print(output)
            #print("Request",i,"took", response.elapsed.total_seconds(), "seconds")
            answers.append(output)
            break  # Exit the retry loop if the request was successful
        except (requests.RequestException, json.JSONDecodeError, Exception) as e:
            print(f"Attempt {attempt} failed for query {i}: {e}")
            if attempt == max_retries:
                print("Max retries reached for query", i)
                exit(1)
            else:
                # Optionally add a delay before retrying
                time.sleep(1)
    #time.sleep(10)  
print("TOTAL TIME:", time.time()-totaltimestart)
passages_ds = load_dataset('parquet', data_files ={  
                                            'train' : p_ds_dir + '/train_passages-00000-of-00001.parquet',
                                            'test'  : p_ds_dir + '/test_passages-00000-of-00001.parquet',
                                            })[use_split]

correct = 0
for i in range(nqueries):
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
print(correct, "out of", nqueries, "correct")