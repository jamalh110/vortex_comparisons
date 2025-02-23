import json
import requests
from datasets import load_dataset
import os 
from PIL import Image


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
    

image_root_dir = "/mydata/EVQA"
use_split = "train"
ds_dir = "/mydata/EVQA/EVQA_data/EVQA_data"

ds = load_dataset('parquet', data_files ={  
                                            'train' : ds_dir + '/train-00000-of-00001.parquet',
                                            'test'  : ds_dir + '/test-00000-of-00001-2.parquet',
                                            })[use_split].select(i for i in range(999))


ds = ds.map(add_path_prefix_in_img_path, fn_kwargs={"prefix": image_root_dir})
#ds = ds.map(process_image)

print(ds[0])
#print(ds['question_id'])
#exit(0)
input = ds[0]
data = input

response = requests.post("http://127.0.0.1:8000/", json=data)

try:
    output = response.json()
except json.JSONDecodeError:
    print(response.text)
    output = {}
print(output)
print("Request took", response.elapsed.total_seconds(), "seconds")
