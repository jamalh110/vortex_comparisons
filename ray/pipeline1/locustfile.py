import os
import json
import time
import random
import threading
from PIL import Image
import requests
from datasets import load_dataset
from locust import HttpUser, task, between, events, constant
from easydict import EasyDict
import torch
from flmr import (
    FLMRConfig,
    FLMRQueryEncoderTokenizer,
)
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
# Global configuration and helper functions
DATA_DIR = "/mydata"

hosts = [
        "http://10.10.1.13:8000"
    ]
def get_random_host():
        """Return a randomly selected host from the list."""
        return random.choice(hosts)

def add_path_prefix_in_img_path(example, prefix):
    if example["img_path"] is not None:
        example["img_path"] = os.path.join(prefix, example["img_path"])
    return example

def process_image(example):
    img_path = example["img_path"]
    if img_path is None:
        image = Image.new("RGB", (336, 336), color="black")
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
    #print(example['pixel_values'])
    #example["input_ids"] = example["input_ids"].numpy().tolist()
    #example["attention_mask"] = example["attention_mask"].numpy().tolist()
    #example["pixel_values"] = example["pixel_values"].numpy().tolist()
    return example

# Dataset directories and loading
image_root_dir = f"{DATA_DIR}/EVQA"
use_split = "train"
ds_dir = f"{DATA_DIR}/EVQA/EVQA_data/EVQA_data"
p_ds_dir = f"{DATA_DIR}/EVQA/EVQA_passages/EVQA_passages"

image_processor_name = '/mydata/clip-vit-large-patch14'
checkpoint_path = '/mydata/PreFLMR_ViT-L'
flmr_config = FLMRConfig.from_pretrained(checkpoint_path)
query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(checkpoint_path,
                                                                text_config=flmr_config.text_config,
                                                                subfolder="query_tokenizer")
image_processor = AutoImageProcessor.from_pretrained(image_processor_name)
# Load the primary dataset for queries
ds = load_dataset(
    "parquet",
    data_files={
        "train": os.path.join(ds_dir, "train-00000-of-00001.parquet"),
        "test": os.path.join(ds_dir, "test-00000-of-00001-2.parquet"),
    },
)[use_split].select(range(1000))
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
print("Dataset length:", len(ds))
#print("First example:", ds[0])
# Uncomment the next line if you want to process images into bytes:
# ds = ds.map(process_image)

# Load the passages dataset used for checking correctness
passages_ds = load_dataset(
    "parquet",
    data_files={
        "train": os.path.join(p_ds_dir, "train_passages-00000-of-00001.parquet"),
        "test": os.path.join(p_ds_dir, "test_passages-00000-of-00001.parquet"),
    },
)[use_split]

# Global counters for accuracy aggregation
total_queries = 0
correct_queries = 0
counter_lock = threading.Lock()


def check_answer(data, answer):
    """
    Check if the answer is correct.
    data: the query data from the ds.
    answer: response from the server, assumed to be a list where the first element is a passage number.
    """
    correct_passage = data["pos_item_ids"]
    if not correct_passage or len(correct_passage) != 1:
        print("WARNING: Unexpected pos_item_ids format", data.get("pos_item_ids"))
        return False
    correct_passage = correct_passage[0]
    try:
        passage_num = answer[0]
        passage = passages_ds[passage_num]
        if correct_passage in passage["passage_id"]:
            return True
    except Exception as e:
        print(f"Error during answer validation: {e}")
    return False

# Event hook to print accuracy when Locust is stopped
@events.quitting.add_listener
def on_quitting(environment, **kwargs):
    with counter_lock:
        if total_queries > 0:
            accuracy = correct_queries / total_queries * 100.0
            print(f"\nFinal Accuracy: {correct_queries} / {total_queries} ({accuracy:.2f}%)")
        else:
            print("No queries were executed.")


class EVQAUser(HttpUser):
    #wait_time = constant(0)  # Adjust wait time as needed
    #wait_time = between(1, 4)  # Random wait time between requests
    @task
    def single_query(self):
        global total_queries, correct_queries

        # Choose a random example from the dataset
        idx = random.randint(0, len(ds) - 1)
        data = ds[idx]
        #print(data['pixel_values'])
        data = convert_to_numpy(data)
        max_retries = 3

        for attempt in range(1, max_retries + 1):
            with self.client.post(f"{get_random_host()}/", json=data, catch_response=True) as response:
                try:
                    response.raise_for_status()  # Raises exception for 4xx/5xx responses
                    output = response.json()
                    if output[0] == "error":
                        response.failure("Server returned an error for this query")
                        raise Exception("Error in query")
                    response.success()
                    # Check the answer and update global counters
                    if False:
                        is_correct = check_answer(data, output)
                        with counter_lock:
                            total_queries += 1
                            if is_correct:
                                correct_queries += 1
                        # Optionally print per-query result
                        print(f"Query idx {idx} processed. Correct: {is_correct}. Response time: {response.elapsed.total_seconds()} s")
                    break  # Exit retry loop on success
                except Exception as e:
                    response.failure(f"Attempt {attempt} failed: {e}")
                    print(f"Attempt {attempt} failed for query idx {idx}: {e}")
                    if attempt == max_retries:
                        # Even on failure, count the query as processed
                        with counter_lock:
                            total_queries += 1
                    else:
                        time.sleep(1)  # Delay before retrying
