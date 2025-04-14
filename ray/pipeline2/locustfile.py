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
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor
import pickle
import string
from tqdm import tqdm
import numpy as np
# Global configuration and helper functions

SELECT_DS = range(164000, 167000)  
data_dir = "/mydata/msmarco"
query_file = "queries_audio5000.pkl"
hosts = [
        "http://10.10.1.9:8000",
        "http://10.10.1.8:8000",
        #"http://10.10.1.7:8000",
        "http://10.10.1.6:8000",
        "http://10.10.1.5:8000",
        #"http://10.10.1.4:8000",
        #"http://10.10.1.3:8000",
    ]

def get_random_host():
        """Return a randomly selected host from the list."""
        return random.choice(hosts)

queries = []
with open(os.path.join(data_dir, "queries_audio5000.pkl"), "rb") as f:
    queries = pickle.load(f)
queries_filtered = []
for i in range(len(queries)):
    if len(queries[i][1]) <= 200000:
    #if len(queries[i][1]) <= 200000000:
        queries_filtered.append(queries[i])

queries = queries_filtered

class Pipeline2(HttpUser):
    #wait_time = constant(0)  # Adjust wait time as needed
    #wait_time = between(1, 4)  # Random wait time between requests
    @task
    def single_query(self):
        global total_queries, correct_queries

        # Choose a random example from the dataset
        idx = random.randint(0, len(queries) - 1)
        data = queries[idx][1].tolist()
        #print(data['pixel_values'])
        #data = convert_to_numpy(data)
        max_retries = 3
        requestid = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
        headers = {'Content-Type': 'application/json', "x-requestid": requestid}
        for attempt in range(1, max_retries + 1):
            with self.client.post(f"{get_random_host()}/", json=data, headers=headers, catch_response=True) as response:
                try:
                    response.raise_for_status()  # Raises exception for 4xx/5xx responses
                    output = response.json()
                    if output[0] == "error":
                        response.failure("Server returned an error for this query")
                        raise Exception("Error in query")
                    response.success()
                    break  # Exit retry loop on success
                except Exception as e:
                    response.failure(f"Attempt {attempt} failed: {e}")
                    print(f"Attempt {attempt} failed for query idx {idx}: {e}")
                    time.sleep(1)  # Delay before retrying
