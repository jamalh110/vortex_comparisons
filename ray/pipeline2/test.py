import json
import string
import sys
import threading
import requests
import pickle
import os
import random
from myutils import make_logger
import time

hosts = [
        "http://10.10.1.11:8000",
        "http://10.10.1.10:8000",
        "http://10.10.1.9:8000",
        "http://10.10.1.8:8000",
    ]
#data_dir = "/mydata/pipeline2"
query_file = "queries_audio1.pkl"
data_dir = "/mydata/msmarco"
query_file = "queries_audio5000.pkl"
LOGGING_DIR = "/users/jamalh11/raylogs"

def get_random_host():
        """Return a randomly selected host from the list."""
        return random.choice(hosts)


def request_task(url, data_json, requestid):
    headers = {'Content-Type': 'application/json', "x-requestid": requestid}
    response = requests.post(url, json=data_json, headers=headers)

def request_task_sync(url, data_json, requestid):
    headers = {'Content-Type': 'application/json', "x-requestid": requestid}
    response = requests.post(url, json=data_json, headers=headers)
    return response

def fire_and_forget(url, data_json, requestid):
    threading.Thread(target=request_task, args=(url, data_json, requestid)).start()
    
# Load the queries from the pickle file 
# using the correct method
queries = []
with open(os.path.join(data_dir, "queries_audio5000.pkl"), "rb") as f:
    queries = pickle.load(f)
queries_filtered = []
for i in range(len(queries)):
    #if len(queries[i][1]) > 200000:
    if len(queries[i][1]) <= 200000:
        queries_filtered.append(queries[i])
    else:
        print("Query too long:", i, len(queries[i][1]))

queries = queries_filtered

queries_tolist = []
for i in range(len(queries)):
    queries_tolist.append((queries[i][0], queries[i][1].tolist()))

queries = queries_tolist
print(len(queries))
print(len(queries[0][1]))
# Send the first query to the server
logger = make_logger(os.getpid(), "Client", LOGGING_DIR)
nqueries = int(sys.argv[2])
rate = 1/int(sys.argv[1])

startime = time.time()
for i in range(nqueries):
    query = queries[i]
    url = get_random_host()
    requestid = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
    #response = request_task_sync(url, query[1], requestid=requestid)
    logger.info(f"Client_Send {requestid}")
    fire_and_forget(url, query[1], requestid=requestid)
    #print(query[0], "|",pickle.loads(response.content))
    #print(query[0], "|",response.json(), "\n\n")
    #print(json.dumps(queries[0][1].tolist(), indent=2))
    time.sleep(rate)
endtime = time.time()
print("Total time taken:", endtime - startime)