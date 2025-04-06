import json
import string
import requests
import pickle
import os
import random

hosts = [
        "http://10.10.1.11:8000",
    ]
data_dir = "/mydata/pipeline2"

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

# Load the queries from the pickle file 
# using the correct method
queries = []
with open(os.path.join(data_dir, "queries_audio1.pkl"), "rb") as f:
    queries = pickle.load(f)

print(len(queries))
print(len(queries[0][1]))


# Send the first query to the server
query = queries[4]
url = get_random_host()
requestid = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(8))
response = request_task_sync(url, query[1].tolist(), requestid=requestid)
print(query[0], "|",response.json())
#print(json.dumps(queries[0][1].tolist(), indent=2))