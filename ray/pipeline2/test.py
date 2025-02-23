import json
import requests

input = "\"what is prescribed to treat thyroid storm\""
data = json.loads(input)

response = requests.post("http://127.0.0.1:8000/", json=data)

output = response.json()
print(output['textcheck'])
print("Request took", response.elapsed.total_seconds(), "seconds")
