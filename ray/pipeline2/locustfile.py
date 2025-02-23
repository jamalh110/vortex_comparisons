from locust import HttpUser, task, constant
import json

class MyUser(HttpUser):
   # host = "http://10.10.1.10:8000"
    wait_time = constant(0)

    @task
    def send_request(self):
        input_str = "\"what is prescribed to treat thyroid storm\""
        data = json.loads(input_str)
        with self.client.post("/", json=data, catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Unexpected status code: {response.status_code}")
            else:
                try:
                    result = response.json()
                    if "textcheck" in result:
                        print(result["textcheck"])
                    else:
                        print("No 'textcheck' field found in response.")
                    print("Request took", response.elapsed.total_seconds(), "seconds")
                except Exception as e:
                    response.failure(f"Failed to parse JSON response: {e}")
