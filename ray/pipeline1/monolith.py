from starlette.requests import Request
from ray.serve.handle import DeploymentHandle
from ray.serve import Application
from ray import serve
import ray

_MAX_BATCH_SIZE = 32

@serve.deployment
class Monolith:
    