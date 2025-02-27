from transformers.models.bert.modeling_bert import BertEncoder
from transformers import BertConfig
import ray
from ray import cloudpickle

import pickle

ray.init(address="auto")

with open("/home/jah649/mydata/oref_a.pkl", "rb") as f:
    my_object = cloudpickle.load(f)

print(my_object)
got = ray.get(my_object)
print(got)

# transformer_mapping_config_base = 'bert-base-uncased'

# transformer_mapping_config = BertConfig.from_pretrained(transformer_mapping_config_base)


# dummy_dict = {
#         'question_id': [0],
#         'question': ["test sentence test sentece, this this, 100"],
#     }
# custom_quries = {
#             question_id: question for question_id, question in zip(dummy_dict["question_id"], dummy_dict["question"])
#         }
# print(custom_quries)