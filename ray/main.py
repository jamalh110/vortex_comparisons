import asyncio
import json
import ray
from ray import serve
import requests
from starlette.requests import Request
from ray.serve.handle import DeploymentHandle
from FlagEmbedding import BGEM3FlagModel, FlagModel
import numpy as np
from typing import Any, Dict, List
import os 
import pickle
from huggingface_hub import login
from datasets import load_dataset

import transformers
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan, BartForSequenceClassification, BartTokenizer, pipeline
import faiss
import torch
import soundfile as sf
import textwrap

_MAX_BATCH_SIZE = 32

#TODO: pass in numpy arrays in the first place so you dont have to keep converting


@serve.deployment(max_ongoing_requests=32, ray_actor_options={"num_gpus": 0.05})
class Encoder:
    def __init__(self):
        # self.encoder = BGEM3FlagModel("BAAI/bge-m3", use_fp16=False, device="cpu")
        # self.centroids_embeddings = np.array([])
        # self.emb_dim = 1024
        self.encoder = FlagModel('BAAI/bge-small-en-v1.5', devices="cuda:0")
        self.emb_dim = 384
        self.centroids_embeddings = np.array([])

    def run(self, query_list: List[str]):
        #print("\n\n\n\n", "here2", "\n\n\n\n")
        # encode_result = self.encoder.encode(
        #        query_list, return_dense=True, return_sparse=False, return_colbert_vecs=False
        #   )
        # query_embeddings = encode_result['dense_vecs']
        # query_embeddings = query_embeddings[:, :self.emb_dim]  

        query_embeddings = self.encoder.encode(query_list)
        return query_embeddings
    
    @serve.batch(max_batch_size=_MAX_BATCH_SIZE)
    async def __call__(self, input: List[str]):
        #print("\n\n\n\n", "here1", "\n\n\n\n")
        return self.run(input)

@serve.deployment(max_ongoing_requests=32, ray_actor_options={"num_gpus": 0.65})
class Search:
    def __init__(self, cluster_dir = "/mydata/msmarco_3_clusters", index_type='Flat', nprobe=1):
        self.index_type = index_type
        self.nprobe = nprobe
        self.index = None
        self.load_index("/mydata/msmarco.index")
        #self.load_cluster_embeddings2(cluster_dir)
        #print("\n\n\n\n", "embeddings shape",self.cluster_embeddings.shape, "\n\n\n\n")
        #self.build_ivf_index()


    def load_cluster_embeddings(self, cluster_dir):
        self.cluster_embeddings = []
        for file in os.listdir(cluster_dir):
            if file.startswith("cluster_") and file.endswith(".pkl"):
                file_path = os.path.join(cluster_dir, file)
                with open(file_path, "rb") as f:
                    emb = pickle.load(f)
                    self.cluster_embeddings.append(emb)
        self.cluster_embeddings = np.vstack(self.cluster_embeddings).astype(np.float32)

    def load_cluster_embeddings2(self, cluster_dir):
        self.cluster_embeddings = None
        with open(cluster_dir+"/embeddings_list.pkl", "rb") as f:
            docs = pickle.load(f)
            self.cluster_embeddings = np.array(docs).astype(np.float32)

    def build_ivf_index(self, nlist=10):
        print("\n\n\n\n", "Faiss GPUS",faiss.get_num_gpus(), "\n\n\n\n")
        dim = self.cluster_embeddings.shape[1]  
        quantizer = faiss.IndexFlatL2(dim) 
        self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        self.index.train(self.cluster_embeddings)  
        self.index.add(self.cluster_embeddings)      
        #print("\n\n\n\n", "here1", "\n\n\n\n")

        res = faiss.StandardGpuResources()  
        #print(res)
        quantizer = faiss.IndexFlatL2(dim)  
        #quantizer = faiss.index_cpu_to_gpu(res, 0, quantizer)  

        #print("\n\n\n\n", "here2", "\n\n\n\n")
        self.index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)  # Move full index to GPU
        #print("\n\n\n\n", "here3", "\n\n\n\n")
          # Train and add embeddings
        self.index.train(self.cluster_embeddings)
        self.index.add(self.cluster_embeddings) 

    def load_index(self, index_file):
        self.index = faiss.read_index(index_file)
        gpu_res = faiss.StandardGpuResources()
        self.index = faiss.index_cpu_to_gpu(gpu_res, 0, self.index)

    #TODO: find a faster way to do the return here
    def run(self, query_list: Any, top_k=5):
        #print("\n\n\n\n", "here2", "\n\n\n\n")
        #while True:
        distances, indices = self.index.search(query_list, top_k)
        combined = [[d, i] for d, i in zip(distances, indices)]
        return combined
    
    @serve.batch(max_batch_size=_MAX_BATCH_SIZE)
    async def __call__(self, input: List[Any]):
        formatted = np.array(input)
        #print("\n\n\n\n", "input shape", formatted.shape, "\n\n\n\n")
        ret = self.run(formatted)
        #print(ret)
        return ret
    
@serve.deployment(max_ongoing_requests=1, ray_actor_options={"num_gpus": 0.2})
class DocGen:
    def __init__(self):
        self.doc_file_name = '/mydata/msmarco_3_clusters/doc_list.pkl'
        self.doc_list = None
        self.pipeline = None
        self.terminators = None
        self.doc_data = None

        login(token="<access_token>")
        self.load_llm()

    def load_llm(self,):
        #model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.float16},
            device_map="auto",   
        )
        self.terminators = [
            self.pipeline.tokenizer.eos_token_id,
            self.pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    def llm_generate(self, query_text, doc_list):
        messages = [
            {"role": "system", "content": "Answer the user query in one clear, concise sentence based on the following documents:"+" ".join(doc_list)},
            {"role": "user", "content": query_text},
        ]
        
        tmp_res = self.pipeline(
            messages,
            max_new_tokens=256,
            eos_token_id=self.terminators[0],
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        raw_text = tmp_res[0]["generated_text"][-1]['content']

        #print(f"for query:{query_text}")
        #print(f"the llm generated response: {raw_text}")
        return raw_text

     # Retrieve Documents Based on FAISS Indices
    def get_documents(self,top_k_idxs):          
        doc_list = []
        if self.doc_data is None:
            with open(self.doc_file_name, "rb") as f:
                self.doc_data = pickle.load(f)  
        
        for idx in top_k_idxs:
            if 0 <= idx < len(self.doc_data):
                doc_text = self.doc_data[idx]
                doc_list.append(doc_text)
            else:
                print(f"Warning: FAISS index {idx} is out of range in doc_list.pkl.")
        #print(doc_list)
        return doc_list
               
     
     
    def generate(self, query_text, doc_ids):
        doc_list = self.get_documents(doc_ids)
        
        llm_res = self.llm_generate(query_text, doc_list)

        return llm_res

    async def __call__(self, query_text: str, search_result: Any):
        doc_ids = search_result[1]
        return self.generate(query_text, doc_ids)


# @serve.deployment(max_ongoing_requests=5, ray_actor_options={"num_gpus": 0.1})
# class TTS:
#     def __init__(self):
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically select GPU if available

#         # Load models and move them to GPU if available
#         self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
#         self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
#         self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
#         self.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
#         self.speaker_embeddings = torch.tensor(self.embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(self.device)
#     def split_text(self, text, chunk_size=600):
#         return textwrap.wrap(text, width=chunk_size)  # Splits into smaller chunks


#     def text2soundfunc(self, text):        
        
#         # Split text into smaller segments
#         text_chunks = self.split_text(text)
#         speech_outputs = []

#         for idx, chunk in enumerate(text_chunks):
#             inputs = self.processor(text=chunk, return_tensors="pt").to(self.device)  # Move inputs to GPU
#             speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
#             speech_outputs.append(speech.cpu().numpy())  # Move back to CPU before converting to NumPy
#             print(f"Chunk {idx+1}/{len(text_chunks)} processed.")
#             del inputs
#             torch.cuda.empty_cache()
        
#         # Concatenate all speech chunks
#         full_speech = np.concatenate(speech_outputs, axis=0)
#         print(f"Size of speech_outputs: {full_speech.shape}")

#         # Save as a WAV file
#         #sf.write("speech.wav", full_speech, samplerate=16000)

#         return full_speech
    
#     async def __call__(self, text: str):
#         return self.text2soundfunc(text)
    
@serve.deployment(max_ongoing_requests=1, ray_actor_options={"num_gpus": 0.01}, num_replicas=2)
class TTS:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # Automatically select GPU if available

        # Load models and move them to GPU if available
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
        self.model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(self.device)
        self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(self.device)
        self.embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
        self.speaker_embeddings = torch.tensor(self.embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(self.device)

    def _process_chunk(self, chunk):
        # Process one chunk synchronously.
        inputs = self.processor(text=chunk, return_tensors="pt").to(self.device)  # Move inputs to GPU
        speech = self.model.generate_speech(inputs["input_ids"], self.speaker_embeddings, vocoder=self.vocoder)
        speech_np = speech.cpu().numpy()  # Move back to CPU before converting to NumPy
        # Free GPU memory for the current chunk.
        del inputs
        torch.cuda.empty_cache()
        return speech_np

    async def __call__(self, text: str):
        return self._process_chunk(text)

@serve.deployment(max_ongoing_requests=32, ray_actor_options={"num_gpus": 0.01})
class TextCheck:
    def __init__(self):
        self.device = torch.device('cuda')
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-large-mnli', device_map = self.device)
        self.model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli').to(self.device)

    def textcheck(self, premises):
        # pose sequence as a NLI premise and label (politics) as a hypothesis
        # premise = 'I hate Asians!'
        # premise = 'A new model offers an explanation for how the Galilean satellites formed around the solar system’s largest world.'
        hypothesis = 'harmful.'

        # Tokenize all premises as a batch
        inputs = self.tokenizer(premises, [hypothesis] * len(premises), 
                        return_tensors='pt', padding=True, truncation=True).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            result = self.model(**inputs)
        
        logits = result.logits  # result[0] is now deprecated, use result.logits instead

        # Take only entailment and contradiction logits
        entail_contradiction_logits = logits[:, [0, 2]]
        probs = entail_contradiction_logits.softmax(dim=1)
        true_probs = probs[:, 1] * 100  # Probability of entailment

        for i, prob in enumerate(true_probs.tolist()):
            print(f'Premise {i+1}: Probability that the label is true: {prob:.2f}%')

        return true_probs.tolist()
    
    @serve.batch(max_batch_size=_MAX_BATCH_SIZE)
    async def __call__(self, input: List[str]):
        #print("\n\n\n\n", "here1", "\n\n\n\n")
        return self.textcheck(input)

@serve.deployment
class Ingress:
    def __init__(self, encoder: DeploymentHandle, search: DeploymentHandle, docgen: DeploymentHandle, tts: DeploymentHandle, textcheck: DeploymentHandle):
        self.search = search
        self.encoder = encoder
        self.docgen = docgen
        self.tts = tts
        self.textcheck = textcheck

    def split_text(self, text, chunk_size=550):
        return textwrap.wrap(text, width=chunk_size)
    
    async def __call__(self, http_request: Request):
        input = await http_request.json()
        #print("\n\n\n\n", input, "\n\n\n\n")
        encode_result = self.encoder.remote(input)
        search_result = self.search.remote(encode_result)
        docgen_result = await self.docgen.remote(input, search_result)
        chunks = self.split_text(docgen_result)

        tasks = [self.tts.remote(chunk) for chunk in chunks]
        textcheck_result = self.textcheck.remote(docgen_result)

        tts_result_raw = await asyncio.gather(*tasks)

        tts_result = np.concatenate(tts_result_raw, axis=0)
        print(f"Size of speech_outputs: {tts_result.shape}")

        return {"tts": tts_result, "textcheck": await textcheck_result}

encoder = Encoder.bind()
search = Search.bind()
docgen = DocGen.bind()
tts = TTS.bind()
textcheck = TextCheck.bind()
app = Ingress.bind(encoder=encoder, search=search, docgen=docgen, tts=tts, textcheck=textcheck)

#handle: DeploymentHandle = serve.run(app)

# input = "\"what is prescribed to treat thyroid storm\""
# data = json.loads(input)
# response = requests.post("http://127.0.0.1:8000/", json=data)
# output = response.text
# print("\n\n\n", "output", len(output), output, "\n\n\n")
