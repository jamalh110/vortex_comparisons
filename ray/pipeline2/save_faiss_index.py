import faiss
import pickle
import numpy as np
import os


#test if gpu is recognized
print(faiss.get_num_gpus())

def load_cluster_embeddings2(cluster_dir):
    cluster_embeddings = None
    with open(cluster_dir+"/embeddings_list.pkl", "rb") as f:
        docs = pickle.load(f)
        cluster_embeddings = np.array(docs).astype(np.float32)
    return cluster_embeddings

def build_ivf_index(cluster_embeddings, nlist=10):
    dim = cluster_embeddings.shape[1]  
    quantizer = faiss.IndexFlatL2(dim) 
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    index.train(cluster_embeddings)  
    index.add(cluster_embeddings)      

    res = faiss.StandardGpuResources()  
    quantizer = faiss.IndexFlatL2(dim)  
    #quantizer = faiss.index_cpu_to_gpu(res, 0, quantizer)  

    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)
    index = faiss.index_cpu_to_gpu(res, 0, index)  # Move full index to GPU
    # Train and add embeddings
    index.train(cluster_embeddings)
    index.add(cluster_embeddings) 

    index = faiss.index_gpu_to_cpu(index)

    return index

def load_index(index_file):
    index = faiss.read_index(index_file)
    gpu_res = faiss.StandardGpuResources()
    index = faiss.index_cpu_to_gpu(gpu_res, 0, index)
    
embs = load_cluster_embeddings2("/mydata/msmarco_3_clusters")
index = build_ivf_index(embs)

faiss.write_index(index, "/mydata/msmarco.index")
