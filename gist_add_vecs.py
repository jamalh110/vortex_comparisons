import numpy as np
import faiss
import pickle
import os 
from collections import defaultdict
import json 

EMBEDDINGS_LOC = './gist'
#NAME='gist_random'
#NAME='gist_repeated'
#NAME='gist_base_random'
NAME='gist_shifted'

np.random.seed(42)
def export_array_to_jsonl(array, file_path, include_metadata=False):
    """
    Export a NumPy array to a .jsonl file in chunks of 10,000 lines.

    Parameters:
        array (numpy.ndarray): The NumPy array to export.
        file_path (str): The file path for the .jsonl file.
        include_metadata (bool): Whether to include metadata for each row. Default is False.
    """
    with open(file_path, 'w') as f:
        buffer = []
        for i, row in enumerate(array):
            if i % 10000 == 0 and i > 0:
                # Write the buffer to the file and clear it
                f.write('\n'.join(buffer) + '\n')
                buffer = []
                print(f"Processed {i} rows...")
            
            if include_metadata:
                # Create a dictionary with metadata
                json_object = {"id": i, "values": row.tolist()}
            else:
                # Use plain list format
                json_object = row.tolist()
                
            # Append JSON object as a string to the buffer
            buffer.append(json.dumps(json_object))
        
        # Write any remaining data in the buffer
        if buffer:
            f.write('\n'.join(buffer) + '\n')

    print(f"Array exported to {file_path}")

def fvecs_read(filename, dtype=np.float32, c_contiguous=True):
    fv = np.fromfile(filename, dtype=dtype)
    if fv.size == 0:
        return np.zeros((0, 0))
    dim = fv.view(np.int32)[0]
    assert dim > 0
    fv = fv.reshape(-1, 1 + dim)
    if not all(fv.view(np.int32)[:, 0] == dim):
        raise IOError("Non-uniform vector sizes in " + filename)
    fv = fv[:, 1:]
    if c_contiguous:
        fv = fv.copy()
    return fv

def new_queries_random():
    new_queries = np.random.rand(19000, 960)
    return new_queries

def new_queries_repeated(queries):
    new_queries = np.tile(queries, (19, 1))
    return new_queries

def new_queries_base_random(base):
    new_queries = base[np.random.choice(base.shape[0], size=19000, replace=False)]
    return new_queries

def modify_vectors(arr, num_changes=5):
    modified_arr = arr.copy()  # Create a copy to modify
    for vector in modified_arr:
        # Randomly select indices
        indices = np.random.choice(vector.shape[0], size=num_changes, replace=False)
        # Modify the selected elements by a random percentage between 0% and 1%
        vector[indices] += vector[indices] * np.random.uniform(-0.01, 0.01, size=num_changes)
    return modified_arr

def new_queries_shifted(queries):
    result = []
    for _ in range(19):
        modified_vectors = modify_vectors(queries)  # Modify the original array
        result.append(modified_vectors)         # Append the modified vectors

    # Combine all iterations into a single array
    final_result = np.vstack(result)
    return final_result

base = fvecs_read('./gist/gist_base.fvecs')
print("Base shape", base.shape)
#print(base[0])


groundtruth = fvecs_read('./gist/gist_groundtruth.ivecs', np.int32)
print("groundtruth shape",groundtruth.shape)
#print(groundtruth[0])

query = fvecs_read('./gist/gist_query.fvecs')
print("query shape", query.shape)


dimension = base.shape[1]  # Assumes emb_list is a 2D array (num_embeddings, embedding_dim)
print("dimension", dimension)
# Create a FAISS index, here we're using an IndexFlatL2 which is a basic index with L2 distance

#new_queries = new_queries_random()
#new_queries = new_queries_repeated(query)
#new_queries = new_queries_base_random(base)
new_queries = new_queries_shifted(query)
print("new_queries shape", new_queries.shape)
#print("new_queries", new_queries[0])

#combine query and new_queries
query = np.concatenate((query, new_queries), axis=0)
#query = new_queries
print("new query shape", query.shape)

index = faiss.IndexFlatL2(dimension)
index.add(base)

k=100

distances, indices = index.search(query[0].reshape(1, -1), k)
# Print the results
print("Indices of nearest neighbors:", indices)
print("Distances to nearest neighbors:", distances)
print("groundtruth", groundtruth[0])

#get distances and indices for all new queries in batches of 100
batch_size = 100
k=100
num_batches = int(query.shape[0]/batch_size)
print("num_batches", num_batches)
distances = np.empty((0,k))
indices = np.empty((0,k), dtype=int)
for i in range(num_batches):
    print("batch number", i)
    start = i*batch_size
    end = (i+1)*batch_size
    print("start", start)
    print("end", end)
    distances_batch, indices_batch = index.search(query[start:end], k)
    distances = np.concatenate((distances, distances_batch), axis=0)
    indices = np.concatenate((indices, indices_batch), axis=0)
    print("distances_batch", distances_batch.shape)
    print("indices_batch", indices_batch.shape)

print("distances", distances.shape)
print("indices", indices.shape)

#compare indices with groundtruth

for i in range(1000):
    count = 0
    for j in range(len(indices[i])):
        if indices[i][j] in groundtruth[i]:
            count += 1
    print("count", count)
    if(count <100):
        print("i", i)
        #print("indices", indices[i])
        #print("groundtruth", groundtruth[i])


export_array_to_jsonl(indices, f'/mydata/{NAME}/neighbours.jsonl', include_metadata=False)
#export_array_to_jsonl(distances, 'distances.jsonl', include_metadata=False)
export_array_to_jsonl(query, f'/mydata/{NAME}/queries.jsonl', include_metadata=False)
export_array_to_jsonl(base, f'/mydata/{NAME}/vectors.jsonl', include_metadata=False)
        
