'''
   lsh-embeddings: calculate LSH value for each embedding, generated from PLM.
   lshash ref: https://github.com/loretoparisi/lshash
   created on Sep 23, 2024, mht
'''
import os
import numpy as np
import argparse
import pickle
import random
from pathlib import Path

from lshashpy3.lshash import LSHash

parser = argparse.ArgumentParser(description='Process TM-Vec arguments', add_help=True)

parser.add_argument("--input-embedding",
                    type=Path,
                    required=False,
                    default="./example/result/embeddings.npy",
                    help=("Input embedding in pkl format to encode LSH.")
                    )

# load arguments
args = parser.parse_args()

embeddings = np.load(args.input_embedding)

# create 6-bit hashes for input data of 8 dimensions:
k = 8 # hash size
L = 5 # number of tables
dim = 512 #dimension of input embedding vector

# local storage for numpy uniform random planes, overwrite matrix file
lsh = LSHash(hash_size=k, input_dim = dim, num_hashtables = L,\
             storage_config={ 'dict': None },
             matrices_filename='./example/weights.npz',
             hashtable_filename='./example/hash.npz',
             overwrite=True)

# index embedding vectors
for i,embed in enumerate(embeddings):
    lsh.index(embed, extra_data='embed'+str(i))

# get the binary hash for an input embedding by iterating through all tables

embed = embeddings[0]
binary_hashes = lsh.get_hashes(embed)
print("hash representation", binary_hashes)
#print("hash representation", int(''.join(binary_hashes[0]),2))

# checking that each table stores the same input vector with different keys
#for key, table in zip(binary_hashes, lsh.hash_tables):
 #   print(key, table.get_list(key))


# query an embedding
# top_n = 4
# nn = lsh.query(embed, num_results=top_n, distance_func="cosine")

# for ((vec, extra_data), distance) in nn:
#     print("query (euclidean):", extra_data, distance)

# save hash table to disk 
lsh.save()


# execute a query loading hash table from local file system
top_n = 3
nn = lsh.query(embed, num_results=top_n, distance_func="cosine")
for ((vec, extra_data), distance) in nn:
    print("query (euclidean):", extra_data, distance)





