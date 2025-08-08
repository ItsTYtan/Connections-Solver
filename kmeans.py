import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
from bertopic import BERTopic
from datasets import load_dataset
from sklearn.cluster import AgglomerativeClustering, KMeans
from tqdm import tqdm
from transformers.pipelines import pipeline

dataset = load_dataset("ItsTYtan/nytimes-connections", split="test")

puzzles = [entry["puzzle"] for entry in dataset]
solutions = [entry["solution"] for entry in dataset]

# results = []
# for puzzle, solution in tqdm(list(zip(puzzles, solutions)), desc="evalutating"):
#     cluster_model = KMeans(n_clusters=4)
#     model = BERTopic(hdbscan_model=cluster_model)
#     topics, _ = model.fit_transform(puzzle) 
    
#     guesses = [[],[],[],[]]
#     for topic, doc in list(zip(topics, puzzle)):
#         guesses[topic].append(doc)

#     results.append({
#         "guesses": guesses,
#         "solution": solution
#     })

# with open("results/kmeans.json", "w") as f:
#     json.dump(results, f, indent=2)


results = []
embedding_model = pipeline("feature-extraction", model="Qwen/Qwen3-Embedding-8B")
cluster_model = KMeans(n_clusters=4)
for puzzle, solution in tqdm(list(zip(puzzles, solutions)), desc="evalutating"):
    model = BERTopic(
        embedding_model=embedding_model,
        hdbscan_model=cluster_model
    )
    topics, _ = model.fit_transform(puzzle) 
    
    guesses = [[],[],[],[]]
    for topic, doc in list(zip(topics, puzzle)):
        guesses[topic].append(doc)

    results.append({
        "guesses": guesses,
        "solution": solution
    })

with open("results/kmeansQwen3_8b_embedding.json", "w") as f:
    json.dump(results, f, indent=2)