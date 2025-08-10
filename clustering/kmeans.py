import os

import hdbscan

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
from bertopic import BERTopic
from datasets import load_dataset
from sklearn.cluster import KMeans
from tqdm import tqdm

dataset = load_dataset("ItsTYtan/nytimes-connections", split="test")

puzzles = [entry["puzzle"] for entry in dataset]
solutions = [entry["solution"] for entry in dataset]

results = []
for puzzle, solution in tqdm(list(zip(puzzles, solutions)), desc="evaluating"):
    cluster_model = KMeans(n_clusters=4)
    model = BERTopic(hdbscan_model=cluster_model)
    topics, probs = model.fit_transform(puzzle) 
    
    guesses = [[],[],[],[]]
    for topic, doc in list(zip(topics, puzzle)):
        guesses[topic].append(doc)

    results.append({
        "guesses": guesses,
        "solution": solution
    })

with open("outputs/kmeans.json", "w") as f:
    json.dump(results, f, indent=2)

