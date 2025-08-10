import os

from sentence_transformers import SentenceTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import json
from bertopic import BERTopic
from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from tqdm import tqdm


dataset = load_dataset("ItsTYtan/nytimes-connections", split="test")

puzzles = [entry["puzzle"] for entry in dataset]
solutions = [entry["solution"] for entry in dataset]

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
cluster_model = KMeans(n_clusters=4)

results = []
for puzzle, solution in tqdm(list(zip(puzzles, solutions)), desc="evaluating"):
    model = BERTopic(
        hdbscan_model=cluster_model
    )

    embeddings = embedding_model.encode(puzzle, show_progress_bar=False)
    embeddings_normalized = normalize(embeddings, norm='l2', axis=1)

    topics, probs = model.fit_transform(puzzle, embeddings=embeddings_normalized) 

    # X = model.umap_model.embedding_
    # dists = model.hdbscan_model.transform(X)
    # print(puzzle)
    # print(topics)
    # print(dists)

    guesses = [[],[],[],[]]
    for topic, doc in list(zip(topics, puzzle)):
        guesses[topic].append(doc)

    results.append({
        "guesses": guesses,
        "solution": solution
    })

with open("outputs/normalized_kmeans.json", "w") as f:
    json.dump(results, f, indent=2)