# Connections-Solver
An attempt to use ml techniques to solve connections with performance equal to SOTA LLM models.

(10/08/25)
Kmeans clustering performs poorly due to several reasons:
- Semantic relations are only one type of connections
- Clusters formed are not neccessarily of size 4

Tried ILP to contrain clusters to same size
ILP performs better than kmeans, but nowhere near good results

kmeans.json avg correct guesses: 0.11392405063291139
ilp.json avg correct guesses: 0.34177215189873417

Embedding model: all-MiniLM-L6-v2
Similarity heuristic: cosine similarity

(13/08/25)
With a better embedding model, ilp score is increased

all-MiniLM-L6-v2  ilp.json avg correct guesses: 0.34177215189873417
all-mpnet-base-v2 ilp.json avg correct guesses: 0.5569620253164557