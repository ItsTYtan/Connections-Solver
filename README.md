# Connections-Solver
An attempt to use ml techniques to solve connections with performance equal to SOTA LLM models.

(10/08/25)
Kmeans clustering performs poorly due to several reasons:
- Semantic relations are only one type of connections
- Clusters formed are not neccessarily of size 4

Tried ILP to contrain clusters to same size
ILP performs better than kmeans, but nowhere near good results

kmeans.json avg correct guesses: 0.11392405063291139
kmeans.json score: 0.028481012658227847
ilp.json avg correct guesses: 0.34177215189873417
ilp.json score: 0.08544303797468354

Embedding model: all-MiniLM-L6-v2
Similarity heuristic: cosine similarity