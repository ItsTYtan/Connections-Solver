#!/usr/bin/env python3
"""
NYT Connections ILP Solver (OR-Tools only)

- Assigns 16 words into 4 groups of 4 to maximize intra-group similarity.
- Uses OR-Tools CP-SAT exclusively (no PuLP fallback).
- Includes a small heuristic similarity builder; you can replace it with your own.

Install:
    pip install ortools

Run (demo):
    python nyt_connections_ortools.py

Use with your own words:
    - Edit the `WORDS` list in `main()` (must be 16 items).
    - Optionally, replace `build_similarity_matrix` with your own similarity.
"""

import json
from typing import List, Dict, Tuple, Optional
import numpy as np
from ortools.sat.python import cp_model
import itertools

from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from tqdm import tqdm

# ------------------------------
# Lightweight similarity builder
# ------------------------------

def char_ngrams(s: str, n: int) -> set:
    s = s.lower()
    if len(s) < n:
        return {s}
    return {s[i:i+n] for i in range(len(s)-n+1)}

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

def affix_signal(a: str, b: str) -> float:
    """Shared prefix/suffix bonus up to 0.4 total (heuristic)."""
    a_l, b_l = a.lower(), b.lower()
    score = 0.0
    for k in (4, 3, 2):
        if k <= len(a_l) and k <= len(b_l):
            if a_l[:k] == b_l[:k]:
                score = max(score, 0.10 * k)  # up to 0.40
            if a_l[-k:] == b_l[-k:]:
                score = max(score, 0.10 * k)
    return min(score, 0.4)

def build_similarity_matrix(words: List[str]) -> List[List[float]]:
    """
    Very small heuristic: Jaccard over char bigrams+trigrams + affix bonus.
    Produces values in [0, 1]; diagonal set to 0.
    For best performance on real boards, replace with a stronger semantic model.
    """
    n = len(words)
    grams = [char_ngrams(w, 2) | char_ngrams(w, 3) for w in words]
    S = [[0.0]*n for _ in range(n)]
    for i, j in itertools.combinations(range(n), 2):
        base = 0.7 * jaccard(grams[i], grams[j])
        base += 0.3 * affix_signal(words[i], words[j])
        if base < 0.0:
            base = 0.0
        if base > 1.0:
            base = 1.0
        S[i][j] = S[j][i] = base
    return S

def cosine_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    E = np.asarray(embeddings, dtype=float)
    # L2-normalize rows (avoid divide-by-zero)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    E_norm = E / norms
    # Cosine similarity = dot of normalized rows
    S = E_norm @ E_norm.T
    # Clip tiny numerical drift
    np.clip(S, -1.0, 1.0, out=S)
    return S

# ------------------------------
# OR-Tools CP-SAT model
# ------------------------------

def solve_connections_ortools(
    words: List[str],
    S: Optional[List[List[float]]] = None,
    groups: int = 4,
    group_size: int = 4,
    time_limit_s: float = 10.0,
    num_workers: int = 8,
    must_together: Optional[List[Tuple[int, int]]] = None,
    cannot_together: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, List[str]]:
    """
    Solve the 4x4 partition maximizing sum of pairwise similarities.

    Args:
        words: list of strings; length must be groups*group_size.
        S: similarity matrix in [0,1]; if None, a heuristic one is built.
        groups: number of groups (default 4).
        group_size: required size of each group (default 4).
        time_limit_s: max solver time in seconds (default 10).
        num_workers: parallel workers for CP-SAT.
        must_together: optional list of (i, j) indices forced into the same group.
        cannot_together: optional list of (i, j) indices forced into different groups.

    Returns:
        dict mapping "Group k" -> list of words assigned.
    """
    assert len(words) == groups * group_size, "words must be exactly groups * group_size"
    n = len(words)
    if S is None:
        S = build_similarity_matrix(words)

    # Pair index list for convenience
    pairs = [(i, j) for i in range(n) for j in range(i+1, n)]

    model = cp_model.CpModel()

    # x[i,g] = 1 if word i is in group g
    x = {(i, g): model.NewBoolVar(f"x[{i},{g}]") for i in range(n) for g in range(groups)}
    # y[i,j,g] = 1 if words i and j are together in group g
    y = {(i, j, g): model.NewBoolVar(f"y[{i},{j},{g}]") for (i, j) in pairs for g in range(groups)}

    # Each word in exactly one group
    for i in range(n):
        model.Add(sum(x[(i, g)] for g in range(groups)) == 1)

    # Each group has exactly group_size words
    for g in range(groups):
        model.Add(sum(x[(i, g)] for i in range(n)) == group_size)

    # Linking: y <= x and y >= x_i + x_j - 1
    for (i, j) in pairs:
        for g in range(groups):
            model.Add(y[(i, j, g)] <= x[(i, g)])
            model.Add(y[(i, j, g)] <= x[(j, g)])
            model.Add(y[(i, j, g)] >= x[(i, g)] + x[(j, g)] - 1)

    # Optional hard constraints
    if must_together:
        for (i, j) in must_together:
            # Force them to share a group: sum_g y[i,j,g] == 1
            model.Add(sum(y[(i, j, g)] for g in range(groups)) == 1)
    if cannot_together:
        for (i, j) in cannot_together:
            # Forbid being in the same group: sum_g y[i,j,g] == 0
            model.Add(sum(y[(i, j, g)] for g in range(groups)) == 0)

    # Objective: maximize sum S_ij * y[i,j,g]
    scale = 1000  # keep 3 decimals while using integer coefficients
    model.Maximize(
        sum(int(round(S[i][j] * scale)) * y[(i, j, g)] for (i, j) in pairs for g in range(groups))
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = num_workers

    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No solution found by OR-Tools.")

    assignment = {g: [] for g in range(groups)}
    for i in range(n):
        for g in range(groups):
            if solver.Value(x[(i, g)]) == 1:
                assignment[g].append(words[i])

    # Make result stable and readable
    result: List[List[str]] = []
    for g in range(groups):
        result.append(sorted(assignment[g]))
    return result


def main():
    dataset = load_dataset("ItsTYtan/nytimes-connections", split="test")

    puzzles = [entry["puzzle"] for entry in dataset]
    solutions = [entry["solution"] for entry in dataset]

    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    results = []
    for puzzle, solution in tqdm(list(zip(puzzles, solutions)), desc="evaluating"):
        embeddings = embedding_model.encode(puzzle, show_progress_bar=False)
        S = cosine_similarity_matrix(embeddings=embeddings)

        guesses = solve_connections_ortools(
            words=puzzle,
            groups=4,
            S=S,
            group_size=4,
            time_limit_s=10.0,
            num_workers=8,
        )

        results.append({
            "guesses": guesses,
            "solution": solution
        })

    with open("outputs/ilp.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

