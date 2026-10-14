"""Pure helpers for the isolated per-dataset dynamic hard-batch experiment."""

import numpy as np


def pair_vectors(query_embeddings, positive_embeddings):
    """Vectors whose dot product equals the four-term pair similarity."""
    query_embeddings = np.asarray(query_embeddings, dtype=np.float32)
    positive_embeddings = np.asarray(positive_embeddings, dtype=np.float32)
    if query_embeddings.shape != positive_embeddings.shape:
        raise ValueError("query and positive caches must have identical shapes")
    return 0.5 * (query_embeddings + positive_embeddings)


def exact_farthest_seeds(pair_embeddings, count, initial_index=0):
    """Deterministic min-max/farthest-point traversal for a small seed set."""
    vectors = np.asarray(pair_embeddings, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) == 0:
        raise ValueError("pair_embeddings must be a nonempty matrix")
    count = min(int(count), len(vectors))
    if count <= 0:
        return np.empty(0, dtype=np.int64)
    current = int(initial_index) % len(vectors)
    selected = np.empty(count, dtype=np.int64)
    chosen = np.zeros(len(vectors), dtype=bool)
    max_similarity = np.full(len(vectors), -np.inf, dtype=np.float32)
    for position in range(count):
        selected[position] = current
        chosen[current] = True
        similarity = vectors @ vectors[current]
        max_similarity = np.maximum(max_similarity, similarity)
        max_similarity[chosen] = np.inf
        if position + 1 < count:
            current = int(np.argmin(max_similarity))
    return selected


def exact_hard_batch(query_embeddings, positive_embeddings, seed, batch_size, candidate_multiplier=8):
    """Exact Top-K retrieval followed by incremental max-sum greedy."""
    query = np.asarray(query_embeddings, dtype=np.float32)
    positive = np.asarray(positive_embeddings, dtype=np.float32)
    if query.shape != positive.shape or query.ndim != 2:
        raise ValueError("query and positive embeddings must be equal-sized matrices")
    size = len(query)
    batch_size = min(int(batch_size), size)
    seed = int(seed)
    if not 0 <= seed < size:
        raise IndexError(seed)
    hardness = query[seed] @ positive.T + query @ positive[seed]
    hardness[seed] = -np.inf
    candidate_count = min(max(batch_size - 1, int(candidate_multiplier) * batch_size), size - 1)
    candidate_ids = np.argpartition(hardness, -candidate_count)[-candidate_count:]
    candidate_ids = candidate_ids[np.argsort(-hardness[candidate_ids], kind="stable")]
    selected = [seed]
    available = np.ones(candidate_count, dtype=bool)
    gain = hardness[candidate_ids].copy()
    while len(selected) < batch_size:
        masked_gain = np.where(available, gain, -np.inf)
        slot = int(np.argmax(masked_gain))
        chosen = int(candidate_ids[slot])
        selected.append(chosen)
        available[slot] = False
        gain += query[chosen] @ positive[candidate_ids].T
        gain += query[candidate_ids] @ positive[chosen]
    return np.asarray(selected, dtype=np.int64), candidate_ids
