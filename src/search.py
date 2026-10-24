import numpy as np


def search_product(index, df, query_vector, top_k=10):
    query_vector = np.asarray(query_vector, dtype=np.float32).reshape(1, -1)

    if query_vector.shape[1] != index.d:
        raise ValueError("Query dimension does not match the index")

    if not np.isfinite(query_vector).all():
        raise ValueError("Query contains invalid values.")

    norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
    if norm[0, 0] == 0:
        raise ValueError("Query embedding has zero norm")

    query_vector /= norm

    k = min(max(1, top_k), index.ntotal)
    scores, indices = index.search(query_vector, k)

    results = df.iloc[indices[0]].copy()
    results["similarity"] = scores[0]
    return results.reset_index(drop=True)