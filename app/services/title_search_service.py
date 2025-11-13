# app/services/title_search_service.py
from typing import List, Dict, Any

import app.dependencies as deps 

def search_anime_titles(query: str, k: int = 5, cutoff: float = 0.55) -> List[Dict[str, Any]]:
    """
    단일 질의로 애니 제목 검색.
    결과: [{anime_id, score, normalized_title}, ...]
    """
    if deps.title_searcher is None:
        raise RuntimeError("title_searcher is not initialized. Call init_models() on startup.")

    hits = deps.title_searcher.search(query, k=k, cutoff=cutoff)
    return [
        {
            "anime_id": anime_id,
            "score": score,
            "normalized_title": norm_title,
        }
        for (anime_id, score, norm_title) in hits
    ]


def batch_search_anime_titles(
    queries: List[str],
    k: int = 5,
    cutoff: float = 0.55
) -> List[Dict[str, Any]]:
    """
    여러 질의를 한 번에 검색 (각 질의마다 상위 1개만 반환).
    """
    if deps.title_searcher is None:
        raise RuntimeError(
            "title_searcher is not initialized. Call init_models() on startup."
        )

    batch_hits = deps.title_searcher.batch_search(queries, k=k, cutoff=cutoff)
    results: List[Dict[str, Any]] = []

    for q, hits in zip(queries, batch_hits):
        if not hits:  # empty (cutoff 때문에 0개일 수 있음)
            results.append({
                "query": q,
                "result": None
            })
            continue

        # 상위 1개만
        anime_id, score, norm_title = hits[0]
        results.append({
            "query": q,
            "result": {
                "anime_id": anime_id,
                "score": score,
                "normalized_title": norm_title
            }
        })

    return results