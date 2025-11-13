# app/services/title_search_service.py
from typing import List, Dict, Any

import app.dependencies as deps


def search_anime_titles(query: str, k: int = 5, cutoff: float = 0.55) -> List[Dict[str, Any]]:
    """
    단일 질의로 애니 제목 검색.
    결과: [{anime_id, score, normalized_title}, ...]
    """
    if deps.title_searcher is None:
        raise RuntimeError(
            "title_searcher is not initialized. Call init_models() on startup.")

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


# app/services/title_search_infer_service.py

from typing import List, Dict, Any

from app import dependencies as deps


def batch_search_anime_titles_top1(
    queries: List[str],
    k: int = 5,
    cutoff: float = 0.55,
) -> List[Dict[str, Any]]:
    """
    title_searcher.batch_search 를 직접 호출해서,
    각 질의마다 top-1 결과만 뽑아서 다음 형태로 리턴:

    [
      {
        "query": "귀멸의 칼날",
        "result": {
          "anime_id": 54944,
          "score": 0.76,
          "normalized_title": "..."
        }
      },
      ...
    ]
    """
    if deps.title_searcher is None:
        raise RuntimeError("title_searcher가 초기화되지 않았습니다. init_models()를 확인하세요.")

    # raw_hits: List[List[Tuple[anime_id, score, norm_title]]]
    raw_hits = deps.title_searcher.batch_search(queries, k=k, cutoff=cutoff)

    payload: List[Dict[str, Any]] = []

    for q, hits in zip(queries, raw_hits):
        # 디버깅용 로그
        print(f"[TITLE-SEARCH] query={q!r}, hits_len={len(hits) if hits is not None else 'None'}")

        top_result: Dict[str, Any] | None = None

        if hits:
            # hits[0] = (anime_id, score, norm_title)
            anime_id, score, norm_title = hits[0]
            top_result = {
                "anime_id": int(anime_id),
                "score": float(score),
                "normalized_title": norm_title,
            }

        payload.append(
            {
                "query": q,
                "result": top_result,  # 없으면 None
            }
        )

    print(f"[TITLE-SEARCH] top1 payload: {payload}")
    return payload