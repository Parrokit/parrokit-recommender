# app/services/recommend_flow_service.py

from typing import List, Dict, Any

from app.services.title_search_infer_service import batch_search_anime_titles_top1
from app.services.mf_recommend_infer_service import recommend_anime_from_ids
from app.services.anime_translator_infer_service import translate_anime_metadata_by_ids


def recommend_from_titles_with_metadata(
    titles: List[str],
    top_k: int = 10,
    cutoff: float = 0.55,
    exclude_watched: bool = True,
) -> Dict[str, Any]:
    """
    1) 사용자가 입력한 제목 리스트 -> 각 제목당 top1 anime_id 뽑기
    2) 그 anime_id 들로 MF 추천
    3) 추천된 anime_id 들에 대해 Name / Synopsis 번역해서 반환
    """

    # 1) 제목 검색 (top-1 결과만)
    search_payload = batch_search_anime_titles_top1(
        queries=titles,
        k=5,
        cutoff=cutoff,
    )

    # search_payload 가 dict인지 / list인지 먼저 정규화
    #  - dict 인 경우: {"queries": [...], "k": ..., "cutoff": ..., "results": [...]}
    #  - list 인 경우: [{"query": "...", "result": {...}}, ...]
    if isinstance(search_payload, dict):
        items_iter = search_payload.get("results", [])
    else:
        items_iter = search_payload

    mapped: List[Dict[str, Any]] = []
    liked_ids: List[int] = []  

    for item in items_iter:
        # 혹시 item 이 dict 가 아닐 수도 있으니 방어 코드
        if not isinstance(item, dict):
            print(f"[WARN] unexpected item type in search_payload: {type(item)} -> {item}")
            continue

        q = item.get("query")
        top = item.get("result")  # dict 또는 None

        mapped.append(
            {
                "query": q,
                "top_hit": top,
            }
        )

        if isinstance(top, dict) and "anime_id" in top:
            liked_ids.append(int(top["anime_id"]))

    if not liked_ids:
        raise ValueError("입력 제목들로부터 유효한 anime_id를 하나도 찾지 못했습니다.")

    # 2) MF 추천
    rec_ids = recommend_anime_from_ids(
        liked_anime_ids=liked_ids,
        top_k=top_k,
        exclude_watched=exclude_watched,
    )

    # 3) 추천 anime_id 들에 대한 메타데이터 번역
    translated_meta = translate_anime_metadata_by_ids(rec_ids)

    return {
        "input_titles": titles,
        "search": mapped,
        "recommend": {
            "input_anime_ids": liked_ids,
            "recommended_anime_ids": rec_ids,
        },
        "translated_metadata": translated_meta,
    }