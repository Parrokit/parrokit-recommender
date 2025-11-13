# app/services/mf_recommend_service.py
from typing import List
import torch
import numpy as np

from app import dependencies as deps


def recommend_anime_from_ids(
    liked_anime_ids: List[int],
    top_k: int = 20,
    exclude_watched: bool = True,
) -> List[int]:
    """
    사용자가 본(또는 좋아한) anime_id 리스트를 받아
    MF 임베딩 평균으로 유저 벡터를 구성하고
    상위 top_k개 추천 anime_id를 반환.
    """
    if deps.recommender is None or deps.mf_items is None or deps.anime_id_to_item_idx is None:
        raise RuntimeError("MF 모델 또는 매핑이 초기화되지 않았습니다. init_models()가 먼저 호출되어야 합니다.")

    model = deps.recommender
    mf_items = deps.mf_items
    anime_id_to_item_idx = deps.anime_id_to_item_idx

    device = next(model.parameters()).device

    # 1) 사용자가 준 anime_id → 내부 item_idx로 변환
    item_indices = []
    for aid in liked_anime_ids:
        idx = anime_id_to_item_idx.get(int(aid))
        if idx is not None:
            item_indices.append(idx)

    if not item_indices:
        # 매핑되는 애니가 하나도 없으면 추천 불가
        raise ValueError("입력한 anime_id 중 MF 모델이 알고 있는 항목이 없습니다.")

    idx_tensor = torch.tensor(item_indices, device=device, dtype=torch.long)

    # 2) 선택한 애니들의 임베딩 평균으로 유저 벡터 근사
    item_vecs = model.item_factors(idx_tensor)     # (n_selected, factors)
    user_vec = item_vecs.mean(dim=0)               # (factors,)

    # 3) 모든 아이템에 대한 점수 계산 (내적)
    with torch.no_grad():
        scores = model.item_factors.weight @ user_vec  # (num_items,)
        scores_np = scores.detach().cpu().numpy()

    # 4) 이미 본 애니는 제외할지 옵션
    watched_set = set(item_indices) if exclude_watched else set()

    ranked_indices = np.argsort(scores_np)[::-1]  # 내림차순 정렬
    recommended_indices = [
        int(i)
        for i in ranked_indices
        if i not in watched_set
    ][:top_k]

    # 5) 내부 item_idx -> 원래 anime_id로 변환
    recommended_anime_ids = [int(mf_items[i]) for i in recommended_indices]
    return recommended_anime_ids