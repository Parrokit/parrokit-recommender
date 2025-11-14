# app/services/mf_recommend_service.py
from typing import List
import torch
import numpy as np

from app import dependencies as deps


def recommend_anime_from_ids(
    liked_anime_ids: List[int],
    top_k: int = 20,
    exclude_watched: bool = True,
    diversity_factor: int = 5,
    temperature: float = 1.5,
) -> List[int]:
    """
    사용자가 본(또는 좋아한) anime_id 리스트를 받아
    MF 임베딩 평균으로 유저 벡터를 구성하고
    상위 top_k개 추천 anime_id를 반환.

    기존에는 단순히 점수 내림차순으로 상위 top_k를 잘라서 반환했지만,
    이제는 상위 pool 내에서 점수 비례 확률로 샘플링하여
    매번 약간씩 다른 조합이 나오도록 함.
    """
    if deps.recommender is None or deps.mf_items is None or deps.anime_id_to_item_idx is None:
        raise RuntimeError("MF 모델 또는 매핑이 초기화되지 않았습니다. init_models()가 먼저 호출되어야 합니다.")

    model = deps.recommender
    mf_items = deps.mf_items
    anime_id_to_item_idx = deps.anime_id_to_item_idx

    device = next(model.parameters()).device

    # 1) 사용자가 준 anime_id → 내부 item_idx로 변환
    item_indices: List[int] = []
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
        scores_np = scores.detach().cpu().numpy()      # (num_items,)

    num_items = scores_np.shape[0]

    # 4) 이미 본 애니는 제외할지 옵션
    watched_set = set(item_indices) if exclude_watched else set()
    if watched_set:
        # 이미 본 항목은 사실상 선택되지 않도록 매우 작은 값으로 설정
        for w_idx in watched_set:
            if 0 <= w_idx < num_items:
                scores_np[w_idx] = -1e9

    # 5) 상위 pool 내에서 점수 비례 랜덤 샘플링
    #    - pool_size: top_k의 diversity_factor 배수만큼 후보 풀을 잡음
    #    - temperature: softmax 온도 (작을수록 상위 점수에 더 쏠림)
    pool_size = min(num_items, max(top_k * diversity_factor, top_k))

    # 점수 내림차순 정렬 후 상위 pool_size 인덱스 추출
    ranked_indices = np.argsort(scores_np)[::-1]
    pool_indices = ranked_indices[:pool_size]

    if len(pool_indices) == 0:
        # 극단적인 경우: 모든 점수가 -inf 비슷하게 되어버린 경우
        raise ValueError("추천 후보를 만들 수 없습니다. 점수 계산 결과가 비정상입니다.")

    # pool 내 점수
    pool_scores = scores_np[pool_indices]

    # 수치 안정성을 위해 최대값을 빼고 softmax 계산
    shifted = pool_scores - np.max(pool_scores)
    # temperature가 너무 작아지는 것을 방지
    temp = max(temperature, 1e-6)
    probs = np.exp(shifted / temp)

    probs_sum = probs.sum()
    if probs_sum <= 0 or not np.isfinite(probs_sum):
        # 모든 값이 0 또는 비정상적인 경우 → 균일 분포로 fallback
        probs = np.ones_like(probs) / len(probs)
    else:
        probs /= probs_sum

    # 점수 비례 softmax와 균일 분포를 혼합해서
    # 특정 상위 아이템(예: 강연금, 슈타게 등)에만 과도하게 쏠리지 않도록 완화
    if len(probs) > 0:
        uniform = np.ones_like(probs) / len(probs)
        alpha = 0.5  # 0이면 pure softmax, 1이면 pure uniform
        probs = (1.0 - alpha) * probs + alpha * uniform

        # 수치 오차를 방지하기 위해 한 번 더 정규화
        probs_sum = probs.sum()
        if probs_sum <= 0 or not np.isfinite(probs_sum):
            probs = np.ones_like(probs) / len(probs)
        else:
            probs /= probs_sum

    # 중복 없이 top_k개를 샘플링 (pool 크기가 top_k보다 작으면 가능한 만큼만)
    choose_k = min(top_k, len(pool_indices))
    sampled_indices = np.random.choice(
        pool_indices,
        size=choose_k,
        replace=False,
        p=probs,
    )

    # 6) 내부 item_idx -> 원래 anime_id로 변환
    recommended_anime_ids = [int(mf_items[int(i)]) for i in sampled_indices]
    return recommended_anime_ids