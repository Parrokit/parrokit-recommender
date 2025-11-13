# app/dependencies.py
import time
import logging
import pandas as pd
import torch

from app.models.anime_title_searcher import AnimeTitleSearcher
from app.models.matrix_factorzation import MatrixFactorization

title_searcher = None
recommender = None
translator_model = None
translator_tokenizer = None


async def init_models():
    global title_searcher, recommender, translator_model, translator_tokenizer

    print("\n==== [INIT] 모델 초기화 시작 ====\n")
    total_start = time.perf_counter()

    # -----------------------------
    # 1) Anime Title Searcher 로드
    # -----------------------------
    t1 = time.perf_counter()
    print("[1] Anime Metadata 불러오는 중...")

    titles_df = pd.read_csv("app/data/animelist-dataset/anime-dataset-2023.csv")

    print(f"[1] CSV 로드 완료: {len(titles_df)} rows")

    print("[1] AnimeTitleSearcher 임베딩 구축 중 (SentenceTransformer 로딩 포함)...")
    title_searcher = AnimeTitleSearcher().fit(titles_df)

    t1_end = time.perf_counter()
    print(f"[1] AnimeTitleSearcher 준비 완료 (소요시간: {t1_end - t1:.3f}초)\n")

    # -----------------------------
    # 2) Matrix Factorization 추천 모델
    # -----------------------------
    t2 = time.perf_counter()
    print("[2] 추천 모델(MF) 가중치 로드 중...")

    state = torch.load("app/models/weights/mf_weight.pt", map_location="cpu", weights_only=True)

    num_users = state["user_factors.weight"].shape[0]
    num_items = state["item_factors.weight"].shape[0]

    print(f"[2] num_users={num_users}, num_items={num_items}")

    recommender = MatrixFactorization(num_users, num_items, factors=64)
    recommender.load_state_dict(state)
    recommender.eval()

    t2_end = time.perf_counter()
    print(f"[2] MF 추천 모델 준비 완료 (소요시간: {t2_end - t2:.3f}초)\n")

    # -----------------------------
    # 3) 번역 모델 (주석된 부분)
    # -----------------------------
    # t3 = time.perf_counter()
    # print("[3] 번역 모델 로드 중...")
    # translator_model, translator_tokenizer = load_translator_model()
    # t3_end = time.perf_counter()
    # print(f"[3] 번역 모델 로드 완료 (소요시간: {t3_end - t3:.3f}초)\n")

    # -----------------------------
    # 전체 시간
    # -----------------------------
    total_end = time.perf_counter()
    print(f"\n==== [INIT] 전체 모델 로드 완료 | 총 소요시간: {total_end - total_start:.3f}초 ====\n")