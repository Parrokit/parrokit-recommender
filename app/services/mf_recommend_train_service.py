# app/services/mf_train_service.py
import os
import time
from typing import Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from app.models.matrix_factorzation import MatrixFactorization


class RatingsDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        super().__init__()
        self.users = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(df["rating"].values, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.ratings)

    def __getitem__(self, idx: int):
        return self.users[idx], self.items[idx], self.ratings[idx]


def select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def train_mf_model(
    csv_path: str = "app/data/animelist-dataset/users-score-2023.csv",
    top_k_users: int = 500,
    factors: int = 64,
    epochs: int = 10,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    save_path: str = "app/models/weights/mf_weight.pt",
) -> Dict[str, Any]:
    """
    MF 모델을 학습하고 가중치를 저장하는 함수.
    FastAPI에서 바로 호출할 수 있도록 서비스 레벨로 분리.
    """
    t0 = time.time()
    device = select_device()
    print(f"[MF TRAIN] device = {device}")

    # 1) 데이터 로드
    ratings = (
        pd.read_csv(csv_path, usecols=["user_id", "anime_id", "rating"])
        .dropna()
        .query("rating > 0")
    )

    # 상위 K명 유저만 사용
    top_users = ratings["user_id"].value_counts().head(top_k_users).index
    filtered = ratings[ratings["user_id"].isin(top_users)]

    # user_id / anime_id -> 연속 인덱스로 factorize
    user_ids, users = pd.factorize(filtered["user_id"])
    item_ids, items = pd.factorize(filtered["anime_id"])

    filtered = filtered.assign(user_idx=user_ids, item_idx=item_ids)
    n_users, n_items = len(users), len(items)
    print(f"[MF TRAIN] n_users={n_users}, n_items={n_items}")

    # 2) train / valid / test split
    train_x, tmp_df = train_test_split(
        filtered[["user_idx", "item_idx", "rating"]],
        test_size=0.3,
        random_state=42,
        stratify=filtered["user_idx"],
    )
    valid_x, test_x = train_test_split(
        tmp_df,
        test_size=0.5,
        random_state=42,
        stratify=tmp_df["user_idx"],
    )

    train_loader = DataLoader(
        RatingsDataset(train_x), batch_size=batch_size, shuffle=True
    )
    valid_loader = DataLoader(
        RatingsDataset(valid_x), batch_size=batch_size * 2
    )
    test_loader = DataLoader(
        RatingsDataset(test_x), batch_size=batch_size * 2
    )

    # 3) 모델 / 옵티마이저 준비
    model = MatrixFactorization(n_users, n_items, factors=factors).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    criterion = torch.nn.MSELoss()

    train_rmse_list = []
    valid_rmse_list = []

    # 4) 학습 루프
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for users_batch, items_batch, ratings_batch in train_loader:
            users_batch = users_batch.to(device)
            items_batch = items_batch.to(device)
            ratings_batch = ratings_batch.to(device)

            preds = model(users_batch, items_batch)
            loss = criterion(preds, ratings_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(ratings_batch)

        train_rmse = (total_loss / len(train_x)) ** 0.5
        train_rmse_list.append(float(train_rmse))

        # --- 검증 ---
        model.eval()
        total_valid = 0.0
        with torch.no_grad():
            for users_batch, items_batch, ratings_batch in valid_loader:
                users_batch = users_batch.to(device)
                items_batch = items_batch.to(device)
                ratings_batch = ratings_batch.to(device)

                preds = model(users_batch, items_batch)
                loss = criterion(preds, ratings_batch)
                total_valid += loss.item() * len(ratings_batch)

        valid_rmse = (total_valid / len(valid_x)) ** 0.5
        valid_rmse_list.append(float(valid_rmse))

        print(
            f"[Epoch: {epoch+1:03d}] "
            f"train RMSE {train_rmse:.3f} | valid RMSE {valid_rmse:.3f}"
        )

    # 5) 최종 테스트 RMSE
    model.eval()
    total_test = 0.0
    with torch.no_grad():
        for users_batch, items_batch, ratings_batch in test_loader:
            users_batch = users_batch.to(device)
            items_batch = items_batch.to(device)
            ratings_batch = ratings_batch.to(device)

            preds = model(users_batch, items_batch)
            loss = criterion(preds, ratings_batch)
            total_test += loss.item() * len(ratings_batch)

    test_rmse = (total_test / len(test_x)) ** 0.5
    test_rmse = float(test_rmse)

    # 6) 가중치 저장
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    elapsed = time.time() - t0
    print(f"[MF TRAIN] done, saved to {save_path} | {elapsed:.1f}s")

    # API 응답으로 쓸 수 있는 요약 리턴
    return {
        "n_users": int(n_users),
        "n_items": int(n_items),
        "epochs": int(epochs),
        "train_rmse": train_rmse_list,
        "valid_rmse": valid_rmse_list,
        "test_rmse": test_rmse,
        "save_path": save_path,
        "elapsed_sec": elapsed,
        "device": device,
        "top_k_users": int(top_k_users),
        "factors": int(factors),
    }