# pip install sentence-transformers
import numpy as np
import pandas as pd
from typing import List, Tuple, Iterable, Optional
from sentence_transformers import SentenceTransformer
from app.utils.normalize_title import normalize_title

class AnimeTitleSearcher:
    """제목 임베딩 기반 애니 검색기 (클래스 버전)

    - SentenceTransformer 임베딩으로 인덱스를 구성하고, 단건/배치 검색을 제공
    - 기존 함수형 구현(search_title/batch_search)을 객체 메서드로 통합
    """

    def __init__(
        self,
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        normalize_embeddings: bool = True,
    ) -> None:
        # 모델은 한 번만 로드
        self.model_name = model_name
        self.normalize_embeddings = normalize_embeddings
        self.model: Optional[SentenceTransformer] = SentenceTransformer(model_name)

        # 인덱스 관련 버퍼
        self.emb: Optional[np.ndarray] = None  # (N, d)
        self.docs: List[str] = []              # 정규화된 제목 문자열
        self.id_map: List[int] = []            # anime_id 리스트
        self.titles_df: Optional[pd.DataFrame] = None

    # ---------- 인덱스 구축 ----------
    def fit(self, df: pd.DataFrame, title_col: str = "Name", id_col: str = "anime_id") -> "AnimeTitleSearcher":
        """주어진 DataFrame으로 인덱스를 재구성합니다."""
        assert title_col in df.columns and id_col in df.columns, f"DataFrame must contain '{title_col}' and '{id_col}'"
        self.titles_df = df[[id_col, title_col]].copy()
        self.docs = [normalize_title(t) for t in df[title_col].tolist()]
        self.id_map = df[id_col].tolist()
        self.emb = self.model.encode(
            self.docs,
            normalize_embeddings=self.normalize_embeddings,
        ).astype("float32")
        return self

    @classmethod
    def from_titles(
        cls,
        titles: Iterable[dict],
        model_name: str = "paraphrase-multilingual-mpnet-base-v2",
        normalize_embeddings: bool = True,
    ) -> "AnimeTitleSearcher":
        """[{"anime_id": int, "title": str}, ...] 리스트로 바로 생성"""
        df = pd.DataFrame(titles)
        return cls(model_name=model_name, normalize_embeddings=normalize_embeddings).fit(df)

    # ---------- 단건 검색 ----------
    def search(self, query: str, k: int = 5, cutoff: float = 0.55) -> List[Tuple[int, float, str]]:
        """질의 문자열로 상위 k개 결과 반환: [(anime_id, score, matched_norm_title), ...]"""
        assert self.emb is not None and len(self.id_map) > 0, "Index is empty. Call fit() first."
        qn = normalize_title(query)
        qv = self.model.encode([qn], normalize_embeddings=self.normalize_embeddings).astype("float32")[0]
        sims = self.emb @ qv  # 코사인 유사도 (normalize_embeddings=True 전제)
        k = min(k, len(sims))
        idx = np.argpartition(-sims, k - 1)[:k]
        idx = idx[np.argsort(-sims[idx])]
        hits = [(self.id_map[i], float(sims[i]), self.docs[i]) for i in idx if sims[i] >= cutoff]
        return hits

    # ---------- 배치 검색 ----------
    def batch_search(self, queries: List[str], k: int = 5, cutoff: float = 0.55) -> List[List[Tuple[int, float, str]]]:
        """여러 질의를 한 번에 검색"""
        assert self.emb is not None and len(self.id_map) > 0, "Index is empty. Call fit() first."
        qn = [normalize_title(q) for q in queries]
        qv = self.model.encode(qn, normalize_embeddings=self.normalize_embeddings).astype("float32")  # (B, d)
        sims = qv @ self.emb.T  # (B, N)
        results: List[List[Tuple[int, float, str]]] = []
        for r in range(sims.shape[0]):
            row = sims[r]
            kk = min(k, len(row))
            idx = np.argpartition(-row, kk - 1)[:kk]
            idx = idx[np.argsort(-row[idx])]
            hits = [(self.id_map[i], float(row[i]), self.docs[i]) for i in idx if row[i] >= cutoff]
            results.append(hits)
        return results

    # ---------- 인덱스 확장(옵션) ----------
    def add_titles(self, new_titles: Iterable[dict], title_col: str = "title", id_col: str = "anime_id") -> None:
        """새로운 타이틀들을 인덱스에 추가(증분 인코딩)."""
        df_new = pd.DataFrame(new_titles)
        assert title_col in df_new.columns and id_col in df_new.columns, f"DataFrame must contain '{title_col}' and '{id_col}'"
        docs_new = [normalize_title(t) for t in df_new[title_col].tolist()]
        ids_new = df_new[id_col].tolist()
        emb_new = self.model.encode(docs_new, normalize_embeddings=self.normalize_embeddings).astype("float32")
        # concat
        if self.emb is None:
            self.emb = emb_new
            self.docs = docs_new
            self.id_map = ids_new
        else:
            self.emb = np.vstack([self.emb, emb_new])
            self.docs.extend(docs_new)
            self.id_map.extend(ids_new)