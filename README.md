# Parrokit Anime Recommender

애니메이션 추천 시스템 백엔드 서버.
사용자가 좋아하는 애니메이션 제목을 입력하면, **의미 기반 검색 → 협업 필터링 추천 → 메타데이터 번역**의 파이프라인을 거쳐 개인화된 추천 결과를 제공한다.

## 기술 스택

| 구분 | 내용 |
|------|------|
| Framework | FastAPI + Uvicorn |
| Language | Python 3.10 |
| ML | PyTorch, Sentence Transformers, mBART-50 (LoRA) |
| Infra | Docker (multi-stage build) |

## 아키텍처

```
사용자 입력 (애니 제목)
       ↓
[1] Title Search — SentenceTransformer(paraphrase-multilingual-mpnet-base-v2)로 의미 유사도 검색
       ↓
[2] Recommendation — Matrix Factorization 임베딩 기반 협업 필터링 + 다양성 샘플링
       ↓
[3] Translation — mBART-50 + LoRA로 일본어 메타데이터를 한국어로 번역
       ↓
추천 결과 (제목, 시놉시스, 장르, 점수, 이미지 등)
```

### 핵심 모듈

- **Title Search** — 다국어 임베딩(384차원)을 활용한 코사인 유사도 기반 애니메이션 제목 매칭
- **Matrix Factorization** — 사용자-아이템 잠재 벡터(64차원) 내적으로 선호도 예측. Temperature softmax + 균등 분포 블렌딩으로 추천 다양성 확보
- **Anime Translator** — mBART-50에 LoRA(r=16, α=32) 어댑터를 얹어 일본어→한국어 제목/시놉시스 번역

## API

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/titles/search` | 단일 제목 검색 |
| POST | `/titles/batch-search` | 복수 제목 일괄 검색 |
| POST | `/mf/recommend` | 애니 ID 기반 추천 |
| POST | `/mf/train` | MF 모델 재학습 |
| POST | `/translator/train` | 번역 모델 파인튜닝 |
| POST | `/translator/anime-metadata` | 메타데이터 번역 |
| POST | `/flow/titles-to-metadata` | 전체 파이프라인 (검색→추천→번역) |
| WS | `/ws/recommend` | 실시간 스트리밍 추천 |

## 시작하기

### 환경 구성

```bash
conda create -n parrokit-recommender python=3.10
conda activate parrokit-recommender
```

### 의존성 설치

```bash
pip install -r requirements.txt
```

### 환경 변수

`.env` 파일에 데이터/가중치 번들 URL을 설정한다. 서버 최초 기동 시 자동으로 다운로드된다.

```env
DATA_BUNDLE_URL=<OCI Object Storage URL>
WEIGHTS_BUNDLE_URL=<OCI Object Storage URL>
```

### 실행

```bash
uvicorn app.main:app --reload
```

Makefile로도 가능하다.

```bash
make dev
```

### Docker

```bash
docker build -t parrokit-recommender .
docker run -p 8000:8000 parrokit-recommender
```