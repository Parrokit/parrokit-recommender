# 애니메이션 추천 시스템

> **애니메이션 추천 시스템 백엔드 서버**
> Python · FastAPI · PyTorch · Matrix Factorization · Collaborative Filtering · Sentence Transformers · mBART-50 (LoRA)
> 개발 기간: 2025년 9월 ~ 2025년 12월 / 개인 프로젝트

---

## 1. 도입 배경

### 문제 인식

개인 프로젝트로 영상 쉐도잉 학습 앱을 개발하고 있었는데, 해당 앱은 애니메이션뿐만 아니라 드라마, 유튜브 등 다양한 영상 소스를 지원하는 구조였습니다. 이 앱의 확장 기능으로 애니메이션 추천 시스템을 고려했지만, 범용 영상 앱에 애니메이션 특화 추천을 결합하는 것은 구조적으로 맞지 않아 **독립 서비스로 분리**하기로 결정했습니다.

추천 시스템을 독립적으로 구축하게 된 핵심 동기는 **추천 알고리즘을 통한 사용자 유입**이었습니다. 취향에 맞는 작품을 정확하게 추천해주는 기능 자체가 유저를 끌어들이는 진입점이 될 수 있다고 판단했고, 이를 학교 프로젝트 과목과 연계하여 본격적으로 개발하게 되었습니다.

이 프로젝트는 **사용자가 좋아하는 애니메이션 제목만 입력하면, 의미 기반 검색 → 협업 필터링 추천 → 메타데이터 번역의 End-to-End 파이프라인**을 통해 개인화된 추천 결과를 제공하는 백엔드 서버를 구축하는 것을 목표로 했습니다.

### 프로젝트 목표

- 한국어 / 일본어 / 영어 등 **다국어 제목 입력을 의미 기반으로 매칭**하여 정확한 anime_id를 추출
- Matrix Factorization 기반 **협업 필터링으로 유사 취향 작품을 추천**하되, 매번 다양한 결과가 나오도록 샘플링 전략 설계
- 추천 결과의 일본어 메타데이터를 **mBART + LoRA로 한국어 번역**하여 사용자 경험 최적화
- Flutter 클라이언트와의 **WebSocket 실시간 스트리밍** 연동

---

## 2. 기술적 문제 해결 과정

### 문제 1 — 추천 결과가 매번 동일한 작품에 편중되는 문제

**상황**
Matrix Factorization 모델로 사용자 벡터와 아이템 벡터의 내적 점수를 계산한 뒤, 단순히 점수 내림차순으로 상위 K개를 잘라서 반환하는 방식이었습니다. 그 결과 인기작(예: 강철의 연금술사, 슈타인즈;게이트 등)만 반복적으로 추천되어, 어떤 입력을 넣어도 결과가 거의 동일하게 나오는 문제가 발생했습니다.

**해결**
상위 점수 아이템만 기계적으로 자르는 대신, **후보 풀(pool) 확장 + Temperature Softmax + 균등 분포 블렌딩** 전략을 도입했습니다.

```python
# 1) 후보 풀을 top_k의 diversity_factor(5)배로 확장
pool_size = min(num_items, max(top_k * diversity_factor, top_k))
pool_indices = ranked_indices[:pool_size]

# 2) Temperature Softmax로 점수를 확률 분포로 변환
shifted = pool_scores - np.max(pool_scores)
probs = np.exp(shifted / temperature)   # temperature=1.5

# 3) 균등 분포와 50:50 혼합하여 과도한 쏠림 방지
uniform = np.ones_like(probs) / len(probs)
alpha = 0.5
probs = (1.0 - alpha) * probs + alpha * uniform

# 4) 확률 기반 비복원 추출
sampled_indices = np.random.choice(pool_indices, size=top_k, replace=False, p=probs)
```

핵심 아이디어는 점수가 높은 아이템이 선택될 확률은 유지하되, **균등 분포를 혼합**함으로써 풀 내의 모든 후보에게 최소한의 선택 확률을 부여하는 것입니다. 이를 통해 매 요청마다 결과 조합이 달라지면서도, 품질 기반 랭킹이 완전히 무시되지 않는 균형을 달성했습니다.

---

### 문제 2 — WebSocket 진행 상황 스트리밍에서 발생한 레이스 컨디션

**상황**
Flutter 클라이언트에서 추천 파이프라인의 진행 상황을 실시간으로 받기 위해 WebSocket을 도입했습니다. 파이프라인의 각 단계(검색 완료 → 추천 완료 → 번역 완료)마다 이벤트를 방출하는 구조였는데, 간헐적으로 서버에서 `RuntimeError: Cannot call "send" after close` 예외가 발생했습니다.

**원인 분석**
두 가지 문제가 겹쳐 있었습니다.

1. **"done" 이벤트 이중 방출**: 서비스 레이어(`recommend_flow_service.py`)에서 `progress_cb`로 `done` 이벤트를 방출하고, 라우터(`ws_recommend_router.py`)에서도 동일한 `done` 이벤트를 보내고 있었습니다. 서비스 쪽에서 `done`을 보낸 직후 클라이언트가 소켓을 닫으면, 라우터의 `done` 전송이 이미 닫힌 소켓에 쓰기를 시도하게 됩니다.

2. **소켓 상태 미확인**: `send_json()` 호출 전에 소켓이 아직 연결 상태인지 확인하지 않았습니다.

**해결**
서비스 레이어에서 `done` 이벤트 방출을 제거하고 라우터에서만 최종 결과를 전송하도록 책임을 분리한 뒤, 모든 전송 시점에 소켓 상태를 확인하는 가드를 추가했습니다.

```python
async def async_emit(event: dict):
    # 소켓이 이미 닫혔거나 닫히는 중이면 전송하지 않음
    if websocket.client_state not in (WebSocketState.CONNECTED, WebSocketState.CONNECTING):
        return
    try:
        await websocket.send_json(event)
    except (RuntimeError, WebSocketDisconnect) as e:
        # close 이후에 늦게 호출된 send는 조용히 무시
        print(f"[ws_recommend] send after close ignored: {e}")
        return
```

> 서비스 레이어는 `progress_cb`로 중간 이벤트만 전파하고, **최종 결과 전송과 소켓 생명주기 관리는 라우터가 담당**하도록 관심사를 분리한 것이 핵심입니다.

---

### 문제 3 — 제목 검색 시 한국어 입력의 낮은 매칭 정확도

**상황**
`SentenceTransformer(paraphrase-multilingual-mpnet-base-v2)`로 애니메이션 제목을 임베딩할 때, 원어(일본어) 제목만 인덱싱하고 있었습니다. 한국어 사용자가 "진격의 거인"을 검색하면, 일본어 원제 "進撃の巨人"과의 의미 유사도가 낮아 매칭에 실패하거나 엉뚱한 결과가 나오는 문제가 있었습니다.

**해결**
데이터셋에 이미 포함된 `Korean Name` 컬럼을 우선적으로 사용하도록 임베딩 인덱스 구축 로직을 변경했습니다.

```python
# fit() 내부: Korean Name이 있으면 우선 사용, 없으면 일본어 원제로 fallback
for idx in range(len(df)):
    ko = title_ko.iloc[idx].strip() if title_ko is not None and isinstance(title_ko.iloc[idx], str) else ""
    jp = title_jp.iloc[idx].strip() if isinstance(title_jp.iloc[idx], str) else ""
    base = ko if ko else jp
    final_titles.append(normalize_title(base))
```

한국어 사용자가 한국어 제목으로 검색하면 한국어 임베딩끼리 비교되어 유사도가 높아지고, 한국어 이름이 없는 작품은 원어 제목으로 자연스럽게 fallback됩니다.

---

### 문제 4 — 번역 레이턴시가 추천 응답 속도를 지배하는 문제

**상황**
추천된 애니메이션의 제목과 시놉시스를 mBART-50 + LoRA로 실시간 번역하는 구조였습니다. 추천 10건에 대해 제목(1건) + 시놉시스(1건) = 건당 2회, 총 20회의 번역 호출이 발생했고, 각 번역에 수백 ms가 소요되어 전체 응답 시간이 수 초에 달했습니다.

**해결**
데이터셋에 미리 가공해 둔 `Korean Name` / `Korean Synopsis` 컬럼이 존재하면 **번역을 스킵**하고 캐시된 값을 즉시 반환하도록 변경했습니다.

```python
# 기존 번역 필드 확인 (이미 번역이 존재하면 번역 스킵)
existing_name_ko = row.get("Korean Name", "")
if isinstance(existing_name_ko, str) and existing_name_ko.strip():
    name_ko = existing_name_ko.strip()     # 캐시 hit → 즉시 반환
else:
    name_ko = translate_jp_to_ko(name_jp)  # 캐시 miss → 모델 추론
```

데이터 가공이 완료된 작품은 번역 추론을 완전히 건너뛰어, 응답 시간이 대폭 단축되었습니다. 아직 한국어 데이터가 없는 작품에 대해서만 모델 추론이 실행되는 **캐시 우선(cache-first) 전략**입니다.

**데이터 가공 — 프롬프트 엔지니어링 기반 한국어 데이터 생성**

캐시 전략의 전제 조건인 한국어 데이터 자체를 확보하기 위해, OpenAI API(`gpt-4.1-mini`)를 활용한 **대규모 자동 번역 파이프라인**을 별도로 구축했습니다. 55K 레코드의 애니메이션 데이터셋에서 `Korean Name` / `Korean Synopsis`가 비어 있는 항목을 Score 내림차순으로 순회하며 번역을 채워넣는 방식입니다.

```python
# 프롬프트 설계: 단순 번역이 아닌 "한국 현지화" 관점으로 유도
system_prompt = (
    "당신은 일본 애니메이션 정보를 한국어로 현지화하는 도우미입니다. "
    "한국에서 통용되는 자연스러운 애니 제목(Korean Name)과 "
    "한국어로 자연스럽게 번역된 시놉시스(Korean Synopsis)를 생성하세요."
)
```

프롬프트에 Main Name, Other Name, English Name, Synopsis를 모두 제공하여 LLM이 **문맥을 종합적으로 판단**해 한국에서 실제 통용되는 제목(예: "鋼の錬金術師" → "강철의 연금술사")을 생성하도록 했습니다. LLM 응답의 JSON 파싱 실패에 대비한 best-effort recovery 로직과, 20건 단위 중간 저장으로 장시간 실행 시 데이터 유실을 방지하는 안전장치도 포함했습니다.

---

### 문제 5 — Docker 이미지 크기 및 빌드 시간 최적화

**상황**
초기 Dockerfile은 단일 스테이지로 구성되어 있었습니다. `build-essential`(컴파일러, 헤더 등)이 런타임 이미지에 그대로 포함되어 이미지 크기가 불필요하게 커졌고, 의존성 설치 시마다 소스 컴파일이 반복되어 빌드 시간도 길었습니다.

**해결**
**Multi-stage build** 패턴을 적용하여 빌드 단계와 런타임 단계를 분리했습니다.

```dockerfile
# 1) Builder: 컴파일러 포함, wheel 파일만 생성
FROM python:3.10-slim AS builder
RUN apt-get install -y build-essential
RUN pip wheel --wheel-dir /wheels -r requirements.txt

# 2) Runtime: 컴파일러 없이 wheel만 설치
FROM python:3.10-slim AS runtime
COPY --from=builder /wheels /wheels
RUN pip install /wheels/* && rm -rf /wheels
```

빌드 결과물(wheel)만 런타임 이미지로 복사하여 컴파일러와 개발 도구가 최종 이미지에서 완전히 제거됩니다. 런타임에 불필요한 바이너리가 없어 이미지 크기가 줄고, 레이어 캐시 효율도 개선되었습니다.

---

### 문제 6 — CI/CD 파이프라인 구축 과정에서의 반복적 디버깅

**상황**
GitHub Actions + OCI(Oracle Cloud) / 자택 서버(WSL) 이중 배포 환경을 구축하는 과정에서, 환경별로 다른 인증 방식(SSH 키 vs 패스워드), 권한 체계(sudo 필요 여부), 런타임 형태(Docker vs venv)를 하나의 워크플로우로 통합해야 했습니다. 이 과정에서 다수의 시행착오를 거쳤습니다.

**주요 이슈 및 해결**

| 이슈 | 원인 | 해결 |
|------|------|------|
| `.env`가 Docker 빌드에 포함되지 않음 | Docker build는 GitHub Actions Runner에서 실행되므로 VM의 `.env`에 접근 불가 | `secrets`에 env 파일 내용을 저장하고, 배포 시 SSH로 VM에 직접 기록 |
| CI에서 Docker build 실패 | ML/DL 의존성(PyTorch 등)으로 Actions Runner 디스크 용량 초과 | CI에서는 Docker 빌드를 비활성화(`if: false`)하고, lint/test만 실행 |
| OCI vs 자택 서버의 환경 차이 | sudo 유무, SSH 인증 방식, 포트, GPU 유무 등이 모두 다름 | `workflow_dispatch`에 target/runtime/accel 파라미터를 두고 분기 처리 |

최종적으로 `workflow_dispatch` 기반의 **메뉴 방식 CD**를 구현하여, 배포 대상(OCI / MyHome)과 런타임(Docker / venv / none), 가속기(CPU / GPU)를 선택할 수 있는 유연한 구조를 갖추었습니다.

---

## 3. 아키텍처 설계

```bash
┌──────────────────────────────────────────────────────────────────┐
│                   Client (Flutter App)                           │
│  WebSocket /ws/recommend ── 실시간 진행 상황 스트리밍                  │
│  REST /flow/titles-to-metadata ── 일괄 요청                        │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│                    FastAPI (app/main.py)                         │
│  Lifespan: 서버 기동 시 데이터 다운로드 + 모델 4종 로드                   │
│  Router: title_search / mf_recommend / translator / flow / ws    │
└───────┬──────────────────┬──────────────────┬────────────────────┘
        │                  │                  │
┌───────▼───────┐  ┌───────▼───────┐  ┌───────▼───────┐
│ Title Search  │  │ MF Recommend  │  │  Translator   │
│               │  │               │  │               │
│ Sentence      │  │ Matrix        │  │ mBART-50      │
│ Transformer   │  │ Factorization │  │ + LoRA        │
│ (384-dim)     │  │ (64-dim)      │  │ (ja→ko)       │
│               │  │               │  │               │
│ 코사인 유사도    │  │ 내적 + 확률      │  │ Beam Search   │
│ 기반 매칭       │  │ 기반 샘플링      │  │ + 캐시 우선      │
└───────────────┘  └───────────────┘  └───────────────┘
        │                  │                  │
┌───────▼──────────────────▼──────────────────▼────────────────────┐
│                      Data Layer                                  │
│  anime-dataset-2023.csv (55K records)                            │
│  users-score-2023.csv (24.3M ratings → top 500 users)            │
│  mf_weight.pt / mbart_lora adapter                               │
│  OCI Object Storage에서 최초 기동 시 자동 다운로드                      │
└──────────────────────────────────────────────────────────────────┘
```

**레이어 분리 원칙**

- `app/api/` — 라우팅, 요청/응답 직렬화, WebSocket 생명주기 관리
- `app/services/` — 비즈니스 로직, 파이프라인 오케스트레이션
- `app/models/` — ML 모델 정의 및 추론 로직
- `app/dependencies.py` — 전역 싱글턴 모델 인스턴스 관리
- `app/init_data.py` — 데이터/가중치 자동 다운로드 및 초기화

---

## 4. 개발 과정

| 단계 | 내용 |
|------|------|
| **기본 셋팅** | FastAPI + Uvicorn 프로젝트 구조, Conda 환경, CI/CD 파이프라인 구축 |
| **Title Search** | SentenceTransformer 기반 다국어 제목 검색 구현 |
| **MF Recommend** | Matrix Factorization 모델 학습 및 추론 서비스 연동 |
| **Translator** | mBART-50 + LoRA 파인튜닝, 일본어→한국어 번역 서비스 연동 |
| **통합 파이프라인** | 검색 → 추천 → 번역의 End-to-End 플로우 구축 |
| **실시간 스트리밍** | WebSocket 기반 진행 상황 스트리밍 + 레이스 컨디션 수정 |
| **품질 개선** | Korean 우선 임베딩, 번역 캐시 전략, 추천 다양성 확보 |
| **인프라 최적화** | Docker multi-stage build, OCI/자택 서버 이중 배포 |

---

## 5. 성과 및 배운 점

### 기술적 성과

- **3단계 ML 파이프라인**을 하나의 FastAPI 서버에 통합하여, 제목 입력만으로 한국어 메타데이터가 포함된 추천 결과를 반환하는 End-to-End 시스템 구현
- **Temperature Softmax + 균등 분포 블렌딩**으로 협업 필터링의 추천 다양성 문제를 해결하여, 매 요청마다 다른 조합의 추천 결과 생성
- **WebSocket 레이스 컨디션**을 소켓 상태 가드와 이벤트 방출 책임 분리로 해결
- **캐시 우선 번역 전략**으로 불필요한 모델 추론을 제거하여 응답 속도 개선
- Docker multi-stage build로 이미지 최적화, `workflow_dispatch` 기반 유연한 배포 파이프라인 구축

### 배운 점

- ML 모델의 추론 결과가 "정확하더라도" 사용자 경험 관점에서는 문제가 될 수 있다는 것을 체감 — 점수 1등이 항상 최선은 아니며, **다양성과 정확도 사이의 트레이드오프**를 설계해야 함
- WebSocket 기반 실시간 통신에서 **비동기 컨텍스트와 소켓 생명주기 관리**의 중요성 — 동기 서비스 레이어에서 비동기 콜백을 호출할 때 발생할 수 있는 타이밍 이슈를 직접 경험하고 해결
- CI/CD 파이프라인은 한 번에 완성되지 않으며, 환경별 차이(인증 방식, 권한, 디스크 용량)를 **반복적으로 디버깅**하면서 점진적으로 안정화하는 과정 경험
- Pre-trained 모델(mBART, SentenceTransformer)을 그대로 쓰는 것과 **LoRA로 도메인 특화 파인튜닝**하는 것의 차이, 그리고 소량 데이터(49 pairs)로도 의미 있는 개선이 가능하다는 점

---

## 6. 개선 가능한 부분

| 항목 | 현황 | 개선 방향 |
|------|------|-----------|
| 추천 모델 | MF만 사용, 콘텐츠 특성 미반영 | 장르/시놉시스 기반 Content-Based Filtering 하이브리드 |
| 번역 품질 | LoRA 학습 데이터 49쌍으로 제한적 | 학습 데이터 확충 및 시놉시스 전용 어댑터 별도 학습 |
| 데이터 저장소 | CSV 파일 기반, 인메모리 로드 | PostgreSQL 또는 벡터 DB(Milvus 등)로 전환 |
| 동시성 | 전역 싱글턴 모델, 단일 프로세스 Uvicorn | 모델 풀링 또는 Gunicorn worker 다중화 |
| 번역 학습 데이터 | 하드코딩된 49쌍 | 외부 데이터셋 또는 크라우드소싱 기반 수집 |
| 학습 파이프라인 | API 엔드포인트에서 동기 실행 (블로킹) | Celery 등 비동기 태스크 큐 도입 |
