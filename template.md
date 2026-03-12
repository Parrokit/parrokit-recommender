# 대학 학점 자동 조회 시스템

> **대전대학교 성적 조회 시각화 서비스**  
> Python · Django · Selenium · Celery · Chart.js  
> 개발 기간: 2023년 05월 ~ 06월 / 개인 프로젝트

---

## 1. 도입 배경

### 문제 인식

대전대학교 통합정보시스템은 성적 조회 기능을 제공하지만, UI 구조가 복잡하고 데이터를 **직관적으로 파악하기 어려운 형태**로만 제공합니다. 특히 여러 학기에 걸친 성적 분포나 평균 학점의 추이를 한눈에 확인하려면 직접 계산이 필요했습니다.

학번과 비밀번호만 입력하면 **전체 학기 성적을 자동으로 수집하고 시각화**해주는 개인 도구를 만들고자 했습니다.

### 프로젝트 목표

- 포털 자동 로그인 → 전체 성적 크롤링 → 차트 시각화의 **End-to-End 자동화**
- 학기별 등급 분포를 도넛 차트로 시각화하여 **직관적인 성적 분석** 제공
- 보안 상 수집한 개인정보(학번, 비밀번호, 성적 데이터)를 **서버에 저장하지 않는** 구조 설계

---

## 2. 기술적 문제 해결 과정

### 문제 1 — 대학 포털의 인증 우회 불가

**상황**  
대전대 통합정보시스템은 포털 로그인 이후 별도의 세션/토큰 인증을 거치는 구조입니다. 단순 HTTP 요청으로 XML 파싱 또는 API 호출을 시도했으나 인증 차단으로 모두 실패했습니다.

**해결**  
브라우저 자동화 도구인 **Selenium**을 채택하여 실제 사용자가 브라우저를 사용하는 방식으로 동작하도록 구현했습니다. 핵심 우회 포인트는 다음과 같습니다.

```python
# 통합 정보 시스템 인증 절차 때문에 xml파일 파싱 불가. 새 탭을 여는 방식
driver.find_element(By.XPATH, "//*[@id='homeLink']").click()
driver.get("https://itics.dju.ac.kr/main.do")
driver.switch_to.window(driver.window_handles[0])
```

포털 로그인 직후 새 창으로 itics에 접근하는 방식으로 토큰 오류 없이 세션을 유지하는 데 성공했습니다.

> **Prototype/Selenium.py**에서 `driver.switch_to.window(driver.window_handles[-1])`를 사용하면 토큰 오류가 빈번하게 발생하는 문제를 실험적으로 발견하고, `window_handles[0]`을 사용하는 방식으로 개선했습니다. 이는 homeLink로 인해 탭이 새로 열리게 되는데, (-1) 인 새탭으로 이동하게 되면 세션 컨텍스트 토큰 오류가 발생하여, 기존에 있던 탭에서 url을 이동시킨 것입니다.

---

### 문제 2 — BeautifulSoup으로 파싱하기 어려운 동적 페이지 구조

**상황**  
itics 성적 조회 페이지는 JavaScript 렌더링 기반의 동적 컴포넌트 구조를 사용합니다. 각 셀의 HTML ID가 아래와 같이 복잡한 패턴으로 생성됩니다.

```
INFODIV01_INFODIV01_DG_GRID00_body_gridrow_0_cell_0_2GridCellTextContainerElement
```

**해결**  
Python 정규식(`re.compile`)을 적극적으로 활용해 학기 인덱스(`i`)와 열 인덱스(`j`)를 기반으로 패턴을 동적으로 생성하여 모든 학기의 전체 과목을 순회 파싱하는 로직을 구현했습니다.

```python
finds_text = soup.find_all(id=re.compile(
    'INFODIV01_INFODIV01_DG_GRID{0}_body_gridrow_._cell_._{1}GridCellTextContainerElement'
    .format(str(i).zfill(2), j)
))
```

최대 16개 학기 × 12개 열 데이터를 자동으로 수집하도록 설계했으며, `try-except`로 마지막 학기 / 마지막 과목의 경계 조건을 자연스럽게 처리했습니다.

---

### 문제 3 — Selenium 크롤링 중 웹 서버가 블로킹되는 문제

**상황**  
Selenium 크롤링은 실행 시간이 평균 10~30초 걸립니다. Django View에서 동기적으로 실행하면 그 시간 동안 HTTP 요청이 블로킹되어 브라우저가 응답 없음 상태가 됩니다.

**해결 과정**

1. **1차 시도**: Django `asyncio` 사용 → Selenium이 내부적으로 동기 드라이버를 사용하므로 실질적인 비동기 효과 없음
2. **최종 해결**: **Celery** 비동기 태스크 큐 도입

```python
# Selenium 함수를 Celery shared_task로 선언
@shared_task
def get_selenium(id, passwd, url):
    ...
```

크롤링 요청 → 로딩 페이지 즉시 반환 → 클라이언트 측 Ajax 폴링으로 완료 확인의 비동기 플로우를 구성했습니다.

**비동기 플로우**

```bash
[POST /] 학번/비밀번호 입력
    ↓ redirect
[GET /loading/] 로딩 페이지 렌더링
    ↓ JS Ajax POST
[POST /get_score/] Selenium 실행 → 성적 데이터 Session 저장 → JsonResponse
    ↓ redirect (JS)
[GET /view/] 성적 시각화 페이지
```

---

### 문제 4 — 개인정보 보안 설계

**상황**  
학번과 비밀번호를 처리하는 서비스인 만큼 데이터 유출에 대한 고려가 필요했습니다.

**해결**
- 성적 데이터를 **DB에 저장하지 않고** Django Session에 임시 보관
- 세션 데이터는 조회 완료 시 또는 탈퇴 요청 시 즉시 삭제
- 크롤링 과정에서 임시 저장한 HTML 파일은 **학번 MD5 해시**를 파일명으로 사용하고, 파싱 완료 즉시 삭제

```python
filename = hashlib.md5(id.encode()).hexdigest()
# ... 파싱 완료 후
if os.path.isfile(file_path):
    os.remove(file_path)
```

---

## 3. 아키텍처 설계

```bash
┌─────────────────────────────────────────────┐
│                  Client (Browser)           │
│  login.html → loading.html → viewScores.html│
└───────────────────┬─────────────────────────┘
                    │ HTTP / Ajax
┌───────────────────▼─────────────────────────┐
│              Django (config/)               │
│  urls.py → views.py → Session 관리           │
└──────────┬────────────────────┬─────────────┘
           │                    │
┌──────────▼──────────┐  ┌──────▼──────── ────┐
│   module/           │  │  callscore/        │
│   getScore.py       │  │  views.py          │
│   (Selenium + BS4)  │  │  (성적 렌더링)        │
│   processingScore.py│  └────────────────────┘
│   (데이터 가공)        │
└─────────────────────┘
```

**레이어 분리 원칙**
- `module/` — 크롤링 및 데이터 처리 순수 함수
- `config/views.py` — 요청 처리 및 세션 관리
- `callscore/views.py` — 성적 렌더링 뷰 분리
- `templates/` — 프레젠테이션 레이어

---

## 4. 개발 과정

| 단계 | 내용 |
|------|------|
| **Prototype** | Selenium.py — 콘솔 출력용 단일 스크립트, pandas 터미널 출력 |
| **웹 서비스화** | Django 연동, 로그인 페이지 및 세션 기반 인증 흐름 구현 |
| **비동기 처리** | Celery 도입으로 크롤링 중 UI 블로킹 해결 |
| **시각화** | Chart.js 도넛 차트, Bootstrap 5 반응형 레이아웃 |
| **보안 강화** | MD5 파일명 익명화, 임시 파일 자동 삭제, 세션 데이터 즉시 폐기 |

---

## 5. 성과 및 배운 점

### 기술적 성과

- **Selenium 브라우저 자동화**로 일반 HTTP 요청으로는 접근 불가한 JavaScript 렌더링 포털 데이터 수집 성공
- **정규식 기반 동적 파싱**으로 복잡한 컴포넌트 ID 패턴을 처리하여 전 학기(최대 16학기) 성적 완전 자동 파싱
- **Celery 비동기 태스크**로 장기 실행 작업에서 서버 블로킹 없이 사용자 경험 유지
- **Headless Chrome** 옵션 최적화(이미지 비활성화, 쿠키 정책 설정)로 크롤링 속도 개선

### 배운 점

- 동적 웹 페이지의 구조를 분석하고 자동화된 방식으로 데이터를 추출하는 능력
- 웹 서비스에서 동기 처리와 비동기 처리의 차이를 직접 체감하고 적절한 도구 선택
- 개인정보를 다루는 서비스의 책임: 최소한의 데이터만 최단 시간 보관하는 설계 원칙 적용
- 프로토타입을 실제 웹 서비스로 리팩토링하는 과정에서의 레이어 분리 경험

---

## 6. 개선 가능한 부분

| 항목 | 현황 | 개선 방향 |
|------|------|-----------|
| Celery 설정 | celery.py의 `proj` 네임스페이스가 실제 프로젝트명 `config`와 불일치 | 프로젝트명 통일 |
| 예외 처리 | 로그인 실패 시 사용자에게 에러 메시지 미표시 | 로그인 실패 피드백 UI 추가 |
| ChromeDriver 의존성 | 로컬 chromedriver.exe 바이너리 포함 | `webdriver-manager`로 자동 버전 관리 |
| 데이터 영속성 | 페이지 새로고침 시 세션 만료로 데이터 소실 | 선택적 로컬 저장 옵션 제공 |
