# 추천 시스템 (parrokit_recommender)

## 프로젝트 개요

- Framework: FastAPI  
- 서버 실행: Uvicorn  
- 언어: Python 3.9  



## 설치 및 실행 방법

### conda 생성 및 실행
> 생성
```bash
conda create -n parrokit-recommender python=3.9
```
>  실행
```bash
conda activate parrokit-recommender
```

### 의존성 설치

```bash
pip install -r requirements.txt
```

### 서버 실행
> 기본
```bash
uvicorn app.main:app --reload
```
> 귀찮으면 makefile
```bash
make dev
```