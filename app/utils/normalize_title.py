import re

# ---------- 0) 노이즈에 강한 정규화 ----------
# - 한글/히라가나/가타카나/한자/영문/숫자만 남김
# - 괄호/부제/기호/이상문자 제거 후 공백 정리

def normalize_title(s: str) -> str:
    NOISE_KEEP = re.compile(r"[^0-9A-Za-z\uAC00-\uD7A3\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\s]+")

    s = s.lower()
    s = re.sub(r"\(.*?\)", " ", s)              # 괄호 내 부제 제거
    s = NOISE_KEEP.sub(" ", s)                   # 허용 외 문자 제거
    s = re.sub(r"\s+", " ", s).strip()         # 공백 정리
    return s
