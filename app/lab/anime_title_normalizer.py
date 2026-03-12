import pandas as pd

import json
from openai import OpenAI
from dotenv import load_dotenv
import os

# Helper to recover fields from loose JSON-like string
def _extract_field_from_loose_json(content: str, key: str) -> str:
    """Best-effort extraction of a string field from almost-JSON content.

    Looks for a pattern like "key": "value" and returns value, even if the
    closing quote/brace is missing. Used as a fallback when json.loads fails.
    """
    marker = f'"{key}"'
    idx = content.find(marker)
    if idx == -1:
        return ""

    colon = content.find(":", idx)
    if colon == -1:
        return ""

    first_quote = content.find('"', colon + 1)
    if first_quote == -1:
        return ""

    second_quote = content.find('"', first_quote + 1)
    if second_quote == -1:
        # No closing quote; use end of line or end of string
        line_end = content.find("\n", first_quote + 1)
        if line_end == -1:
            line_end = len(content)
        return content[first_quote + 1 : line_end].strip()

    return content[first_quote + 1 : second_quote].strip()

# 1) 데이터 로드
df = pd.read_csv("data/animelist-dataset/new-anime-dataset-2023-with-korean.csv")

# 2) Korean Name / Korean Synopsis 컬럼 없으면 추가 (NaN으로 초기화)
for col in ["Korean Name", "Korean Synopsis"]:
    if col not in df.columns:
        df[col] = pd.NA

# 3) Score가 UNKNOWN이 아닌 행만, 점수 내림차순으로 정렬
df_known = df[df['Score'] != "UNKNOWN"].copy()
df_sorted_score = df_known.sort_values('Score', ascending=False)

# 4) 원본 df에서 사용할 인덱스 리스트 (정렬된 순서대로)
sorted_indices = list(df_sorted_score.index)

len(sorted_indices), df.shape


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = "gpt-4.1-mini"           # 필요하면 다른 모델명으로 변경
client = OpenAI(api_key=OPENAI_API_KEY, timeout=20.0)

def call_translation_api(
    main_name: str,
    other_name: str,
    english_name: str,
    synopsis: str,
) -> dict:
    main_name = main_name or ""
    other_name = other_name or ""
    english_name = english_name or ""
    synopsis = synopsis or ""

    system_prompt = (
        "당신은 일본 애니메이션 정보를 한국어로 현지화하는 도우미입니다. "
        "입력으로 애니의 Main Name, Other Name, English Name과 영어 Synopsis가 주어집니다. "
        "한국에서 통용되는 자연스러운 애니 제목(Korean Name)과 "
        "한국어로 자연스럽게 번역된 시놉시스(Korean Synopsis)를 생성하세요. "
        "반드시 JSON 형식으로만 응답하세요. 키 이름은 'korean_name', 'korean_synopsis' 입니다."
    )

    user_prompt = f"""
다음 애니메이션 정보를 한국어로 정리해 주세요.

[Main Name]
{main_name}

[Other Name]
{other_name}

[English Name]
{english_name}

[Synopsis (EN)]
{synopsis}

출력 형식 (JSON 예시):

{{
  "korean_name": "강철의 연금술사",
  "korean_synopsis": "엘릭 형제가 연금술을 사용해..."
}}
"""

    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            timeout=20.0,
        )
        content = resp.choices[0].message.content.strip()

        # ```json ... ``` 형태로 오는 경우 처리
        if content.startswith("```"):
            content = content.strip("`")
            lines = content.splitlines()
            if lines and lines[0].strip().lower().startswith("json"):
                content = "\n".join(lines[1:])

        try:
            data = json.loads(content, strict=False)
        except json.JSONDecodeError as e:
            print("[API ERROR] JSON decode error:", e)
            print("[API ERROR] Raw content snippet:", content[:200])

            # Try best-effort recovery from almost-JSON
            recovered_name = _extract_field_from_loose_json(content, "korean_name")
            recovered_synopsis = _extract_field_from_loose_json(content, "korean_synopsis")

            if recovered_name or recovered_synopsis:
                print("[API ERROR] Recovered from loose JSON ->",
                      "name=", recovered_name,
                      "synopsis(prefix)=", recovered_synopsis[:60])
                return {
                    "korean_name": recovered_name,
                    "korean_synopsis": recovered_synopsis,
                }

            # If recovery fails, fall back to empty values
            return {
                "korean_name": "",
                "korean_synopsis": "",
            }

        korean_name = data.get("korean_name", "").strip()
        korean_synopsis = data.get("korean_synopsis", "").strip()

        return {
            "korean_name": korean_name,
            "korean_synopsis": korean_synopsis,
        }

    except Exception as e:
        print("[API ERROR]", e)
        return {
            "korean_name": "",
            "korean_synopsis": "",
        }
    

from tqdm import tqdm
import pandas as pd
import json

def is_incomplete_like_annotator(kname, ksyn) -> bool:
    """KoreanAnnotator.find_next_incomplete_index()와 동일한 기준."""
    kname_empty = (isinstance(kname, str) and kname.strip() == "")
    ksyn_empty = (isinstance(ksyn, str) and ksyn.strip() == "")
    
    if pd.isna(kname) or pd.isna(ksyn) or kname_empty or ksyn_empty:
        return True
    return False

def auto_fill_korean_from_annotator_logic(
    df: pd.DataFrame,
    sorted_indices,
    save_path: str = "data/animelist-dataset/new-anime-dataset-2023-with-korean.csv",
    save_every: int = 20,
    show_synopsis: bool = False,
):
    total = len(sorted_indices)
    updated_count = 0

    print("=== 자동 번역 (KoreanAnnotator 로직 그대로) 시작 ===")
    print(f"총 대상 행: {total}\n")

    for i, idx in enumerate(tqdm(sorted_indices, desc="Translating anime", unit="anime")):
        row = df.loc[idx]

        # 기존 값
        kname_old = row.get("Korean Name")
        ksyn_old  = row.get("Korean Synopsis")

        # ✅ KoreanAnnotator와 동일한 '미완료' 판정
        if not is_incomplete_like_annotator(kname_old, ksyn_old):
            # 둘 다 채워져 있으면 그대로 둠
            continue

        # 원본 정보
        main_name = row.get("Name") or ""
        other_name = row.get("Other name") or ""
        english_name = (
            row.get("English name")
            if "English name" in df.columns
            else row.get("English Name")
        ) or ""
        syn = row.get("Synopsis") or ""

        # 🔍 번역 API 호출 (suggestion 생성)
        auto = call_translation_api(
            str(main_name or ""),
            str(other_name or ""),
            str(english_name or ""),
            str(syn or ""),
        )
        suggested_kname = auto.get("korean_name", "") or ""
        suggested_ksyn  = auto.get("korean_synopsis", "") or ""

        # ✅ KoreanAnnotator와 동일한 우선순위:
        #    - 기존 값이 문자열이고 비어있지 않으면 그걸 유지
        #    - 아니면 suggestion 사용
        final_kname = (
            kname_old
            if isinstance(kname_old, str) and kname_old.strip() != ""
            else suggested_kname
        )
        final_ksyn = (
            ksyn_old
            if isinstance(ksyn_old, str) and ksyn_old.strip() != ""
            else suggested_ksyn
        )

        # 🔎 로그로 다 보여주기
        print("\n=======================================")
        print(f"[{updated_count+1}] index={idx} | Score={row.get('Score')}")
        print("=== 원본 정보 ===")
        print(f"- Main Name     : {main_name}")
        print(f"- Other Name    : {other_name}")
        print(f"- English Name  : {english_name}")
        if show_synopsis:
            print(f"- Synopsis (EN) : {syn[:250]}{'...' if len(syn) > 250 else ''}")

        print("\n=== 기존 Korean 값 ===")
        print(f"- 기존 Korean Name : {kname_old}")
        print(f"- 기존 Korean Syn  : {ksyn_old[:150] if isinstance(ksyn_old, str) else ksyn_old}")

        print("\n=== 제안 번역(Suggestion) ===")
        print(f"- 제안 Korean Name : {suggested_kname}")
        print(f"- 제안 Korean Syn  : {suggested_ksyn[:150]}{'...' if len(suggested_ksyn) > 150 else ''}")

        print("\n=== 최종 적용 값(Final) ===")
        print(f"➡️ 최종 Korean Name : {final_kname}")
        print(f"➡️ 최종 Korean Syn  : {final_ksyn[:150] if isinstance(final_ksyn, str) else final_ksyn}")
        print("=======================================")

        # df에 최종 값 저장
        df.at[idx, "Korean Name"] = final_kname
        df.at[idx, "Korean Synopsis"] = final_ksyn
        updated_count += 1

        # 중간 저장
        if updated_count % save_every == 0:
            df.to_csv(save_path, index=False)
            print(f"💾 중간 저장 완료 -> {save_path}")

    df.to_csv(save_path, index=False)
    print("\n=== 자동 번역 완료 ===")
    print(f"총 업데이트된 행: {updated_count}")
    print(f"파일 저장 위치: {save_path}")

    return df

save_path = "data/animelist-dataset/new-anime-dataset-2023-with-korean.csv"

df = auto_fill_korean_from_annotator_logic(
    df=df,
    sorted_indices=sorted_indices,
    save_path=save_path,
    save_every=10,
    show_synopsis=False,  # True로 하면 EN 시놉도 같이 찍힘
)