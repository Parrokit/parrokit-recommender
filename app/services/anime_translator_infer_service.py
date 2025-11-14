# app/services/anime_translator_infer_service.py

import os
import re
from typing import Tuple, List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel

from app import dependencies as deps  # 전역 translator_model, anime_meta_df 사용


def _select_device() -> str:
    """MPS → CUDA → CPU 순서로 선택."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_translator_model(
    adapter_dir: str | None = None,
    base_model_name: str = "facebook/mbart-large-50-many-to-many-mmt",
) -> Tuple[torch.nn.Module, AutoTokenizer]:
    """
    mBART base + LoRA adapter 로드해서 (model, tokenizer) 반환.

    adapter_dir 기본값:
      - env MBART_ADAPTER_DIR
      - 없으면 "app/models/weights/mbart_ja2ko_title_lora_mps/adapter"
    """
    adapter_dir = adapter_dir or os.getenv(
        "MBART_ADAPTER_DIR",
        "app/models/weights/mbart_ja2ko_title_lora_mps/adapter",
    )

    device = _select_device()

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=False)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    base_model.to(device)

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.to(device)
    return model, tokenizer


def translate_jp_to_ko(text: str) -> str:
    """
    dependencies에 미리 로드된 translator_model / translator_tokenizer를 사용해서
    일본어 텍스트를 한국어로 번역.
    """
    if deps.translator_model is None or deps.translator_tokenizer is None:
        raise RuntimeError(
            "translator_model / tokenizer가 초기화되지 않았습니다. init_models()를 확인하세요.")

    model = deps.translator_model
    tokenizer = deps.translator_tokenizer

    device = next(model.parameters()).device
    tokenizer.src_lang = "ja_XX"

    if text.strip() == "":
        print("[번역 스킵] 빈 문자열 감지됨.")
        return ""

    print(f"[번역 시작] 원문(JP): {text[:50]}{'...' if len(text) > 50 else ''}")

    enc = tokenizer(text, return_tensors="pt").to(device)
    model.eval()

    with torch.inference_mode():
        outputs = model.generate(
            **enc,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("ko_KR"),
            num_beams=5,
            length_penalty=1.1,
            max_new_tokens=128,
            early_stopping=True,
        )

    out = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    out = re.sub(r"^(한국어|번역).*?:\s*", "", out).strip()

    print(f"[번역 완료] 결과(KO): {out[:50]}{'...' if len(out) > 50 else ''}")

    return out

def translate_anime_metadata_by_ids(anime_ids: List[int]) -> List[Dict[str, Any]]:
    """
    anime_id 기준으로 Name/Synopsis를 번역 후 JSON 반환.
    Score / Genres / Type / Favorites / Image URL 포함
    """
    if deps.anime_meta_df is None:
        raise RuntimeError("anime_meta_df가 초기화되지 않았습니다. init_models()를 확인하세요.")

    df = deps.anime_meta_df

    print(f"[메타데이터 요청] anime_ids={anime_ids}")

    subset = df[df["anime_id"].isin(anime_ids)]
    print(f"[메타데이터 필터링] 요청 {len(anime_ids)}개 → 존재 {len(subset)}개")

    results: List[Dict[str, Any]] = []

    for idx, (_, row) in enumerate(subset.iterrows(), start=1):
        aid = int(row["anime_id"])
        name_jp = str(row.get("Name", "") or "")
        syn_jp = str(row.get("Synopsis", "") or "")

        # 기존 번역 필드 확인 (이미 번역이 존재하면 번역 스킵)
        existing_name_ko = row.get("Korean Name", "")
        existing_syn_ko = row.get("Korean Synopsis", "")

        if isinstance(existing_name_ko, str) and existing_name_ko.strip():
            name_ko = existing_name_ko.strip()
        else:
            name_ko = translate_jp_to_ko(name_jp) if name_jp else ""

        if isinstance(existing_syn_ko, str) and existing_syn_ko.strip():
            syn_ko = existing_syn_ko.strip()
        else:
            syn_ko = translate_jp_to_ko(syn_jp) if syn_jp else ""

        print(f"\n--- [{idx}/{len(subset)}] anime_id={aid} ---")
        print(f"[JP Name] {name_jp[:50]}{'...' if len(name_jp) > 50 else ''}")
        print(f"[JP Synopsis] {syn_jp[:50]}{'...' if len(syn_jp) > 50 else ''}")

        print(f"[결과] name_ko={name_ko[:50]}{'...' if len(name_ko) > 50 else ''}")
        print(f"[결과] synopsis_ko={syn_ko[:50]}{'...' if len(syn_ko) > 50 else ''}")

        score = row.get("Score", None)
        genres = row.get("Genres", None)
        type_ = row.get("Type", None)
        favorites = row.get("Favorites", None)
        image_url = row.get("Image URL", None)

        results.append(
            {
                "anime_id": aid,

                # 기존 필드
                "name_jp": name_jp,
                "name_ko": name_ko,
                "synopsis_jp": syn_jp,
                "synopsis_ko": syn_ko,

                "score": score,
                "genres": genres,
                "type": type_,
                "favorites": favorites,
                "image_url": image_url,
            }
        )

    print("\n[메타데이터 번역 완료] 총 처리 개수:", len(results))

    return results