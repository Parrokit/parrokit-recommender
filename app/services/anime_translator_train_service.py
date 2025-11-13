# app/services/anime_translator_train_service.py

"""
서비스용 모듈: 일본어 → 한국어 애니 제목 번역 모델(mBART + LoRA) 파인튜닝

- HuggingFace Transformers + PEFT(LoRA) 사용
- mini 데이터셋으로 제목 번역에 특화된 어댑터 학습
- 어댑터 가중치는 app/models/weights/mbart_ja2ko_title_lora_mps/adapter 에 저장
"""

import os
import random
from typing import List, Tuple, Optional, Dict, Any

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from peft import LoraConfig, get_peft_model


# =========================
# 1) 데이터셋 준비 유틸
# =========================

def get_default_title_pairs() -> List[Tuple[str, str]]:
    """하드코딩된 일본어-한국어 애니 제목 페어 리턴."""
    return [
        ("カウボーイビバップ", "카우보이 비밥"),
        ("カウボーイビバップ 天国の扉", "카우보이 비밥 천국의 문"),
        ("トライガン", "트라이건"),
        ("新世紀エヴァンゲリオン", "신세기 에반게리온"),
        ("ナルト", "나루토"),
        ("ONE PIECE", "원피스"),
        ("テニスの王子様", "테니스의 왕자"),
        ("スクールランブル", "스쿨 럼블"),
        ("頭文字〈イニシャル〉D", "이니셜 D"),
        ("頭文字〈イニシャル〉D FOURTH STAGE", "이니셜 D 포스 스테ージ"),
        ("ハングリーハート", "헝그리 하트"),
        ("ハングリーハート Wild Striker", "헝그리 하트 와일드 스트라이커"),
        ("ハチミツとクローバー", "허니와 클로버"),
        ("モンスター", "몬스터"),
        ("冒険王ビィト", "모험왕 비트"),
        ("アイシールド21", "아이실드 21"),
        ("機動戦士ガンダム", "기동전사 건담"),
        ("コードギアス 反逆のルルーシュ", "코드 기아스 반역의 를르슈"),
        ("魔法少女まどか☆マギカ", "마법소녀 마도카☆마기카"),
        ("ジパング", "지팡"),
        ("進撃の巨人", "진격의 거인"),
        ("鬼滅の刃", "귀멸의 칼날"),
        ("SPY×FAMILY", "스파이 패밀리"),
        ("ジョジョの奇妙な冒険", "죠죠의 기묘한 모험"),
        ("銀魂", "은혼"),
        ("鋼の錬金術師", "강철의 연금술사"),
        ("デスノート", "데스노트"),
        ("ソードアート・オンライン", "소드 아트 온라인"),
        ("Re:ゼロから始める異世界生活", "Re:제로부터 시작하는 이세계 생활"),
        ("この素晴らしい世界に祝福を！", "이 멋진 세계에 축복을!"),
        ("ノーゲーム・ノーライフ", "노 게임 노 라이프"),
        ("涼宮ハルヒの憂鬱", "스즈미야 하루히의 우울"),
        ("らき☆すた", "러키☆스타"),
        ("けいおん！", "케이온!"),
        ("シュタインズ・ゲート", "슈타인즈 게이트"),
        ("攻殻機動隊", "공각기동대"),
        ("サイコパス", "사이코패스"),
        ("プラスティック・メモリーズ", "플라스틱 메모리즈"),
        ("ヴァイオレット・エヴァーガーデン", "바이올렛 에버가든"),
        ("四月は君の嘘", "4월은 너의 거짓말"),
        ("化物語", "바케모노가타리"),
        ("とある科学の超電磁砲", "어떤 과학의 초전자포"),
        ("とある魔術の禁書目録", "어떤 마법의 금서목록"),
        ("五等分の花嫁", "5등분의 신부"),
    ]


def prepare_dataset(
    pairs: List[Tuple[str, str]],
    split_ratio: float = 0.8,
) -> Tuple[Dataset, Dataset]:
    """(ja, ko) 페어 리스트를 섞어서 train / valid 로 나눈다."""
    random.seed(42)
    random.shuffle(pairs)
    split = int(len(pairs) * split_ratio)
    train_pairs = pairs[:split]
    valid_pairs = pairs[split:]

    train_ds = Dataset.from_dict({
        "ja": [j for j, _ in train_pairs],
        "ko": [k for _, k in train_pairs],
    })
    valid_ds = Dataset.from_dict({
        "ja": [j for j, _ in valid_pairs],
        "ko": [k for _, k in valid_pairs],
    })
    return train_ds, valid_ds


# =========================
# 2) 모델 구성
# =========================

def select_device() -> str:
    """MPS → CUDA → CPU 순으로 디바이스 선택."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_model(device: str) -> Tuple[torch.nn.Module, Any]:
    """
    mBART base 모델을 로드하고 LoRA 어댑터를 붙여서 반환.
    """
    MODEL_NAME = "facebook/mbart-large-50-many-to-many-mmt"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    base_model.to(device)

    lora_cfg = LoraConfig(
        task_type="SEQ_2_SEQ_LM",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
    )
    model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


def preprocess_function(examples, tokenizer, max_src: int = 64, max_tgt: int = 64):
    """토크나이즈 + 레이블 생성."""
    tokenizer.src_lang = "ja_XX"
    tokenizer.tgt_lang = "ko_KR"

    model_inputs = tokenizer(examples["ja"], max_length=max_src, truncation=True)
    labels = tokenizer(text_target=examples["ko"], max_length=max_tgt, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# =========================
# 3) 학습 루프
# =========================

def train_title_translator(
    pairs: Optional[List[Tuple[str, str]]] = None,
    output_root: str = "app/models/weights",
    subdir: str = "mbart_ja2ko_title_lora_mps",
    num_train_epochs: int = 8,
    learning_rate: float = 2e-4,
) -> Dict[str, Any]:
    """
    일본어 → 한국어 애니 제목 번역기를 LoRA로 파인튜닝하고,
    어댑터를 app/models/weights/... 에 저장한다.

    반환:
      {
        "adapter_dir": <어댑터 저장 경로>,
        "device": "mps" | "cuda" | "cpu",
        "num_train_epochs": ...,
        "learning_rate": ...,
        "train_size": ...,
        "valid_size": ...,
      }
    """
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    device = select_device()
    print(f"[anime_translator_train] device = {device}")

    # 1) 데이터 준비
    if pairs is None:
        pairs = get_default_title_pairs()

    train_ds, valid_ds = prepare_dataset(pairs)
    print(f"[anime_translator_train] train/valid size = {len(train_ds)} / {len(valid_ds)}")

    # 2) 모델 구성
    model, tokenizer = build_model(device)

    # 3) 토크나이징
    max_src, max_tgt = 64, 64
    train_tok = train_ds.map(
        lambda batch: preprocess_function(batch, tokenizer, max_src, max_tgt),
        batched=True,
        remove_columns=train_ds.column_names,
    )
    valid_tok = valid_ds.map(
        lambda batch: preprocess_function(batch, tokenizer, max_src, max_tgt),
        batched=True,
        remove_columns=valid_ds.column_names,
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 4) output 경로 설정 (FastAPI 프로젝트 구조 반영)
    output_dir = os.path.join(output_root, subdir)
    os.makedirs(output_dir, exist_ok=True)

    args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=1,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=2,
        predict_with_generate=True,
        generation_max_length=64,
        report_to=[],  # WandB 같은 거 안 씀
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=valid_tok,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # 5) LoRA 어댑터만 저장
    adapter_dir = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_dir)
    print(f"[anime_translator_train] LoRA adapter saved to: {adapter_dir}")

    return {
        "adapter_dir": adapter_dir,
        "device": device,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "train_size": len(train_ds),
        "valid_size": len(valid_ds),
    }