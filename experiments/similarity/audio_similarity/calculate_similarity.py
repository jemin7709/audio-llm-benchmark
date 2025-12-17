#!/usr/bin/env python3
"""
두 예측 파일의 답변 간 코사인 유사도를 카테고리별, 전체로 계산합니다.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_NOISE_FILE = PROJECT_DIR / "predictions_with_noise.parquet"
DEFAULT_AUDIO_FILE = PROJECT_DIR / "predictions_with_audio.parquet"
DEFAULT_OUTPUT_FILE = PROJECT_DIR / "similarity_results.json"


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """코사인 유사도 계산"""
    if np.all(vec1 == 0) or np.all(vec2 == 0):
        return 0.0
    return 1 - cosine(vec1, vec2)


def calculate_similarities(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    id_col: str = "id",
    text_col: str = "model_output",
    model_name: str = "BAAI/bge-base-en-v1.5",
    batch_size: int = 32,
) -> dict:
    """두 데이터프레임의 텍스트 유사도를 계산"""
    # 임베딩 모델 로드
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name)

    # ID로 매칭
    merged = pd.merge(
        df1[[id_col, text_col, "category"]],
        df2[[id_col, text_col]],
        on=id_col,
        suffixes=("_noise", "_audio"),
    )

    # 텍스트 준비
    texts_noise = [
        str(text) if not pd.isna(text) else "" for text in merged[f"{text_col}_noise"]
    ]
    texts_audio = [
        str(text) if not pd.isna(text) else "" for text in merged[f"{text_col}_audio"]
    ]

    # 임베딩 생성 (BGE 모델은 normalize_embeddings=True 권장)
    print("Generating embeddings...")
    embeddings_noise = model.encode(
        texts_noise,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings_audio = model.encode(
        texts_audio,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # 전체 유사도 계산 (코사인 유사도 = 정규화된 벡터의 내적)
    similarities = (embeddings_noise * embeddings_audio).sum(axis=1)
    # 소수점 4번째 자리까지 (5번째에서 반올림)
    similarities = [round(float(s), 4) for s in similarities]

    merged["similarity"] = similarities

    # 카테고리별 유사도
    category_results = {}
    for category in merged["category"].unique():
        cat_data = merged[merged["category"] == category]
        if len(cat_data) > 0:
            category_results[category] = {
                "mean": round(float(cat_data["similarity"].mean()), 4),
                "std": round(float(cat_data["similarity"].std()), 4),
                "count": int(len(cat_data)),
            }

    # 전체 평균
    overall_mean = round(float(merged["similarity"].mean()), 4)
    overall_std = round(float(merged["similarity"].std()), 4)

    return {
        "overall": {
            "mean": overall_mean,
            "std": overall_std,
            "count": int(len(merged)),
        },
        "by_category": category_results,
        "detailed": merged[[id_col, "category", "similarity"]].to_dict("records"),
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="두 예측 파일의 코사인 유사도 계산")
    parser.add_argument(
        "--noise-file",
        default=str(DEFAULT_NOISE_FILE),
        help="노이즈 예측 파일 경로",
    )
    parser.add_argument(
        "--audio-file",
        default=str(DEFAULT_AUDIO_FILE),
        help="오디오 예측 파일 경로",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_FILE),
        help="결과 저장 경로 (JSON)",
    )
    parser.add_argument(
        "--model",
        default="BAAI/bge-base-en-v1.5",
        help="임베딩 모델 이름 (기본값: BAAI/bge-base-en-v1.5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="배치 크기 (기본값: 32)",
    )

    args = parser.parse_args()

    print(f"Loading {args.noise_file}...")
    df_noise = pd.read_parquet(args.noise_file)
    print(f"Loaded {len(df_noise)} samples")

    print(f"Loading {args.audio_file}...")
    df_audio = pd.read_parquet(args.audio_file)
    print(f"Loaded {len(df_audio)} samples")

    print("Calculating similarities...")
    results = calculate_similarities(
        df_noise, df_audio, model_name=args.model, batch_size=args.batch_size
    )

    # 결과 출력
    print("\n" + "=" * 60)
    print("코사인 유사도 결과")
    print("=" * 60)
    print(
        f"\n전체 평균: {results['overall']['mean']:.4f} (±{results['overall']['std']:.4f})"
    )
    print(f"전체 샘플 수: {results['overall']['count']}")

    print("\n카테고리별 결과:")
    print("-" * 60)
    for category, stats in sorted(results["by_category"].items()):
        print(
            f"{category:30s} | 평균: {stats['mean']:.4f} (±{stats['std']:.4f}) | 샘플: {stats['count']}"
        )

    # JSON 저장
    import json

    # numpy 타입을 Python 기본 타입으로 변환
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj

    results_serializable = convert_numpy(results)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results_serializable, f, ensure_ascii=False, indent=2)

    print(f"\n결과가 {args.output}에 저장되었습니다.")


if __name__ == "__main__":
    main()
