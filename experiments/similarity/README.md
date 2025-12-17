# 유사도 분석

**목표**: 예측값 간 유사도, Clotho 참조 캡션 유사도 등을 분석합니다.

## 하위 프로젝트

### `audio_similarity/`
노이즈가 포함된 오디오와 원본 오디오 예측값의 유사도를 비교합니다.

```bash
uv run python experiments/similarity/audio_similarity/calculate_similarity.py [--noise-file PATH] [--audio-file PATH] [--output PATH]
```

**입력**: 예측 Parquet 파일 (기본값: `data/artifacts/similarity/audio_similarity/`)
**출력**: `data/artifacts/similarity/audio_similarity/` 아래 `similarity_results.json`, `predictions_with_audio.parquet`, `predictions_with_noise.parquet`

---

### `clotho_ref_similarity/`
Clotho-v2 데이터셋의 참조 캡션 유사도 및 아웃라이어를 분석합니다.

```bash
uv run python experiments/similarity/clotho_ref_similarity/similarity.py --split development [--output-dir PATH]
```

**입력**: Clotho 데이터셋 분할 파일 (`--split`: development, validation, evaluation)
**출력**: `data/artifacts/similarity/clotho_ref_similarity/` 아래 CSV/JSON 형식의 유사도 통계 및 아웃라이어 목록

---

각 하위 프로젝트의 상세 사용법은 해당 디렉토리 내 스크립트의 도움말을 참조하세요.

