# 서브 프로젝트 정리

`lalm_bench`에는 본 벤치마크 파이프라인 외에도 분석·평가를 보조하는 개별 스크립트/실험 폴더가 포함되어 있습니다. 아래는 자주 쓰이는 세부 작업들의 목적과 입·출력 요약입니다.

## 1. 예측 유사도 비교 (`projects/audio_similarity/`)
- **위치**: `projects/audio_similarity/` (`calculate_similarity.py` + Parquet/JSON 샘플)
- **기능**: `predictions_with_noise.parquet`와 `predictions_with_audio.parquet`에서 동일 ID의 캡션을 매칭한 뒤, SentenceTransformer 임베딩(BGE 기본)으로 코사인 유사도를 계산합니다.
- **입력**: 두 개의 Parquet 예측 파일(`--noise-file`, `--audio-file`), 텍스트 칼럼은 `model_output`, 카테고리는 `category` (기본값은 폴더 내 샘플 데이터).
- **출력**:
  - 전체/카테고리별 평균·표준편차·샘플 수가 담긴 JSON (`--output`, 기본 `similarity_results.json`)
  - 세부 결과는 JSON `detailed` 필드에 ID별 similarity 기록으로 포함.
- **의존성/환경**: `sentence-transformers`, `scipy`. GPU 선택 옵션은 없으며 CPU에서도 동작.
- **실행 예시**:
  ```bash
  uv run python projects/audio_similarity/calculate_similarity.py \
    --model BAAI/bge-base-en-v1.5 \
    --batch-size 64
  ```

## 2. Clotho 레퍼런스 유사도 (`projects/clotho_ref_similarity/similarity.py`)
- **위치**: `projects/clotho_ref_similarity/`
- **핵심 스크립트**: `similarity.py`
- **기능**: Clotho v2.1 각 오디오에 대해 5개의 reference 캡션 조합(10쌍)의 코사인 유사도를 계산하고 통계/이상치를 추출합니다.
- **입력**:
  - HuggingFace 캐시에 있는 CSV(`~/.cache/huggingface/hub/datasets--woongvy--clotho-v2.1/snapshots/…/clotho_captions_<split>.csv`)
  - 옵션: `--split [development|validation|evaluation]`, `--model`(기본 `BAAI/bge-m3`), `--device`, `--batch-size`.
- **출력**:
  - 쌍별 유사도 CSV `clotho_pairs_<split>.csv`
  - 이상치(Z-score ≥ 2) CSV `clotho_outliers_<split>.csv`
  - 개요 지표 JSON `clotho_summary_<split>.json`
- **주의 사항**:
  - SentenceTransformer 모델이 GPU를 사용할 수 있도록 `--device cuda` 또는 기본 `auto` 사용.
  - 데이터셋 미다운로드 시 오류가 발생하므로 `scripts/download_datasets.sh`로 선행 준비.
- **실행 예시**:
  ```bash
  uv run python projects/clotho_ref_similarity/similarity.py \
    --split evaluation \
    --model BAAI/bge-m3 \
    --device cuda \
    --batch-size 128
  ```

## 3. 자동 루브릭 생성 (`projects/make_rubrics/`)
- **위치**: `projects/make_rubrics/`
- **구성**:
  - `run.py`: Clotho 캡션(참조)과 Qwen 계열 모델 응답을 비교하여 새로운 평가 루브릭을 시뮬레이션으로 생성하고 중복 제거까지 수행합니다.
  - `rm_dup.py`: `run.py` 결과 JSON 리스트에서 마지막 항목만 추출해 별도 폴더로 모으는 후처리 스크립트.
  - `deter/`, `stoch/`: 반복 실험 결과(JSON) 보관소.
- **동작 개요 (`run.py`)**:
  1. `HF_HOME` 아래 캐시된 Clotho CSV를 로드하고, 각 샘플에 대해 Qwen3Omni 모델로 baseline 응답을 생성.
  2. 참조 캡션 중 하나와 baseline 응답을 비교하여 LLM에게 새로운 평가 기준을 추출.
  3. 추출된 기준을 다시 LLM으로 중복 제거해 가중치가 포함된 루브릭 세트를 업데이트.
  4. 진행 로그는 `projects/make_rubrics/results_<index>.json`에 순차적으로 누적.
- **필수 환경**:
  - `HF_HOME` 환경변수 설정(예: `export HF_HOME=~/.cache/huggingface`).
  - Qwen 계열 모델 키/토큰이 `src/models` 래퍼에서 요구하는 방식으로 준비돼야 함.
- **실행 예시**:
  ```bash
  export HF_HOME=~/.cache/huggingface
  uv run python projects/make_rubrics/run.py
  ```
  대량의 JSON이 생성되므로 필요 시 `rm_dup.py`로 마지막 상태만 추려 새 디렉터리(`stoch_new/` 등)에 정리합니다.
- **후처리 예시**:
  ```bash
  uv run python projects/make_rubrics/rm_dup.py \
    --src-dir projects/make_rubrics/stoch \
    --dst-dir projects/make_rubrics/stoch_dedup \
    --overwrite
  ```

세 스크립트는 상호 독립적으로 동작하므로 필요 작업에 맞춰 해당 폴더에서 바로 `uv run python ...` 형태로 실행하면 됩니다.

