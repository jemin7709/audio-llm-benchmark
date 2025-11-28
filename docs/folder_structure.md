# 폴더 구조 안내

## 1. 루트 디렉터리 개요

| 경로 | 설명 |
| --- | --- |
| `cli.py` | Typer 기반 메인 CLI (`uv run lalm ...`) |
| `src/` | Clotho·MMAU-Pro 파이프라인, 모델 래퍼, 유틸 등 핵심 패키지 |
| `projects/` | 개별 서브 프로젝트 묶음(`audio_similarity`, `clotho_ref_similarity`, `make_rubrics`) |
| `extra/` | 기타 실험 스크립트/데이터(`clair_a.py`, `clotho.json` 등) |
| `docs/` | 프로젝트 문서 (`subprojects.md`, 본 문서 등) |
| `datasets/` | 다운로드된 데이터셋 캐시/샘플 |
| `envs/` | `uv` 전용 가상환경 정의(`inference`, `evaluation`) |
| `scripts/` | 데이터 다운로드, vLLM 설치 등 보조 셸 스크립트 |
| `temp/` | 임시 산출물/중간 결과 (예: `temp/outputs`) |
| `outputs/` *(실행 시 생성)* | 모델별 벤치마크 결과, 어텐션 산출물 |
| `run.sh`, `visualization.py` | 자주 쓰는 배치 스크립트·시각화 도구 |
| `Dockerfile`, `docker-compose.yaml` | GPU 환경용 컨테이너 구성 |
| `pyproject.toml`, `uv.lock`, `envs/**/pyproject.toml` | 루트/서브 프로젝트 의존성 정의 |

## 2. `src/` 세부 구조

```
src/
├── clotho/            # Clotho 벤치마크 (inference/evaluation)
├── mmau-pro/          # MMAU-Pro 벤치마크 (inference/evaluation)
├── models/            # Gemma3N, Qwen2.5/3 Omni 래퍼
├── utils/             # 어텐션 I/O·시드·데이터 다운로드 유틸
├── fense/             # FENSE 점수 계산 및 모델 래퍼
├── clotho/__init__.py … (패키지 초기화)
└── lalm_bench.egg-info/ (패키징 메타)
```

- `src/clotho/`: `inference.py`, `evaluation.py` 두 단계로 구성. `BenchmarkConfig`에서 사용.
- `src/mmau-pro/`: 동일한 인터페이스로 멀티모달 질의/평가 로직을 제공.
- `src/models/`: `Gemma3N`, `Qwen2_5Omni`, `Qwen3Omni` 클래스와 공용 헬퍼.
- `src/utils/`: `attention_plot.py`, `attention_io.py`, `clotho_download.py`, `mmau-pro_download.py`, `seed.py` 등 공용 도구.
- `src/fense/`: FENSE 평가를 위한 `Evaluator`, 데이터 로더, 모델 다운로드 유틸.

## 3. `projects/` 서브 프로젝트 묶음

```
projects/
├── audio_similarity/
│   ├── calculate_similarity.py
│   ├── predictions_with_noise.parquet / predictions_with_audio.parquet
│   └── similarity_results.json (생성 결과)
├── clotho_ref_similarity/
│   ├── similarity.py
│   ├── clotho_pairs_*.csv / clotho_outliers_*.csv / clotho_summary_*.json
├── make_rubrics/
│   ├── run.py / rm_dup.py
│   ├── deter/ , stoch/      # 반복 실험 결과 JSON
```

세부 사용법은 `docs/subprojects.md` 참고.

## 4. `docs/` 및 기타 보조 폴더

- `docs/`
  - `subprojects.md`: calculate similarity, Clotho ref similarity, make rubrics 설명
  - `folder_structure.md`(본 문서): 루트 및 핵심 폴더 구조 개요
- `envs/`
  - `inference/pyproject.toml`: vLLM + 최신 transformers 환경
  - `evaluation/pyproject.toml`: 평가용 고정 버전 세트 (aac-metrics 포함)
- `scripts/`
  - `download_datasets.sh`: Clotho/MMAU-Pro 데이터 일괄 다운로드
  - `install_vllm.sh`: Docker 이미지 빌드시 vLLM 설치
- `extra/`
  - `clair_a.py`, `judge.py`, `clotho.json` 등 실험/데이터 드롭존
  - `predictions_comprehensive_results.json`, `run.sh` 등 레거시 산출물

필요한 폴더만 체크하면 되도록 상위 2~3단계 구조만 정리했습니다. 자세한 파일 설명은 각 모듈의 docstring과 README 섹션을 참고하세요.

