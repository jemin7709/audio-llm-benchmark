# LALM Bench: Large Audio Language Model Benchmark

오디오 기반 대규모 언어 모델(Large Audio Language Model)의 성능을 벤치마킹하는 프로젝트입니다. Clotho-v2 데이터셋과 MMAU-Pro 데이터셋을 활용하여 여러 모델의 음성 이해 및 설명 생성 능력을 평가합니다.

## 🎯 주요 기능

- **다중 모델 지원**: Qwen2.5-Omni, Qwen3-Omni, Gemma3N
- **벤치마크 데이터셋**: Clotho-v2, MMAU-Pro
- **평가 지표**: CIDEr-D, FENSE
- **자동화된 파이프라인**: Inference 및 Evaluation 단계를 자동으로 처리
- **Docker 지원**: NVIDIA GPU 환경에서 일관성 있는 실행

---

## 📋 필수 요구사항

- **Python**: 3.12 이상
- **GPU**: NVIDIA GPU (docker-compose.yaml에서 4개 GPU 기본 설정)
- **시스템**: Linux/Mac

### 주요 의존성

```
torch>=2.8.0              # PyTorch 프레임워크
torchaudio>=2.8.0         # 오디오 처리
transformers>=4.57.1      # HuggingFace 모델
aac-metrics>=0.6.0        # 평가 메트릭
datasets>=4.0.0           # 데이터셋 로더
```

자세한 의존성은 `pyproject.toml` 참조.

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# uv 설치 (아직 설치 안 했다면)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 루트 환경 동기화
cd /home/jemin/lalm_bench
uv sync

# 전용 가상환경 준비
uv sync --project envs/inference
uv sync --project envs/evaluation
```

### 2. 데이터셋 다운로드

```bash
# 모든 데이터셋 다운로드
bash scripts/download_datasets.sh
```

> **주의**: HuggingFace 인증 토큰이 필요합니다.
> `~/.cache/huggingface/hub/`에 저장되거나 환경변수 `HF_TOKEN` 설정 필요.

### 3. 전체 벤치마크 실행

Typer CLI(엔트리포인트 `lalm`)를 사용합니다. `--model` 옵션을 생략하면 기본 모델 3종(Gemma3N, Qwen2.5-Omni, Qwen3-Omni)을 순차 실행합니다.

```bash
# 모든 모델 대상으로 Clotho + MMAU-Pro 전체 파이프라인 실행
uv run lalm run clotho
uv run lalm run mmau-pro

# 특정 모델만 실행 (예: Gemma3N)
uv run lalm run clotho --model gemma3n
uv run lalm run mmau-pro --model gemma3n
```

각 모델의 결과는 `./outputs/{MODEL}/result_{벤치마크}.txt`에 저장됩니다.

## 🗂️ 서브 프로젝트 안내
- 예측 유사도 비교(`projects/audio_similarity/`), Clotho 참조 유사도 분석(`projects/clotho_ref_similarity/`), 자동 루브릭 생성(`projects/make_rubrics/`) 등 보조 실험 스크립트의 목적·입출력·실행 예시는 `docs/subprojects.md`에서 확인할 수 있습니다.

---

## 📊 사용 방법

### 명령어 개요

| 목적 | 명령 |
|------|------|
| 전체 파이프라인 | `uv run lalm run <benchmark> [--model MODEL]` |
| Inference만 | `uv run lalm inference <benchmark> [--model MODEL]` |
| Evaluation만 | `uv run lalm eval <benchmark> [--model MODEL]` |

- `<benchmark>`: `clotho` 또는 `mmau-pro`
- `--model`을 생략하면 `gemma3n`, `qwen2_5-omni`, `qwen3-omni` 순으로 실행
- `--output-root` 옵션으로 결과 디렉토리를 바꿀 수 있음 (기본값 `./outputs`)

### 예시

```bash
# Gemma3N으로 Clotho 전체 파이프라인 실행
uv run lalm run clotho --model gemma3n

# 모든 기본 모델로 MMAU-Pro inference만 실행
uv run lalm inference mmau-pro

# qwen3-omni 결과를 이용해 Clotho 평가만 수행
uv run lalm eval clotho --model qwen3-omni
```

## 🔍 어텐션 시각화

`visualization.py`를 실행하면 Gemma3N의 레이어별 어텐션을 이미지·NPY·JSON 형태로 저장할 수 있습니다.

```bash
uv run python visualization.py --prompt "Test" --layers 0 1 --limit-samples 1 --output-dir outputs/attn/smoke
```

결과물은 `outputs/attn/{run_name}/{sample_id}/` 및 `outputs/attn/{run_name}/global_*` 위치에 생성됩니다.

### 어텐션 수집
- `--save-attn`, `--attn-layers`, `--attn-run-name` 옵션을 Inference 명령에 붙이면 배치 중 어텐션을 저장합니다.
- 예: `uv run lalm inference clotho --model gemma3n --save-attn`, `uv run lalm run mmau-pro --model qwen3-omni --save-attn --attn-run-name debug`.
- 출력은 `./outputs/{MODEL}/{benchmark}/attn/{run_name}/sample_{idx}` 아래 `attn.npy`, `tokens.json`, `meta.json`으로 확인합니다.

---

## 📁 프로젝트 구조

```
lalm_bench/
├── src/                           # 핵심 코드
│   ├── clotho/                    # Clotho-v2 벤치마크
│   │   ├── inference.py           # 음성 → 설명 생성
│   │   └── evaluation.py          # 생성된 설명 평가
│   ├── mmau-pro/                  # MMAU-Pro 벤치마크
│   │   ├── inference.py
│   │   └── evaluation.py
│   ├── models/                    # 모델 로더
│   │   ├── qwen2_5_omni.py
│   │   ├── qwen3_omni.py
│   │   └── gemma3n.py
│   └── utils/                     # 유틸리티
│       ├── audio_length.py        # 오디오 길이 계산
│       ├── clotho_download.py     # 데이터셋 다운로드
│       └── seed.py                # 난수 시드 설정
│
├── envs/                          # 전용 가상환경 정의
│   ├── inference/pyproject.toml   # 최신 transformers + vLLM 사용
│   └── evaluation/pyproject.toml  # transformers==4.42.4 + aac-metrics
│
├── cli.py                         # Typer 기반 통합 CLI
│
├── scripts/                       # 보조 스크립트
│   ├── download_datasets.sh       # 데이터 다운로드
│   └── install_vllm.sh            # Docker 빌드 시 추가 설치
│
├── projects/                      # 서브 프로젝트 묶음
│   ├── audio_similarity/          # 노이즈 vs 오디오 예측 유사도
│   ├── clotho_ref_similarity/     # Clotho 캡션 유사도/아웃라이어 분석
│   └── make_rubrics/              # Qwen 기반 루브릭 생성/후처리
│
├── datasets/                      # 데이터셋 (다운로드 후 저장)
├── extra/                         # 기타 실험 스크립트와 데이터 드롭존
├── outputs/                       # 벤치마크 결과
├── pyproject.toml                 # 프로젝트 설정
├── Dockerfile                     # Docker 이미지
└── docker-compose.yaml            # Docker Compose 설정
```

> 전체 디렉터리 설명은 `docs/folder_structure.md`에서 더 자세히 확인할 수 있습니다.

---

## 🐳 Docker 사용

### Docker 컨테이너 실행

```bash
# 환경 변수 설정 (HuggingFace 토큰)
export HF_TOKEN=your_hf_token_here

# 컨테이너 시작
docker compose up -d

# 컨테이너 내부에서 명령 실행
docker compose exec lalm_bench uv run lalm run clotho --model gemma3n

# 컨테이너 종료
docker compose down
```

### 주요 설정

- **GPU**: 기본값으로 4개 GPU 할당 (`docker-compose.yaml` 수정으로 변경 가능)
- **PYTHONPATH**: `src` 디렉토리로 자동 설정
- **캐시**: HuggingFace 캐시를 Docker 볼륨에 저장하여 지속성 보장

---

## 📈 출력 파일 위치

| 실행 유형 | 출력 파일 |
|----------|---------|
| Clotho 전체 파이프라인 | `./outputs/{MODEL}/result_clotho.txt` |
| MMAU-Pro 전체 파이프라인 | `./outputs/{MODEL}/result_mmau_pro.txt` |
| Clotho inference만 | `./outputs/{MODEL}/result_clotho_inference.txt` |
| MMAU-Pro inference만 | `./outputs/{MODEL}/result_mmau_pro_inference.txt` |
| Clotho evaluation만 | `./outputs/{MODEL}/result_clotho_evaluation.txt` |
| MMAU-Pro evaluation만 | `./outputs/{MODEL}/result_mmau_pro_evaluation.txt` |
| 에러 로그 (Inference) | `./outputs/{MODEL}/*_infer.stderr.log` |
| 에러 로그 (Evaluation) | `./outputs/{MODEL}/*_eval.stderr.log` |

---

## 🔧 고급 설정

### 커스텀 환경 설정

`envs/` 디렉토리마다 독립적인 `pyproject.toml`을 사용하므로, 필요한 경우 개별적으로 재동기화하면 됩니다.

```bash
# Inference venv 재구성 (최신 transformers + vLLM)
uv sync --project envs/inference --reinstall

# Evaluation venv 재구성 (transformers==4.42.4 + aac-metrics)
uv sync --project envs/evaluation --reinstall
```

### 데이터셋 샘플링

Inference/Evaluation 스크립트에 `--sample_size` 옵션을 추가하여 테스트 실행:

```bash
python src/clotho/inference.py --model gemma3n --sample_size 10
```

---

## 📝 주요 모델 정보

| 모델 | 제공자 | 특징 |
|------|-------|------|
| Qwen2.5-Omni | Alibaba | 음성, 텍스트, 비전 통합 |
| Qwen3-Omni | Alibaba | 최신 버전 |
| Gemma3N | Google | 경량화 모델 |

---

## 🎓 Clotho-v2 & MMAU-Pro 평가 지표

### CIDEr-D (Consensus-based Image Description Evaluation)
- 생성된 설명이 참조 설명과 얼마나 유사한지 측정
- 0~10 범위

### FENSE (Fluency, Extent, Naturalness, Sequence)
- 생성된 텍스트의 유창성과 자연스러움 평가
- 0~1 범위

---

## ⚠️ 주의사항

1. **GPU 메모리**: 일부 모델은 많은 VRAM 필요. 충분한 메모리 확보 필수.
2. **인터넷 연결**: 모델 다운로드 시 안정적인 인터넷 필요.
3. **데이터셋 크기**: Clotho-v2는 ~50GB, MMAU-Pro는 추가 용량 필요.
4. **실행 시간**: 전체 벤치마크는 GPU 성능에 따라 수시간 소요.

---

## 🛠️ 문제 해결

### 1. HuggingFace 인증 오류

```bash
# HuggingFace 로그인
huggingface-cli login

# 또는 환경변수 설정
export HF_TOKEN=your_token_here
```

### 2. GPU 메모리 부족

```bash
# 더 작은 배치 크기로 실행 (스크립트 수정 필요)
# 또는 모델을 float16으로 로드하도록 수정
```

### 3. 데이터셋 다운로드 실패

```bash
# 수동으로 다운로드하여 datasets/ 디렉토리에 저장
# 또는 Hugging Face에서 직접 다운로드
```

---

## 📞 지원

자세한 내용은 `scripts/README.md`를 참조하세요.

---

## 📄 라이선스

프로젝트 라이선스 정보는 LICENSE 파일을 참조하세요.
