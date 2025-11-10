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

# 프로젝트 의존성 설치
cd /home/jemin/lalm_bench
uv sync
```

### 2. 데이터셋 다운로드

```bash
# 모든 데이터셋 다운로드
bash scripts/download_datasets.sh
```

> **주의**: HuggingFace 인증 토큰이 필요합니다.
> `~/.cache/huggingface/hub/`에 저장되거나 환경변수 `HF_TOKEN` 설정 필요.

### 3. 전체 벤치마크 실행

```bash
# 모든 모델, 모든 벤치마크 (Clotho-v2 + MMAU-Pro)
bash scripts/run.sh

# 특정 모델만 실행 (예: Gemma3N)
bash scripts/run.sh gemma3n
```

결과는 `./outputs/{MODEL}/result.txt`에 저장됩니다.

---

## 📊 사용 방법

### 벤치마크별 실행

#### Clotho-v2 벤치마크

```bash
# 전체 파이프라인 (inference + evaluation)
bash scripts/run_clotho.sh gemma3n

# Inference만 (음성에서 텍스트 생성)
bash scripts/run_clotho_inference.sh gemma3n

# Evaluation만 (생성된 결과 평가)
bash scripts/run_clotho_evaluation.sh gemma3n
```

#### MMAU-Pro 벤치마크

```bash
# 전체 파이프라인
bash scripts/run_mmau_pro.sh qwen2_5-omni

# Inference만
bash scripts/run_mmau_pro_inference.sh qwen2_5-omni

# Evaluation만
bash scripts/run_mmau_pro_evaluation.sh qwen2_5-omni
```

### 단계별 실행

```bash
# Inference 단계만 (모든 벤치마크)
bash scripts/run_inference.sh gemma3n

# Evaluation 단계만 (모든 벤치마크)
bash scripts/run_evaluation.sh gemma3n
```

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
├── scripts/                       # 실행 스크립트
│   ├── run.sh                     # 전체 벤치마크
│   ├── run_clotho.sh              # Clotho만
│   ├── run_mmau_pro.sh            # MMAU-Pro만
│   ├── run_inference.sh           # Inference 단계
│   ├── run_evaluation.sh          # Evaluation 단계
│   ├── pipelines/                 # 파이프라인 조합
│   ├── tasks/                     # 개별 작업
│   └── env/                       # 환경 설정
│
├── datasets/                      # 데이터셋 (다운로드 후 저장)
├── outputs/                       # 벤치마크 결과
├── pyproject.toml                 # 프로젝트 설정
├── Dockerfile                     # Docker 이미지
└── docker-compose.yaml            # Docker Compose 설정
```

---

## 🐳 Docker 사용

### Docker 컨테이너 실행

```bash
# 환경 변수 설정 (HuggingFace 토큰)
export HF_TOKEN=your_hf_token_here

# 컨테이너 시작
docker compose up -d

# 컨테이너 내부에서 명령 실행
docker compose exec lalm_bench bash scripts/run_clotho.sh gemma3n

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
| 전체 벤치마크 | `./outputs/{MODEL}/result.txt` |
| Clotho만 | `./outputs/{MODEL}/result_clotho.txt` |
| MMAU-Pro만 | `./outputs/{MODEL}/result_mmau_pro.txt` |
| Inference만 | `./outputs/{MODEL}/result_inference.txt` |
| Evaluation만 | `./outputs/{MODEL}/result_evaluation.txt` |
| 에러 로그 | `./outputs/{MODEL}/*_infer.stderr.log` |
| 에러 로그 | `./outputs/{MODEL}/*_eval.stderr.log` |

---

## 🔧 고급 설정

### 커스텀 환경 설정

**Inference 환경 준비**
```bash
bash scripts/env/setup_inference.sh
```

**Evaluation 환경 준비**
```bash
bash scripts/env/setup_evaluation.sh
```

**기본 환경으로 복원**
```bash
bash scripts/env/restore_env.sh
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
