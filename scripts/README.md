# Scripts Directory Structure

벤치마크 실행 스크립트의 계층적 구조

## 디렉토리 구조

```
scripts/
├── run.sh                        # 전체 워크플로우 (모든 벤치마크)
├── run_inference.sh              # 모든 벤치마크 inference만
├── run_evaluation.sh             # 모든 벤치마크 evaluation만
├── run_mmau_pro.sh               # MMAU-Pro만 (inference + evaluation)
├── run_clotho.sh                 # Clotho만 (inference + evaluation)
├── run_mmau_pro_inference.sh     # MMAU-Pro inference만
├── run_clotho_inference.sh       # Clotho inference만
├── run_mmau_pro_evaluation.sh    # MMAU-Pro evaluation만
├── run_clotho_evaluation.sh      # Clotho evaluation만
├── env/                          # 최하위 계층: 환경 설정 (uv add/remove)
│   ├── setup_inference.sh        # Inference 환경 설정
│   ├── setup_evaluation.sh       # Evaluation 환경 설정
│   └── restore_env.sh            # 기본 환경 복원
├── tasks/                        # 중간 계층: 개별 작업
│   ├── mmau_pro_inference.sh
│   ├── clotho_inference.sh
│   ├── mmau_pro_evaluation.sh
│   └── clotho_evaluation.sh
└── pipelines/                    # 상위 계층: 워크플로우 조합
    ├── inference_pipeline.sh     # 모든 벤치마크 inference
    ├── evaluation_pipeline.sh    # 모든 벤치마크 evaluation
    ├── mmau_pro_pipeline.sh      # MMAU-Pro 전체
    ├── clotho_pipeline.sh        # Clotho 전체
    ├── mmau_pro_inference_only.sh
    ├── clotho_inference_only.sh
    ├── mmau_pro_evaluation_only.sh
    └── clotho_evaluation_only.sh
```

## 사용법

### 1. 전체 워크플로우 (모든 벤치마크)

```bash
# 모든 모델, 모든 벤치마크 실행
./run.sh

# 특정 모델만 실행
./run.sh gemma3n
```

결과: `./outputs/{MODEL}/result.txt`

### 2. 벤치마크별 실행

#### 2-1. MMAU-Pro만 실행

```bash
# MMAU-Pro 전체 (inference + evaluation)
./run_mmau_pro.sh gemma3n
# 결과: ./outputs/gemma3n/result_mmau_pro.txt

# MMAU-Pro inference만
./run_mmau_pro_inference.sh gemma3n
# 결과: ./outputs/gemma3n/result_mmau_pro_inference.txt

# MMAU-Pro evaluation만
./run_mmau_pro_evaluation.sh gemma3n
# 결과: ./outputs/gemma3n/result_mmau_pro_evaluation.txt
```

#### 2-2. Clotho만 실행

```bash
# Clotho 전체 (inference + evaluation)
./run_clotho.sh gemma3n
# 결과: ./outputs/gemma3n/result_clotho.txt

# Clotho inference만
./run_clotho_inference.sh gemma3n
# 결과: ./outputs/gemma3n/result_clotho_inference.txt

# Clotho evaluation만
./run_clotho_evaluation.sh gemma3n
# 결과: ./outputs/gemma3n/result_clotho_evaluation.txt
```

### 3. 단계별 실행 (모든 벤치마크)

```bash
# Inference만 (MMAU-Pro + Clotho)
./run_inference.sh gemma3n
# 결과: ./outputs/gemma3n/result_inference.txt

# Evaluation만 (MMAU-Pro + Clotho, inference 결과 필요)
./run_evaluation.sh gemma3n
# 결과: ./outputs/gemma3n/result_evaluation.txt
```

## 계층 구조 설명

### 최하위 계층: Environment Setup (`env/`)
- 각 단계에 필요한 라이브러리를 `uv add/remove`로 설치/제거
- `setup_inference.sh`: transformers(git), vllm 설치
- `setup_evaluation.sh`: transformers==4.42.4 설치
- `restore_env.sh`: 기본 환경으로 복원

### 중간 계층: Task Execution (`tasks/`)
- 개별 작업(inference, evaluation) 실행
- 각 작업의 결과를 로그 파일에 기록

### 상위 계층: Pipeline (`pipelines/`)
- 환경 설정 + 작업 실행을 조합
- `inference_pipeline.sh`: inference 환경 설정 → inference 작업 실행
- `evaluation_pipeline.sh`: evaluation 환경 설정 → evaluation 작업 실행

### 최상위 계층: Orchestration (`run*.sh`)
- 전체 워크플로우 오케스트레이션
- 디렉토리 준비, 결과 파일 초기화, 환경 복원 등

## 출력 파일

- `./outputs/{MODEL}/result.txt` - 전체 워크플로우 결과
- `./outputs/{MODEL}/result_inference.txt` - Inference만 실행한 경우
- `./outputs/{MODEL}/result_evaluation.txt` - Evaluation만 실행한 경우
- `./outputs/{MODEL}/*_infer.stderr.log` - Inference 에러 로그
- `./outputs/{MODEL}/*_eval.stderr.log` - Evaluation 에러 로그

