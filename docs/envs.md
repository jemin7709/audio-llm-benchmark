# 실행 환경(envs) 설명

## 개요: 왜 2개 환경으로 분리하는가?

`lalm_bench`는 **벤치마크 inference**(모델 추론)와 **벤치마크 evaluation**(결과 평가)을 2개의 독립적인 Python 환경으로 분리합니다.

주요 이유는 **버전 충돌**입니다:
- **`envs/inference`**: 최신 `transformers` (>=4.51.1), 최신 `tokenizers` (>=0.21.1), 그리고 고속 추론용 `vllm`(==0.10.2)을 사용합니다.
- **`envs/evaluation`**: 기존 `aac-metrics`, `sentence-transformers` 등과의 호환성을 위해 **구버전 `transformers==4.42.4`와 `tokenizers>=0.19.0,<0.20.0`**을 고정해야 합니다.

단일 환경에서 두 버전을 동시에 만족할 수 없으므로, 역할별로 환경을 나눕니다.

---

## 버전 매트릭스

### `envs/inference` (추론 환경)
- **transformers**: >=4.51.1 (최신)
- **tokenizers**: >=0.21.1 (최신)
- **vllm**: ==0.10.2 (고속 추론 서버)
- **torch**: >=2.7.0
- **기타**: 추론에 필요한 대규모 패키지들 (`flash-attn`, `xformers`, 등)

### `envs/evaluation` (평가 환경)
- **transformers**: ==4.42.4 (구버전 고정)
- **tokenizers**: >=0.19.0,<0.20.0 (구버전)
- **aac-metrics**: >=0.6.0 (평가 메트릭)
- **sentence-transformers**: >=5.1.2 (텍스트 임베딩)
- **torch**: ==2.7.0 (inference와 동일 버전 유지)

---

## 사용 방법

### 환경 초기화
repo 루트에서 두 env를 모두 동기화합니다:

```bash
uv sync --project envs/inference
uv sync --project envs/evaluation
```

### 실행 표준 명령

- **전체 파이프라인** (추론 + 평가):
  ```bash
  uv run --project envs/inference python cli.py run <benchmark> [--model MODEL]
  ```

- **추론만**:
  ```bash
  uv run --project envs/inference python cli.py inference <benchmark> [--model MODEL]
  ```

- **평가만**:
  ```bash
  uv run --project envs/evaluation python cli.py eval <benchmark> [--model MODEL]
  ```

---

## 버전 업데이트 규칙

두 환경의 의존성을 변경할 때:

1. **각 환경의 `pyproject.toml` 수정** (예: 새 패키지 추가, 버전 변경)
2. **각 env의 `uv.lock` 갱신**:
   ```bash
   uv lock --project envs/inference
   uv lock --project envs/evaluation
   ```
3. **동작 확인** (smoke test):
   ```bash
   uv run --project envs/inference python cli.py --help
   uv run --project envs/evaluation python cli.py --help
   ```
4. **최소한 한 모델로 실행 테스트** (예: `gemma3n`으로 clotho inference/eval 각 1회)

---

## 주의사항

- 루트 `pyproject.toml`은 **런타임 의존성을 가지지 않으며** (최소화), 메타데이터만 담습니다.
- 실행 시에는 **항상 `uv run --project envs/<...>`를 통해 env별 가상환경을 사용**합니다.
- 두 env는 독립적이므로, 한쪽 업데이트가 다른 쪽에 영향을 주지 않습니다.

