# 자동 루브릭 생성 & 후처리

**목표**: Qwen 모델 기반으로 자동 루브릭을 생성하고 중복을 제거합니다.

## 진입점

```bash
uv run python experiments/rubrics/make_rubrics/run.py [options]
```

## 주요 스크립트

| 스크립트 | 목표 |
|----------|------|
| `run.py` | 주 루브릭 생성 파이프라인 |
| `judge.py` | Qwen 모델 기반 판단 로직 |
| `rm_dup.py` | 생성된 루브릭 중복 제거 |

## 입출력

**입력**: 벤치마크 예측 결과 (Parquet/JSON)
**출력**: 
- `stoch/results_*.json`: 개별 루브릭 결과
- `stoch_new/`: 중복 제거 후 결과

## 사용 예시

```bash
# 기본 실행
uv run python experiments/rubrics/make_rubrics/run.py

# 커스텀 출력 디렉토리 지정
uv run python experiments/rubrics/make_rubrics/run.py --output-dir custom_out
```

> 상세 옵션은 `run.py --help` 를 참조하세요.

