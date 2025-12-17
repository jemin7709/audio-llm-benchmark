# 어텐션 시각화 (Gemma3N)

**목표**: Gemma3N 모델의 레이어별 어텐션 맵을 추출하여 이미지, NPY, JSON 형식으로 저장합니다.

## 입력

- `--prompt`: 단일 사용자 메시지 (문자열)
- `--input-jsonl` (선택): JSONL 파일 경로. 각 줄은 `{'prompt': '...'}` 또는 `{'messages': [...]}` 형식

## 출력

결과는 `--output-dir` 아래 생성됩니다(기본값: `outputs/attn`):

```
outputs/attn/{run_name}/
├── sample_{idx}/
│   ├── attn.npy             # 레이어별 어텐션 행렬
│   ├── tokens.json          # 토큰 목록
│   ├── meta.json            # 메타데이터
│   ├── layer_*.jpg          # 개별 레이어 시각화
│   └── layer_mean.jpg       # 평균 어텐션 시각화
├── global_layer_mean.npy    # 전체 평균 (레이어 차원)
├── global_mean.npy          # 전체 평균 (모든 차원)
├── global_layer_mean.jpg
├── global_mean.jpg
└── stats.json               # 수집 통계
```

## 사용 예시

### 단일 프롬프트 처리
```bash
uv run python experiments/attention/visualization.py \
  --prompt "Test" \
  --layers 0 1 \
  --limit-samples 1 \
  --output-dir outputs/attn/smoke
```

### JSONL 파일 처리
```bash
uv run python experiments/attention/visualization.py \
  --input-jsonl data/samples.jsonl \
  --layers all \
  --output-dir outputs/attn/batch
```

## 옵션

- `--model`: HuggingFace 모델 ID (기본: `google/gemma-3n-E4B-it`)
- `--device`: GPU 디바이스 (기본: `cuda:0`)
- `--dtype`: 모델 dtype (기본: `auto`)
- `--layers`: 시각화할 레이어 인덱스 또는 `all` (기본: `all`)
- `--max-layers`: 처리할 상위 N개 레이어만
- `--limit-samples`: 처리 샘플 수 상한
- `--seed`: 난수 시드 (기본: 42)

