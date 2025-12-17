# Experiments & Analysis

이 폴더는 LALM Bench의 메인 벤치마크 파이프라인이 아니며, **보조 분석 및 검증용 실험 스크립트**를 제공합니다.

## 실행 규약

모든 사이드 실험은 다음 형태로 실행합니다:

```bash
uv run python experiments/<area>/<script>.py [options]
```

각 실험 디렉토리의 상세 입력/출력 경로 및 사용 예시는 해당 폴더의 `README.md`를 참조하세요.

## 실험 목록

| 영역 | 목표 | 진입점 |
|------|------|--------|
| `attention/` | 어텐션 맵 추출 & 시각화 | `visualization.py` |
| `similarity/` | 예측 유사도 분석 | `audio_similarity/`, `clotho_ref_similarity/` |
| `rubrics/` | 자동 루브릭 생성 | `make_rubrics/` |

