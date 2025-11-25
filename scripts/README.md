# Scripts Directory

이제 벤치마크 실행은 Typer 기반 CLI(`cli.py`)가 담당하며, `scripts/` 디렉토리는 필수 보조 스크립트만 포함합니다.

## 남아 있는 스크립트

| 파일 | 설명 |
|------|------|
| `download_datasets.sh` | Hugging Face에서 필요한 데이터셋을 일괄 다운로드합니다. |
| `install_vllm.sh` | Docker 이미지 빌드 시 vLLM 관련 의존성을 설치합니다. |

## 실행 흐름

1. 루트 환경 동기화  
   ```bash
   uv sync
   ```
2. 전용 가상환경 준비  
   ```bash
   uv sync --project envs/inference
   uv sync --project envs/evaluation
   ```
3. (선택) 데이터 다운로드  
   ```bash
   bash scripts/download_datasets.sh
   ```
4. Typer CLI 사용  
   ```bash
   # 예시: Gemma3N으로 Clotho 전체 파이프라인 실행
   uv run lalm run clotho --model gemma3n
   ```

세부 옵션과 결과 구조는 루트 `README.md`를 참고하세요.

