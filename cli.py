from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

import typer


def _detect_root() -> Path:
    env_override = os.environ.get("LALM_BENCH_ROOT")
    candidates = []
    if env_override:
        candidates.append(Path(env_override).expanduser())
    candidates.append(Path.cwd())
    candidates.append(Path(__file__).resolve().parent)
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "envs").exists():
            return resolved
    raise typer.BadParameter(
        "프로젝트 루트를 찾을 수 없습니다. "
        "프로젝트 루트에서 명령을 실행하거나 LALM_BENCH_ROOT를 설정하세요."
    )


ROOT = _detect_root()
SRC_DIR = ROOT / "src"
INFERENCE_PROJECT = ROOT / "envs" / "inference"
EVALUATION_PROJECT = ROOT / "envs" / "evaluation"
DEFAULT_MODELS = ("gemma3n", "qwen2_5-omni", "qwen3-omni")

app = typer.Typer(
    help="Unified launcher for lalm_bench benchmarks. "
    "Always run via `uv run --project envs/<inference|evaluation> python cli.py <command>`."
)


class Benchmark(str, Enum):
    clotho = "clotho"
    mmau_pro = "mmau-pro"


@dataclass(frozen=True)
class BenchmarkConfig:
    display_name: str
    subdir: str
    inference_log: str
    evaluation_log: str
    inference_builder: Callable[
        [str, Path, bool, Optional[List[int]], Optional[str], bool], List[str]
    ]
    evaluation_builder: Callable[[Path], List[str]]


def _clotho_inference_args(
    model: str,
    bench_dir: Path,
    save_attn: bool = False,
    attn_layers: Optional[List[int]] = None,
    attn_run_name: Optional[str] = None,
    use_white_noise: bool = False,
) -> List[str]:
    args = [
        "src/clotho/inference.py",
        "--split",
        "evaluation",
        "--model",
        model,
        "--output_json_path",
        str(bench_dir),
        "-t",
    ]
    if save_attn:
        args.append("--save-attn")
    if attn_layers:
        args.append("--attn-layers")
        args.extend(str(layer) for layer in attn_layers)
    if attn_run_name:
        args.extend(["--attn-run-name", attn_run_name])
    if use_white_noise:
        args.append("--use-white-noise")
    return args


def _clotho_evaluation_args(bench_dir: Path) -> List[str]:
    predictions = bench_dir / "predictions.json"
    if not predictions.exists():
        raise FileNotFoundError(
            f"Clotho predictions not found at {predictions}. Run inference first."
        )
    scores = bench_dir / "scores.json"
    return [
        "src/clotho/evaluation.py",
        "--input_json_path",
        str(predictions),
        "--output_json_path",
        str(scores),
        "-t",
    ]


def _mmau_pro_inference_args(
    model: str,
    bench_dir: Path,
    save_attn: bool = False,
    attn_layers: Optional[List[int]] = None,
    attn_run_name: Optional[str] = None,
    use_white_noise: bool = False,
) -> List[str]:
    args = [
        "src/mmau-pro/inference.py",
        "--split",
        "evaluation",
        "--verbose",
        "--model",
        model,
        "--output",
        str(bench_dir),
        "-t",
    ]
    if save_attn:
        args.append("--save-attn")
    if attn_layers:
        args.append("--attn-layers")
        args.extend(str(layer) for layer in attn_layers)
    if attn_run_name:
        args.extend(["--attn-run-name", attn_run_name])
    if use_white_noise:
        args.append("--use-white-noise")
    return args


def _mmau_pro_evaluation_args(bench_dir: Path) -> List[str]:
    predictions = bench_dir / "predictions.parquet"
    if not predictions.exists():
        raise FileNotFoundError(
            f"MMAU-Pro predictions not found at {predictions}. Run inference first."
        )
    return [
        "src/mmau-pro/evaluation.py",
        str(predictions),
        "-t",
    ]


CONFIGS: Dict[Benchmark, BenchmarkConfig] = {
    Benchmark.clotho: BenchmarkConfig(
        display_name="Clotho",
        subdir="clotho",
        inference_log="clotho_infer.stderr.log",
        evaluation_log="clotho_eval.stderr.log",
        inference_builder=_clotho_inference_args,
        evaluation_builder=_clotho_evaluation_args,
    ),
    Benchmark.mmau_pro: BenchmarkConfig(
        display_name="MMAU-Pro",
        subdir="mmau-pro",
        inference_log="mmau-pro_infer.stderr.log",
        evaluation_log="mmau-pro_eval.stderr.log",
        inference_builder=_mmau_pro_inference_args,
        evaluation_builder=_mmau_pro_evaluation_args,
    ),
}


def _timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _safe_name(benchmark: Benchmark) -> str:
    return benchmark.value.replace("-", "_")


def _append_line(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text.rstrip("\n") + "\n")


def _append_block(path: Path, block: str) -> None:
    for line in block.rstrip("\n").splitlines():
        _append_line(path, line)


def _initialize_result_file(path: Path, title: str, model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"{title} for {model}\n")
        handle.write(f"Started at: {_timestamp()}\n")
        handle.write("=================================\n")


def _finalize_result_file(path: Path, label: str) -> None:
    _append_line(path, f"Finished at: {_timestamp()}")
    _append_line(path, f"=== Completed {label} ===")


def _env_with_src() -> Dict[str, str]:
    env = os.environ.copy()
    root_path = str(ROOT)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{root_path}{os.pathsep}{existing}" if existing else root_path
    return env


def _tail_file(path: Path, limit: int = 50) -> str:
    if not path.exists():
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        lines = handle.readlines()
    return "".join(lines[-limit:])


def _run_phase(
    phase_label: str,
    project_path: Path,
    args: List[str],
    result_file: Path,
    error_log: Path,
) -> None:
    env = _env_with_src()
    error_log.parent.mkdir(parents=True, exist_ok=True)
    start = time.time()
    typer.echo(f"  [TASK] {phase_label}")
    _append_line(result_file, f"[{phase_label}] Started at {_timestamp()}")
    command = ["uv", "run", "--project", str(project_path), *args]
    try:
        with open(error_log, "w", encoding="utf-8") as err_stream:
            subprocess.run(
                command,
                cwd=ROOT,
                env=env,
                check=True,
                stderr=err_stream,
            )
    except subprocess.CalledProcessError as exc:
        duration = int(time.time() - start)
        _append_line(
            result_file,
            f"❌ {phase_label} failed ({duration}s). See {error_log}",
        )
        tail = _tail_file(error_log)
        if tail:
            _append_line(result_file, "--- stderr (tail) ---")
            _append_block(result_file, tail)
        typer.secho(
            f"{phase_label} failed. Inspect {error_log} for details.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(exc.returncode)
    duration = int(time.time() - start)
    _append_line(result_file, f"✅ {phase_label} completed ({duration}s)")


def _resolve_models(models: Optional[List[str]]) -> List[str]:
    return list(models) if models else list(DEFAULT_MODELS)


def _model_output_dir(model: str, use_white_noise: bool) -> str:
    if use_white_noise:
        return f"{model}_with_noise"
    return model


def _normalize_output_root(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _ensure_env_exists(path: Path, kind: str) -> None:
    if not path.exists():
        raise typer.BadParameter(
            f"{kind} environment is missing at {path}. "
            f"Run `uv sync --project {path}` first."
        )


def _build_benchmark_dir(
    output_root: Path, model: str, config: BenchmarkConfig
) -> Path:
    bench_dir = output_root / model / config.subdir
    bench_dir.mkdir(parents=True, exist_ok=True)
    return bench_dir


def _run_inference(
    benchmark: Benchmark,
    models: List[str],
    output_root: Path,
    save_attn: bool = False,
    attn_layers: Optional[List[int]] = None,
    attn_run_name: Optional[str] = None,
    use_white_noise: bool = False,
) -> None:
    _ensure_env_exists(INFERENCE_PROJECT, "Inference")
    config = CONFIGS[benchmark]
    for model in models:
        typer.echo(f"=== Running {config.display_name} inference for {model} ===")
        model_dir = _model_output_dir(model, use_white_noise)
        bench_dir = _build_benchmark_dir(output_root, model_dir, config)
        result_file = (
            output_root / model_dir / f"result_{_safe_name(benchmark)}_inference.txt"
        )
        _initialize_result_file(
            result_file, f"{config.display_name} Inference Results", model
        )
        args = config.inference_builder(
            model,
            bench_dir,
            save_attn=save_attn,
            attn_layers=attn_layers,
            attn_run_name=attn_run_name,
            use_white_noise=use_white_noise,
        )
        _run_phase(
            f"{config.display_name} inference",
            INFERENCE_PROJECT,
            args,
            result_file,
            (output_root / model_dir / config.inference_log),
        )
        _finalize_result_file(
            result_file, f"{config.display_name} inference for {model}"
        )


def _run_evaluation(
    benchmark: Benchmark,
    models: List[str],
    output_root: Path,
) -> None:
    _ensure_env_exists(EVALUATION_PROJECT, "Evaluation")
    config = CONFIGS[benchmark]
    for model in models:
        typer.echo(f"=== Running {config.display_name} evaluation for {model} ===")
        bench_dir = _build_benchmark_dir(output_root, model, config)
        result_file = (
            output_root / model / f"result_{_safe_name(benchmark)}_evaluation.txt"
        )
        _initialize_result_file(
            result_file, f"{config.display_name} Evaluation Results", model
        )
        try:
            args = config.evaluation_builder(bench_dir)
        except FileNotFoundError as err:
            typer.secho(str(err), fg=typer.colors.RED)
            raise typer.Exit(1) from err
        _run_phase(
            f"{config.display_name} evaluation",
            EVALUATION_PROJECT,
            args,
            result_file,
            (output_root / model / config.evaluation_log),
        )
        _finalize_result_file(
            result_file, f"{config.display_name} evaluation for {model}"
        )


def _run_pipeline(
    benchmark: Benchmark,
    models: List[str],
    output_root: Path,
    save_attn: bool = False,
    attn_layers: Optional[List[int]] = None,
    attn_run_name: Optional[str] = None,
    use_white_noise: bool = False,
) -> None:
    _ensure_env_exists(INFERENCE_PROJECT, "Inference")
    _ensure_env_exists(EVALUATION_PROJECT, "Evaluation")
    config = CONFIGS[benchmark]
    for model in models:
        typer.echo(f"=== Running {config.display_name} benchmark for {model} ===")
        model_dir = _model_output_dir(model, use_white_noise)
        bench_dir = _build_benchmark_dir(output_root, model_dir, config)
        result_file = output_root / model_dir / f"result_{_safe_name(benchmark)}.txt"
        _initialize_result_file(
            result_file, f"{config.display_name} Benchmark Results", model
        )
        inference_args = config.inference_builder(
            model,
            bench_dir,
            save_attn=save_attn,
            attn_layers=attn_layers,
            attn_run_name=attn_run_name,
            use_white_noise=use_white_noise,
        )
        _run_phase(
            f"{config.display_name} inference",
            INFERENCE_PROJECT,
            inference_args,
            result_file,
            (output_root / model_dir / config.inference_log),
        )
        try:
            eval_args = config.evaluation_builder(bench_dir)
        except FileNotFoundError as err:
            typer.secho(str(err), fg=typer.colors.RED)
            raise typer.Exit(1) from err
        _run_phase(
            f"{config.display_name} evaluation",
            EVALUATION_PROJECT,
            eval_args,
            result_file,
            (output_root / model_dir / config.evaluation_log),
        )
        _finalize_result_file(
            result_file, f"{config.display_name} benchmark for {model}"
        )


@app.command()
def inference(
    benchmark: Benchmark = typer.Argument(..., help="Target benchmark to run."),
    model: Optional[List[str]] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name(s). Repeat the flag to run multiple models.",
    ),
    output_root: Path = typer.Option(
        Path("./outputs"),
        "--output-root",
        help="Base directory for benchmark outputs.",
    ),
    save_attn: bool = typer.Option(
        False,
        "--save-attn/--no-save-attn",
        help="Attention map을 저장하려면 활성화합니다.",
    ),
    attn_layers: Optional[List[int]] = typer.Option(
        None,
        "--attn-layers",
        help="저장할 레이어 인덱스(0 기반)",
    ),
    attn_run_name: Optional[str] = typer.Option(
        None,
        "--attn-run-name",
        help="어텐션 저장 run 이름(미지정 시 timestamp).",
    ),
    use_white_noise: bool = typer.Option(
        False,
        "--use-white-noise/--no-use-white-noise",
        help="white-noise-358382.mp3를 강제로 사용합니다.",
    ),
) -> None:
    """
    Run inference only for the selected benchmark.
    """
    selected_models = _resolve_models(model)
    output_root = _normalize_output_root(output_root)
    _run_inference(
        benchmark,
        selected_models,
        output_root,
        save_attn=save_attn,
        attn_layers=attn_layers,
        attn_run_name=attn_run_name,
        use_white_noise=use_white_noise,
    )


@app.command("eval")
def evaluate(
    benchmark: Benchmark = typer.Argument(..., help="Target benchmark to run."),
    model: Optional[List[str]] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name(s). Repeat the flag to run multiple models.",
    ),
    output_root: Path = typer.Option(
        Path("./outputs"),
        "--output-root",
        help="Base directory for benchmark outputs.",
    ),
) -> None:
    """
    Run evaluation only for the selected benchmark.
    """
    selected_models = _resolve_models(model)
    output_root = _normalize_output_root(output_root)
    _run_evaluation(benchmark, selected_models, output_root)


@app.command("run")
def run_pipeline(
    benchmark: Benchmark = typer.Argument(..., help="Target benchmark to run."),
    model: Optional[List[str]] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model name(s). Repeat the flag to run multiple models.",
    ),
    output_root: Path = typer.Option(
        Path("./outputs"),
        "--output-root",
        help="Base directory for benchmark outputs.",
    ),
    save_attn: bool = typer.Option(
        False,
        "--save-attn/--no-save-attn",
        help="Attention map을 저장하려면 활성화합니다.",
    ),
    attn_layers: Optional[List[int]] = typer.Option(
        None,
        "--attn-layers",
        help="저장할 레이어 인덱스(0 기반)",
    ),
    attn_run_name: Optional[str] = typer.Option(
        None,
        "--attn-run-name",
        help="어텐션 저장 run 이름(미지정 시 timestamp).",
    ),
    use_white_noise: bool = typer.Option(
        False,
        "--use-white-noise/--no-use-white-noise",
        help="white-noise-358382.mp3를 강제로 사용합니다.",
    ),
) -> None:
    """
    Run inference followed by evaluation for the selected benchmark.
    """
    selected_models = _resolve_models(model)
    output_root = _normalize_output_root(output_root)
    _run_pipeline(
        benchmark,
        selected_models,
        output_root,
        save_attn=save_attn,
        attn_layers=attn_layers,
        attn_run_name=attn_run_name,
        use_white_noise=use_white_noise,
    )


if __name__ == "__main__":
    app()
