from __future__ import annotations

import json
import getpass
import os
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

# -----------------------------------------------------------------------------
# Pretty terminal printing
# -----------------------------------------------------------------------------

USE_COLOR_PRINTS = True


class T:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    MAGENTA = "\033[95m"


def color_text(text: str, color: str = "", bold: bool = False) -> str:
    if not USE_COLOR_PRINTS:
        return text

    prefix = ""
    if bold:
        prefix += T.BOLD
    prefix += color
    return f"{prefix}{text}{T.RESET}"


def print_section(title: str, color: str = T.CYAN) -> None:
    line = "=" * 90
    print("\n" + color_text(line, color, bold=True))
    print(color_text(title, color, bold=True))
    print(color_text(line, color, bold=True))


def print_subsection(title: str, color: str = T.BLUE) -> None:
    line = "-" * 80
    print("\n" + color_text(line, color, bold=True))
    print(color_text(title, color, bold=True))
    print(color_text(line, color, bold=True))


def print_info(msg: str) -> None:
    print(color_text("[INFO] ", T.GREEN, bold=True) + msg)


def print_warn(msg: str) -> None:
    print(color_text("[WARN] ", T.YELLOW, bold=True) + msg)


# -----------------------------------------------------------------------------
# GPU setup
# -----------------------------------------------------------------------------

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


def choose_gpu_for_rl(
    preferred_max_mem_mb: int = 2000,
    preferred_max_util: int = 20,
) -> int | None:
    try:
        result = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,nounits,noheader",
            ],
            stderr=subprocess.STDOUT,
        ).decode("utf-8", errors="replace").strip()

        if not result:
            print_warn("nvidia-smi returned no GPU information.")
            return None

        rows: list[tuple[int, int, int]] = []
        for line in result.splitlines():
            idx, mem, util = [x.strip() for x in line.split(",")]
            rows.append((int(idx), int(mem), int(util)))

        if not rows:
            print_warn("No GPUs parsed from nvidia-smi output.")
            return None

        preferred = [
            row for row in rows
            if row[1] <= preferred_max_mem_mb and row[2] <= preferred_max_util
        ]

        candidates = preferred if preferred else rows
        candidates.sort(key=lambda x: (x[1], x[2], x[0]))
        gpu_id, mem_mb, util_pct = candidates[0]

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ["PYOPENCL_CTX"] = f"0:{gpu_id}"

        print_info(f"Selected GPU {gpu_id} (mem={mem_mb} MiB, util={util_pct}%)")
        return gpu_id

    except Exception as e:
        print_warn(f"Automatic GPU selection failed: {e}")
        print_warn("Proceeding without forcing CUDA_VISIBLE_DEVICES.")
        return None


choose_gpu_for_rl()

# -----------------------------------------------------------------------------
# Imports after GPU selection
# -----------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from csbdeep.io import load_training_data
from csbdeep.models import CARE, Config
from csbdeep.utils import axes_dict, plot_history, plot_some

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print_info(f"TensorFlow sees {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print_warn(f"Could not set TensorFlow memory growth: {e}")
else:
    print_info("TensorFlow sees no GPU. Training will run on CPU.")


# -----------------------------------------------------------------------------
# JSON helper
# -----------------------------------------------------------------------------

def to_jsonable(obj):
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def load_json_file(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# -----------------------------------------------------------------------------
# Patch file helpers
# -----------------------------------------------------------------------------

def find_patch_dir(
    care_root: Path,
    patch_dirname_candidates: Sequence[str] = ("train_patches", "train patches"),
) -> Path:
    for name in patch_dirname_candidates:
        p = care_root / name
        if p.exists():
            return p
    return care_root / patch_dirname_candidates[0]


def resolve_patch_file(
    care_root: Path,
    patch_dirname_candidates: Sequence[str] = ("train_patches", "train patches"),
    patch_file_name: Optional[str] = None,
) -> tuple[Path, Path]:
    patch_dir = find_patch_dir(care_root, patch_dirname_candidates)
    patch_dir.mkdir(parents=True, exist_ok=True)

    if patch_file_name is not None:
        patch_file = patch_dir / patch_file_name
        if not patch_file.exists():
            raise FileNotFoundError(f"Patch file not found: {patch_file}")
        print_info(f"Using explicit patch file: {patch_file}")
        return patch_dir, patch_file

    patch_files = sorted(patch_dir.glob("*.npz"))

    if not patch_files:
        raise FileNotFoundError(f"No .npz patch files found in {patch_dir}")

    if len(patch_files) == 1:
        print_info(f"Only one patch file found; using: {patch_files[0].name}")
        return patch_dir, patch_files[0]

    print_warn(f"Multiple patch files found in {patch_dir}.")
    print("Available patch files:")
    for p in patch_files:
        print("  -", p.name)

    raise FileNotFoundError(
        "\nMultiple patch files found.\n"
        "Please specify patch_file_name explicitly to keep training reproducible."
    )


def resolve_patch_metadata_file(patch_file: Path) -> Optional[Path]:
    metadata_file = patch_file.with_suffix(".json")

    if metadata_file.exists():
        return metadata_file

    print_info(f"No metadata file found for: {patch_file.name}")
    return None


def build_model_run_name(
    *,
    patch_file: Path,
    prefix: str = "CARE",
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    return f"{prefix}__{patch_file.stem}__{timestamp}"


# -----------------------------------------------------------------------------
# Patch loading / validation
# -----------------------------------------------------------------------------

def summarize_loaded_patches(X: np.ndarray, Y: np.ndarray, name: str = "training") -> None:
    print_subsection(f"Patch summary: {name}", color=T.MAGENTA)

    print_info(f"X shape: {X.shape}")
    print_info(f"Y shape: {Y.shape}")
    print_info(f"X has NaN: {bool(np.isnan(X).any())}")
    print_info(f"Y has NaN: {bool(np.isnan(Y).any())}")
    print_info(f"X has inf: {bool(np.isinf(X).any())}")
    print_info(f"Y has inf: {bool(np.isinf(Y).any())}")
    print_info(f"X min/max: {float(np.min(X)):.6g} / {float(np.max(X)):.6g}")
    print_info(f"Y min/max: {float(np.min(Y)):.6g} / {float(np.max(Y)):.6g}")

    x_std = np.std(X.astype(np.float64), axis=(1, 2, 3))
    y_std = np.std(Y.astype(np.float64), axis=(1, 2, 3))

    print_info(f"Constant/near-constant X patches: {int(np.sum(x_std <= 1e-6))}")
    print_info(f"Constant/near-constant Y patches: {int(np.sum(y_std <= 1e-6))}")


def validate_loaded_patches(X: np.ndarray, Y: np.ndarray, name: str = "training") -> None:
    if X.size == 0 or Y.size == 0:
        raise ValueError(f"{name} patches are empty.")

    if np.isnan(X).any() or np.isnan(Y).any():
        raise ValueError(f"{name} patches contain NaNs.")
    if np.isinf(X).any() or np.isinf(Y).any():
        raise ValueError(f"{name} patches contain inf values.")

    if np.max(np.abs(X)) > 1e6 or np.max(np.abs(Y)) > 1e6:
        raise ValueError(
            f"{name} patches contain extreme values (> 1e6). "
            "This usually indicates a bad patch file."
        )

    x_std = np.std(X.astype(np.float64), axis=(1, 2, 3))
    y_std = np.std(Y.astype(np.float64), axis=(1, 2, 3))

    if np.all(x_std <= 1e-6):
        raise ValueError(f"All {name} X patches are constant/near-constant.")
    if np.all(y_std <= 1e-6):
        raise ValueError(f"All {name} Y patches are constant/near-constant.")


def load_care_training_data(
    patch_file: Path,
    validation_split: float,
):
    print_section("Loading CARE training data")

    (X, Y), (X_val, Y_val), axes = load_training_data(
        str(patch_file),
        validation_split=validation_split,
    )

    ad = axes_dict(axes)
    if "C" not in ad:
        raise ValueError(f"Expected channel axis in axes='{axes}', but none was found.")

    c = ad["C"]
    n_channel_in = X.shape[c]
    n_channel_out = Y.shape[c]

    print_info(f"Patch file: {patch_file}")
    print_info(f"Axes: {axes}")
    print_info(f"Training X: {X.shape}")
    print_info(f"Training Y: {Y.shape}")
    print_info(f"Validation X: {X_val.shape}")
    print_info(f"Validation Y: {Y_val.shape}")
    print_info(f"Input channels: {n_channel_in}")
    print_info(f"Output channels: {n_channel_out}")
    print_info(f"Validation split: {validation_split}")

    patch_metadata_file = resolve_patch_metadata_file(patch_file)

    if patch_metadata_file is not None:
        print_info(f"Patch metadata file: {patch_metadata_file}")
        try:
            patch_metadata = load_json_file(patch_metadata_file)
            if "n_patches_total" in patch_metadata:
                print_info(f"Total patches in metadata: {patch_metadata['n_patches_total']}")
            if "n_pairs_used" in patch_metadata:
                print_info(f"Image pairs used in metadata: {patch_metadata['n_pairs_used']}")
        except Exception as e:
            print_warn(f"Could not read patch metadata file: {e}")
    else:
        print_warn("Patch metadata file not found.")

    summarize_loaded_patches(X, Y, name="training")
    summarize_loaded_patches(X_val, Y_val, name="validation")

    validate_loaded_patches(X, Y, name="training")
    validate_loaded_patches(X_val, Y_val, name="validation")

    return X, Y, X_val, Y_val, axes, n_channel_in, n_channel_out


# -----------------------------------------------------------------------------
# Optional normalization
# -----------------------------------------------------------------------------

def normalize_percentile(
    arr: np.ndarray,
    pmin: float = 1.0,
    pmax: float = 99.8,
    eps: float = 1e-8,
) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo = np.percentile(arr, pmin)
    hi = np.percentile(arr, pmax)
    return np.clip((arr - lo) / (hi - lo + eps), 0, 1)


def normalize_patch_dataset(
    X: np.ndarray,
    Y: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    *,
    enabled: bool = False,
    pmin: float = 1.0,
    pmax: float = 99.8,
    eps: float = 1e-8,
):
    print_section("Normalization settings")

    print_info(f"Enabled: {enabled}")

    if not enabled:
        print_info("Percentile normalization is OFF. Raw patch values will be used.")
        return X, Y, X_val, Y_val

    print_info(f"Applying percentile normalization with pmin={pmin}, pmax={pmax}, eps={eps}")

    print_info(f"Before normalization X max: {X.max():.6g}")
    print_info(f"Before normalization Y max: {Y.max():.6g}")

    X = np.stack([normalize_percentile(x, pmin=pmin, pmax=pmax, eps=eps) for x in X], axis=0)
    Y = np.stack([normalize_percentile(y, pmin=pmin, pmax=pmax, eps=eps) for y in Y], axis=0)
    X_val = np.stack([normalize_percentile(x, pmin=pmin, pmax=pmax, eps=eps) for x in X_val], axis=0)
    Y_val = np.stack([normalize_percentile(y, pmin=pmin, pmax=pmax, eps=eps) for y in Y_val], axis=0)

    print_info(f"After normalization X max: {X.max():.6g}")
    print_info(f"After normalization Y max: {Y.max():.6g}")

    return X, Y, X_val, Y_val


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------

def plot_patch_examples(
    X: np.ndarray,
    Y: np.ndarray,
    n_show: int = 5,
    title: str = "Example patch pairs (top: input, bottom: target)",
    pmin: float = 1.0,
    pmax: float = 99.8,
):
    """
    Plot a few input/target patch pairs.

    The pmin/pmax values are used only for display scaling.
    They do not modify the arrays used for training.
    """
    n_show = min(n_show, len(X))

    plt.figure(figsize=(12, 5))
    plot_some(
        X[:n_show],
        Y[:n_show],
        pmin=pmin,
        pmax=pmax,
    )
    plt.suptitle(title)
    plt.show()


def plot_training_curves(history):
    print_section("Training curves")

    keys = sorted(history.history.keys())
    print_info(f"Available metrics: {keys}")

    loss_keys = [k for k in ["loss", "val_loss"] if k in history.history]
    metric_keys = [k for k in ["mse", "val_mse", "mae", "val_mae"] if k in history.history]

    plt.figure(figsize=(12, 4))
    plot_history(history, loss_keys, metric_keys)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# CARE model config / training
# -----------------------------------------------------------------------------

def build_care_config(
    *,
    axes: str,
    n_channel_in: int,
    n_channel_out: int,
    train_batch_size: int,
    train_steps_per_epoch: int,
    train_epochs: int,
    unet_kern_size: int = 3,
    unet_n_depth: int = 2,
    train_learning_rate: float = 2e-4,
    train_reduce_lr: Optional[dict] = None,
    probabilistic: bool = False,
    train_loss: Optional[str] = None,
):
    print_section("CARE model configuration")

    if train_reduce_lr is None:
        train_reduce_lr = {"factor": 0.5, "patience": 10}

    config = Config(
        axes=axes,
        n_channel_in=n_channel_in,
        n_channel_out=n_channel_out,
        probabilistic=probabilistic,
        unet_kern_size=unet_kern_size,
        unet_n_depth=unet_n_depth,
        train_batch_size=train_batch_size,
        train_steps_per_epoch=train_steps_per_epoch,
        train_epochs=train_epochs,
        train_learning_rate=train_learning_rate,
        train_reduce_lr=train_reduce_lr,
    )

    if train_loss is not None:
        try:
            config.train_loss = train_loss
            print_info(f"Using training loss: {config.train_loss}")
        except Exception as e:
            print_warn(
                f"Could not set train_loss={train_loss!r}. "
                f"Proceeding with default CARE loss. Error: {e}"
            )

    print_info(f"axes: {config.axes}")
    print_info(f"n_channel_in: {config.n_channel_in}")
    print_info(f"n_channel_out: {config.n_channel_out}")
    print_info(f"probabilistic: {config.probabilistic}")
    print_info(f"unet_kern_size: {config.unet_kern_size}")
    print_info(f"unet_n_depth: {config.unet_n_depth}")
    print_info(f"train_batch_size: {config.train_batch_size}")
    print_info(f"train_steps_per_epoch: {config.train_steps_per_epoch}")
    print_info(f"train_epochs: {config.train_epochs}")
    print_info(f"train_learning_rate: {config.train_learning_rate}")
    print_info(f"train_reduce_lr: {config.train_reduce_lr}")

    if hasattr(config, "train_loss"):
        print_info(f"train_loss: {config.train_loss}")

    return config


def create_care_model(
    *,
    config: Config,
    model_name: str,
    model_dir: Path,
) -> CARE:
    print_section("Creating CARE model")
    print_info(f"Model name: {model_name}")
    print_info(f"Model directory: {model_dir}")

    model = CARE(
        config=config,
        name=model_name,
        basedir=str(model_dir),
    )
    return model


def print_model_save_info(
    *,
    model_dir: Path,
    model_name: str,
):
    out_dir = model_dir / model_name

    print_section("Model output locations")
    print_info(f"Model output directory: {out_dir}")
    print_info("Expected files:")
    print("  - weights_best.h5")
    print("  - weights_last.h5")
    print("  - config.json")
    print("  - training_metadata.json")
    print("  - training_history.json")
    print_info("CARE usually restores weights_best.h5 at the end of training.")


def train_care_model(
    *,
    model: CARE,
    X: np.ndarray,
    Y: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
):
    print_section("Starting CARE training")

    try:
        history = model.train(X, Y, validation_data=(X_val, Y_val))
    except Exception:
        print_warn("Training failed.")
        print_warn("This is often caused by bad patches, NaNs, infs, constants, or extreme values.")
        raise

    print_section("CARE training complete", color=T.GREEN)
    return history


def save_training_history(
    history,
    *,
    model_dir: Path,
    model_name: str,
    filename: str = "training_history.json",
) -> Path:
    out_dir = model_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    history_clean = to_jsonable(history.history)

    history_file = out_dir / filename
    with open(history_file, "w") as f:
        json.dump(history_clean, f, indent=2)

    print_info(f"Saved training history: {history_file}")
    return history_file


# -----------------------------------------------------------------------------
# Validation prediction check
# -----------------------------------------------------------------------------

def summarize_validation_prediction(
    Y_true: np.ndarray,
    Y_pred: np.ndarray,
) -> dict[str, float]:
    mae = float(np.mean(np.abs(Y_true - Y_pred)))
    mse = float(np.mean((Y_true - Y_pred) ** 2))
    return {"mae": mae, "mse": mse}


def predict_on_validation_examples(
    *,
    model: CARE,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    probabilistic: bool,
    n_examples: int = 5,
):
    print_section("Validation prediction check")

    n_examples = min(n_examples, len(X_val))

    X_example = X_val[:n_examples]
    Y_example = Y_val[:n_examples]

    Y_pred = model.keras_model.predict(X_example, verbose=0)

    if probabilistic:
        Y_pred = Y_pred[..., : Y_pred.shape[-1] // 2]

    plt.figure(figsize=(20, 12))
    plot_some(X_example, Y_example, Y_pred, pmax=99.5)
    plt.suptitle(
        f"{n_examples} validation patches\n"
        "top: input, middle: target, bottom: prediction",
        y=0.98,
    )
    plt.show()

    summary = summarize_validation_prediction(Y_example, Y_pred)

    print_info(f"Validation MAE: {summary['mae']:.6g}")
    print_info(f"Validation MSE: {summary['mse']:.6g}")

    return X_example, Y_example, Y_pred


# -----------------------------------------------------------------------------
# Metadata
# -----------------------------------------------------------------------------

def build_training_metadata(
    *,
    care_root: Path,
    patch_dir: Path,
    patch_file: Path,
    model_dir: Path,
    model_name: str,
    validation_split: float,
    train_batch_size: int,
    train_steps_per_epoch: int,
    train_epochs: int,
    unet_kern_size: int,
    unet_n_depth: int,
    train_learning_rate: float,
    train_loss: Optional[str],
    probabilistic: bool,
    axes: str,
    X_shape,
    Y_shape,
    X_val_shape,
    Y_val_shape,
    n_channel_in: int,
    n_channel_out: int,
    normalization_enabled: bool,
    normalization_pmin: Optional[float],
    normalization_pmax: Optional[float],
    normalization_eps: Optional[float],
):
    model_output_dir = (model_dir / model_name).resolve()

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "user": getpass.getuser(),
        "hostname": socket.gethostname(),
        "paths": {
            "CARE_ROOT": str(care_root),
            "PATCH_DIR": str(patch_dir),
            "PATCH_FILE": str(patch_file),
            "MODEL_DIR": str(model_dir),
            "MODEL_OUTPUT_DIR": str(model_output_dir),
        },
        "training_target": {
            "PATCH_FILE_NAME": patch_file.name,
            "MODEL_NAME": model_name,
            "MODEL_CLASS": "CARE",
            "BEST_WEIGHTS_FILE": str(model_output_dir / "weights_best.h5"),
            "LAST_WEIGHTS_FILE": str(model_output_dir / "weights_last.h5"),
            "CONFIG_FILE": str(model_output_dir / "config.json"),
            "METADATA_FILE": str(model_output_dir / "training_metadata.json"),
        },
        "training_parameters": {
            "VALIDATION_SPLIT": validation_split,
            "TRAIN_BATCH_SIZE": train_batch_size,
            "TRAIN_STEPS_PER_EPOCH": train_steps_per_epoch,
            "TRAIN_EPOCHS": train_epochs,
            "UNET_KERN_SIZE": unet_kern_size,
            "UNET_N_DEPTH": unet_n_depth,
            "TRAIN_LEARNING_RATE": train_learning_rate,
            "TRAIN_LOSS": train_loss,
            "PROBABILISTIC": probabilistic,
        },
        "normalization": {
            "ENABLED": normalization_enabled,
            "PMIN": normalization_pmin,
            "PMAX": normalization_pmax,
            "EPS": normalization_eps,
        },
        "data": {
            "axes": axes,
            "X_shape": list(X_shape),
            "Y_shape": list(Y_shape),
            "X_val_shape": list(X_val_shape),
            "Y_val_shape": list(Y_val_shape),
            "n_channel_in": int(n_channel_in),
            "n_channel_out": int(n_channel_out),
        },
        "environment": {
            "tensorflow_version": tf.__version__,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pyopencl_ctx": os.environ.get("PYOPENCL_CTX"),
        },
    }

    return to_jsonable(metadata)


def save_training_metadata(
    metadata: dict,
    *,
    model_dir: Path,
    model_name: str,
    filename: str = "training_metadata.json",
) -> Path:
    out_dir = model_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_clean = to_jsonable(metadata)

    metadata_file = out_dir / filename
    with open(metadata_file, "w") as f:
        json.dump(metadata_clean, f, indent=2)

    print_info(f"Saved training metadata: {metadata_file}")
    return metadata_file