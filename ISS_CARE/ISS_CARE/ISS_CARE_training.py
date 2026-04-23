from __future__ import annotations

import json
import getpass
import os
import socket
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
from csbdeep.io import load_training_data
from csbdeep.models import CARE, Config
from csbdeep.utils import axes_dict, plot_history, plot_some


def get_free_gpu() -> Optional[int]:
    """
    Return the GPU index with the lowest currently used memory.

    Returns
    -------
    int or None
        GPU index if available, otherwise None.

    Notes
    -----
    This uses `nvidia-smi` and is meant for shared GPU servers where we want
    to pick the least busy device automatically.
    """
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"]
        )
        memory_used = [int(x) for x in result.decode("utf-8").strip().split("\n")]
        if not memory_used:
            return None
        return memory_used.index(min(memory_used))
    except Exception:
        return None


def select_gpu_with_lowest_memory() -> Optional[int]:
    """
    Restrict TensorFlow visibility to the GPU with the lowest current memory usage.

    Returns
    -------
    int or None
        Physical GPU index selected from the machine-wide GPU list.
    """
    gpu_id = get_free_gpu()
    if gpu_id is None:
        print("No GPU selected with nvidia-smi. Training may run on CPU.")
        return None

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Selected physical GPU {gpu_id} via nvidia-smi.")

    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow sees {len(gpus)} visible GPU(s) after CUDA_VISIBLE_DEVICES masking.")
    for i, gpu in enumerate(gpus):
        print(
            f"  TensorFlow visible GPU {i}: {gpu} "
            f"(corresponds to physical GPU {gpu_id})"
        )

    if not gpus:
        print("WARNING: TensorFlow does not see any GPUs after selection.")

    return gpu_id


def configure_tensorflow_memory_growth():
    """
    Enable TensorFlow memory growth on all currently visible GPUs.

    Why this helps
    --------------
    Without memory growth, TensorFlow often grabs most GPU memory up front.
    On shared systems this is inconvenient and can cause conflicts.

    Notes
    -----
    This function also uses a lazy TensorFlow import. It should be called
    after `select_gpu_with_lowest_memory()` and before model creation/training.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"WARNING: Could not set memory growth: {e}")

    print("TensorFlow:", tf.__version__)
    print("Visible GPUs:", len(gpus))

    if not gpus:
        print("WARNING: No GPU detected — training will run on CPU.")

    return gpus


def find_patch_dir(
    care_root: Path,
    patch_dirname_candidates: Sequence[str] = ("train_patches", "train patches"),
) -> Path:
    """
    Find the patch directory under `care_root`.
    """
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
    """
    Resolve patch directory and patch file.
    """
    patch_dir = find_patch_dir(care_root, patch_dirname_candidates)
    patch_dir.mkdir(parents=True, exist_ok=True)

    if patch_file_name is not None:
        patch_file = patch_dir / patch_file_name
        if not patch_file.exists():
            raise FileNotFoundError(f"Patch file not found: {patch_file}")
        return patch_dir, patch_file

    patch_files = sorted(patch_dir.glob("*.npz"))
    if not patch_files:
        raise FileNotFoundError(f"No .npz patch files found in {patch_dir}")

    if len(patch_files) == 1:
        return patch_dir, patch_files[0]

    merged_non_dapi = [
        p for p in patch_files
        if "NON_DAPI" in p.name and ("ALL_SAMPLES" in p.name or "merged" in p.name.lower())
    ]
    if len(merged_non_dapi) == 1:
        return patch_dir, merged_non_dapi[0]

    raise FileNotFoundError(
        f"Multiple patch files found in {patch_dir}. "
        "Please specify patch_file_name explicitly."
    )


def summarize_loaded_patches(X: np.ndarray, Y: np.ndarray, name: str = "training"):
    """
    Print basic sanity checks for loaded patch arrays.
    """
    print(f"\nPatch summary ({name}):")
    print("  X shape:", X.shape)
    print("  Y shape:", Y.shape)

    print("  X has NaN:", bool(np.isnan(X).any()))
    print("  Y has NaN:", bool(np.isnan(Y).any()))
    print("  X has inf:", bool(np.isinf(X).any()))
    print("  Y has inf:", bool(np.isinf(Y).any()))

    print("  X min/max:", float(np.min(X)), float(np.max(X)))
    print("  Y min/max:", float(np.min(Y)), float(np.max(Y)))

    x_std = np.std(X.astype(np.float64), axis=(1, 2, 3))
    y_std = np.std(Y.astype(np.float64), axis=(1, 2, 3))

    n_const_x = int(np.sum(x_std <= 1e-6))
    n_const_y = int(np.sum(y_std <= 1e-6))

    print("  constant/near-constant X patches:", n_const_x)
    print("  constant/near-constant Y patches:", n_const_y)


def validate_loaded_patches(X: np.ndarray, Y: np.ndarray, name: str = "training"):
    """
    Raise an error if loaded patches contain obviously bad values.
    """
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
    """
    Load CARE training data from a `.npz` patch file.
    """
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

    print("PATCH_FILE:", patch_file)
    print("Axes:", axes)
    print("Training X:", X.shape)
    print("Training Y:", Y.shape)
    print("Validation X:", X_val.shape)
    print("Validation Y:", Y_val.shape)
    print("Input channels:", n_channel_in)
    print("Output channels:", n_channel_out)
    print("Validation split:", validation_split)

    summarize_loaded_patches(X, Y, name="training")
    summarize_loaded_patches(X_val, Y_val, name="validation")

    validate_loaded_patches(X, Y, name="training")
    validate_loaded_patches(X_val, Y_val, name="validation")

    return X, Y, X_val, Y_val, axes, n_channel_in, n_channel_out


def normalize_percentile(
    arr: np.ndarray,
    pmin: float = 1.0,
    pmax: float = 99.8,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Percentile-normalize a single patch/image to [0, 1].
    """
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
    """
    Optionally percentile-normalize training and validation patches.

    Notes
    -----
    This is applied in the training notebook/script rather than patch generation
    so users can change the normalization strategy without regenerating patches.
    """
    print("\n" + "=" * 60)
    print("Normalization settings")
    print("=" * 60)
    print("Enabled:", enabled)

    if not enabled:
        print("Percentile normalization is OFF. Raw loaded patch values will be used.")
        print("=" * 60 + "\n")
        return X, Y, X_val, Y_val

    print("Applying percentile normalization to training and validation data")
    print(f"pmin = {pmin}")
    print(f"pmax = {pmax}")
    print(f"eps = {eps}")

    print("\nBefore normalization:")
    print(f"  X:     min={X.min():.6g}, max={X.max():.6g}")
    print(f"  Y:     min={Y.min():.6g}, max={Y.max():.6g}")
    print(f"  X_val: min={X_val.min():.6g}, max={X_val.max():.6g}")
    print(f"  Y_val: min={Y_val.min():.6g}, max={Y_val.max():.6g}")

    X = np.stack([normalize_percentile(x, pmin=pmin, pmax=pmax, eps=eps) for x in X], axis=0)
    Y = np.stack([normalize_percentile(y, pmin=pmin, pmax=pmax, eps=eps) for y in Y], axis=0)
    X_val = np.stack([normalize_percentile(x, pmin=pmin, pmax=pmax, eps=eps) for x in X_val], axis=0)
    Y_val = np.stack([normalize_percentile(y, pmin=pmin, pmax=pmax, eps=eps) for y in Y_val], axis=0)

    print("\nAfter normalization:")
    print(f"  X:     min={X.min():.6g}, max={X.max():.6g}")
    print(f"  Y:     min={Y.min():.6g}, max={Y.max():.6g}")
    print(f"  X_val: min={X_val.min():.6g}, max={X_val.max():.6g}")
    print(f"  Y_val: min={Y_val.min():.6g}, max={Y_val.max():.6g}")
    print("=" * 60 + "\n")

    return X, Y, X_val, Y_val


def plot_patch_examples(
    X: np.ndarray,
    Y: np.ndarray,
    n_show: int = 5,
    title: str = "Example patch pairs (top: input, bottom: target)",
):
    """
    Plot a few input/target patch pairs.
    """
    n_show = min(n_show, len(X))
    plt.figure(figsize=(12, 5))
    plot_some(X[:n_show], Y[:n_show])
    plt.suptitle(title)
    plt.show()


def build_care_config(
    *,
    axes: str,
    n_channel_in: int,
    n_channel_out: int,
    train_batch_size: int,
    train_steps_per_epoch: int,
    train_epochs: int,
    unet_kern_size: int = 3,
    train_learning_rate: float = 2e-4,
    train_reduce_lr: Optional[dict] = None,
    probabilistic: bool = False,
):
    """
    Build a csbdeep CARE Config object.
    """
    if train_reduce_lr is None:
        train_reduce_lr = {"factor": 0.5, "patience": 10}

    config = Config(
        axes=axes,
        n_channel_in=n_channel_in,
        n_channel_out=n_channel_out,
        probabilistic=probabilistic,
        unet_kern_size=unet_kern_size,
        train_batch_size=train_batch_size,
        train_steps_per_epoch=train_steps_per_epoch,
        train_epochs=train_epochs,
        train_learning_rate=train_learning_rate,
        train_reduce_lr=train_reduce_lr,
    )
    return config


def create_care_model(
    *,
    config: Config,
    model_name: str,
    model_dir: Path,
) -> CARE:
    """
    Create a CARE model instance.
    """
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
    """
    Print where the model and its training artifacts are expected to be saved.
    """
    out_dir = model_dir / model_name

    print("\n" + "=" * 60)
    print("Model output locations")
    print("=" * 60)
    print("Model output directory:", out_dir)
    print("CARE saves training artifacts in this directory.")
    print("Common files include:")
    print("  - weights_best.h5        (best validation checkpoint)")
    print("  - weights_last.h5        (last epoch checkpoint, depending on version/setup)")
    print("  - config.json            (model/config description)")
    print("  - training_metadata.json (run metadata from this script)")
    print("\nAt the end of training, CARE usually restores the best checkpoint")
    print("when it prints: Loading network weights from 'weights_best.h5'.")
    print("=" * 60 + "\n")


def train_care_model(
    *,
    model: CARE,
    X: np.ndarray,
    Y: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
):
    """
    Train the CARE model.
    """
    try:
        history = model.train(X, Y, validation_data=(X_val, Y_val))
    except Exception:
        print("Training failed.")
        print(
            "This is often caused by bad patches "
            "(NaN, inf, constant patches, or extreme values)."
        )
        raise
    return history


def plot_training_curves(history):
    """
    Plot training history from CARE training.
    """
    keys = sorted(history.history.keys())
    print("Available metrics:", keys)

    loss_keys = [k for k in ["loss", "val_loss"] if k in history.history]
    metric_keys = [k for k in ["mse", "val_mse", "mae", "val_mae"] if k in history.history]

    plt.figure(figsize=(12, 4))
    plot_history(history, loss_keys, metric_keys)
    plt.tight_layout()
    plt.show()


def predict_on_validation_examples(
    *,
    model: CARE,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    probabilistic: bool,
    n_examples: int = 5,
):
    """
    Run a quick prediction check on validation patches.
    """
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

    return X_example, Y_example, Y_pred


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
    train_learning_rate: float,
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
    """
    Build metadata dictionary for a CARE training run.
    """
    import tensorflow as tf

    model_output_dir = (model_dir / model_name).resolve()

    return {
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
            "NOTES": (
                "During/after training, CARE typically uses weights_best.h5 as the "
                "best validation checkpoint. Depending on version/setup, "
                "weights_last.h5 may also be present."
            ),
        },
        "training_parameters": {
            "VALIDATION_SPLIT": validation_split,
            "TRAIN_BATCH_SIZE": train_batch_size,
            "TRAIN_STEPS_PER_EPOCH": train_steps_per_epoch,
            "TRAIN_EPOCHS": train_epochs,
            "UNET_KERN_SIZE": unet_kern_size,
            "TRAIN_LEARNING_RATE": train_learning_rate,
            "PROBABILISTIC": probabilistic,
        },
        "normalization": {
            "ENABLED": normalization_enabled,
            "METHOD": "per-patch percentile normalization" if normalization_enabled else None,
            "PMIN": normalization_pmin,
            "PMAX": normalization_pmax,
            "EPS": normalization_eps,
            "APPLIED_TO": ["X", "Y", "X_val", "Y_val"] if normalization_enabled else [],
            "NOTE": (
                "If normalization is enabled here, inference inputs should be "
                "normalized the same way before prediction."
            ),
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
        },
    }


def save_training_metadata(
    metadata: dict,
    *,
    model_dir: Path,
    model_name: str,
    filename: str = "training_metadata.json",
) -> Path:
    """
    Save training metadata next to the trained model.
    """
    out_dir = model_dir / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata_file = out_dir / filename
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print("Saved training metadata:", metadata_file)
    return metadata_file