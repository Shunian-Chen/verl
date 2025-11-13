# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A unified tracking interface that supports logging data to different backend
"""

import base64
import dataclasses
import os
from enum import Enum
from functools import partial
from io import BytesIO
from pathlib import Path
from typing import Any


class Tracking:
    """A unified tracking interface for logging experiment data to multiple backends.

    This class provides a centralized way to log experiment metrics, parameters, and artifacts
    to various tracking backends including WandB, MLflow, SwanLab, TensorBoard, and console.

    Attributes:
        supported_backend: List of supported tracking backends.
        logger: Dictionary of initialized logger instances for each backend.
    """

    supported_backend = [
        "wandb",
        "mlflow",
        "swanlab",
        "vemlp_wandb",
        "tensorboard",
        "console",
        "clearml",
        "trackio",
        "localfile",
    ]

    def __init__(self, project_name, experiment_name, default_backend: str | list[str] = "console", config=None):
        if isinstance(default_backend, str):
            default_backend = [default_backend]
        for backend in default_backend:
            if backend == "tracking":
                import warnings

                warnings.warn("`tracking` logger is deprecated. use `wandb` instead.", DeprecationWarning, stacklevel=2)
            else:
                assert backend in self.supported_backend, f"{backend} is not supported"

        self.logger = {}

        if "tracking" in default_backend or "wandb" in default_backend:
            import wandb

            settings = None
            if config and config["trainer"].get("wandb_proxy", None):
                settings = wandb.Settings(https_proxy=config["trainer"]["wandb_proxy"])
            wandb.init(project=project_name, name=experiment_name, config=config, settings=settings)
            self.logger["wandb"] = wandb

        if "trackio" in default_backend:
            import trackio

            trackio.init(project=project_name, name=experiment_name, config=config)
            self.logger["trackio"] = trackio

        if "mlflow" in default_backend:
            import os

            import mlflow

            MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:////tmp/mlruns.db")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

            # Project_name is actually experiment_name in MLFlow
            # If experiment does not exist, will create a new experiment
            experiment = mlflow.set_experiment(project_name)
            mlflow.start_run(experiment_id=experiment.experiment_id, run_name=experiment_name)
            mlflow.log_params(_compute_mlflow_params_from_objects(config))
            self.logger["mlflow"] = _MlflowLoggingAdapter()

        if "swanlab" in default_backend:
            import os

            import swanlab

            SWANLAB_API_KEY = os.environ.get("SWANLAB_API_KEY", None)
            SWANLAB_LOG_DIR = os.environ.get("SWANLAB_LOG_DIR", "swanlog")
            SWANLAB_MODE = os.environ.get("SWANLAB_MODE", "cloud")
            if SWANLAB_API_KEY:
                swanlab.login(SWANLAB_API_KEY)  # NOTE: previous login information will be overwritten

            if config is None:
                config = {}  # make sure config is not None, otherwise **config will raise error
            swanlab.init(
                project=project_name,
                experiment_name=experiment_name,
                config={"FRAMEWORK": "verl", **config},
                logdir=SWANLAB_LOG_DIR,
                mode=SWANLAB_MODE,
            )
            self.logger["swanlab"] = swanlab

        if "vemlp_wandb" in default_backend:
            import os

            import volcengine_ml_platform
            from volcengine_ml_platform import wandb as vemlp_wandb

            volcengine_ml_platform.init(
                ak=os.environ["VOLC_ACCESS_KEY_ID"],
                sk=os.environ["VOLC_SECRET_ACCESS_KEY"],
                region=os.environ["MLP_TRACKING_REGION"],
            )

            vemlp_wandb.init(
                project=project_name,
                name=experiment_name,
                config=config,
                sync_tensorboard=True,
            )
            self.logger["vemlp_wandb"] = vemlp_wandb

        if "tensorboard" in default_backend:
            self.logger["tensorboard"] = _TensorboardAdapter(project_name, experiment_name)

        if "console" in default_backend:
            from verl.utils.logger import LocalLogger

            self.console_logger = LocalLogger(print_to_console=True)
            self.logger["console"] = self.console_logger

        if "clearml" in default_backend:
            self.logger["clearml"] = ClearMLLogger(project_name, experiment_name, config)

        if "localfile" in default_backend:
            self.logger["localfile"] = _LocalFileAdapter(project_name, experiment_name)

    def log(self, data, step, backend=None):
        for default_backend, logger_instance in self.logger.items():
            if backend is None or default_backend in backend:
                logger_instance.log(data=data, step=step)

    def __del__(self):
        if "wandb" in self.logger:
            self.logger["wandb"].finish(exit_code=0)
        if "swanlab" in self.logger:
            self.logger["swanlab"].finish()
        if "vemlp_wandb" in self.logger:
            self.logger["vemlp_wandb"].finish(exit_code=0)
        if "tensorboard" in self.logger:
            self.logger["tensorboard"].finish()
        if "clearnml" in self.logger:
            self.logger["clearnml"].finish()
        if "trackio" in self.logger:
            self.logger["trackio"].finish()


class ClearMLLogger:
    def __init__(self, project_name: str, experiment_name: str, config):
        self.project_name = project_name
        self.experiment_name = experiment_name

        import clearml

        self._task: clearml.Task = clearml.Task.init(
            task_name=experiment_name,
            project_name=project_name,
            continue_last_task=True,
            output_uri=False,
        )

        self._task.connect_configuration(config, name="Hyperparameters")

    def _get_logger(self):
        return self._task.get_logger()

    def log(self, data, step):
        import numpy as np
        import pandas as pd

        # logs = self._rewrite_logs(data)
        logger = self._get_logger()
        for k, v in data.items():
            title, series = k.split("/", 1)

            if isinstance(v, int | float | np.floating | np.integer):
                logger.report_scalar(
                    title=title,
                    series=series,
                    value=v,
                    iteration=step,
                )
            elif isinstance(v, pd.DataFrame):
                logger.report_table(
                    title=title,
                    series=series,
                    table_plot=v,
                    iteration=step,
                )
            else:
                logger.warning(
                    f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}". This '
                    f"invocation of ClearML logger's function is incorrect so this attribute was dropped. "
                )

    def finish(self):
        self._task.mark_completed()


class _TensorboardAdapter:
    def __init__(self, project_name, experiment_name):
        import os

        from torch.utils.tensorboard import SummaryWriter

        tensorboard_dir = os.environ.get("TENSORBOARD_DIR", f"tensorboard_log/{project_name}/{experiment_name}")
        os.makedirs(tensorboard_dir, exist_ok=True)
        print(f"Saving tensorboard log to {tensorboard_dir}.")
        self.writer = SummaryWriter(tensorboard_dir)

    def log(self, data, step):
        for key in data:
            self.writer.add_scalar(key, data[key], step)

    def finish(self):
        self.writer.close()


class _MlflowLoggingAdapter:
    def log(self, data, step):
        import mlflow

        results = {k.replace("@", "_at_"): v for k, v in data.items()}
        mlflow.log_metrics(metrics=results, step=step)


class _LocalFileAdapter:
    """Append metrics to a local JSONL file for simple offline visualization.

    Environment variables:
        LOCAL_METRICS_DIR: base directory. Default: logs/verl/local_metrics
    Directory layout:
        <base>/<project_name>/<experiment_name>/metrics.jsonl
    Each line is a JSON object with keys: {"step": int, <metric>: number, ...}
    """

    def __init__(self, project_name: str, experiment_name: str):
        import json

        self._json = json
        base_dir = os.environ.get("LOCAL_METRICS_DIR", os.path.join("logs", "verl", "local_metrics"))
        # Use safe subpaths
        safe_project = str(project_name) if project_name is not None else "default_project"
        safe_experiment = str(experiment_name) if experiment_name is not None else "default_experiment"
        self.output_dir = os.path.join(base_dir, safe_project, safe_experiment)
        os.makedirs(self.output_dir, exist_ok=True)
        self.jsonl_path = os.path.join(self.output_dir, "metrics.jsonl")
        # Touch file to ensure it exists
        open(self.jsonl_path, "a").close()

    def _is_number(self, v):
        try:
            import numpy as _np  # local import to avoid hard dependency at module import time

            return isinstance(v, (int, float, _np.floating, _np.integer))
        except Exception:
            return isinstance(v, (int, float))

    def log(self, data, step):
        # Keep only scalar numeric values for simplicity
        numeric_data = {k: v for k, v in data.items() if self._is_number(v)}
        record = {"step": int(step) if step is not None else None}
        # Replace characters that are problematic in some tools (keep as-is in JSON)
        record.update(numeric_data)
        try:
            with open(self.jsonl_path, "a", encoding="utf-8") as f:
                f.write(self._json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"WARNING: failed to write local metrics to {self.jsonl_path}: {e}")


def _compute_mlflow_params_from_objects(params) -> dict[str, Any]:
    if params is None:
        return {}

    return _flatten_dict(_transform_params_to_json_serializable(params, convert_list_to_dict=True), sep="/")


def _transform_params_to_json_serializable(x, convert_list_to_dict: bool):
    _transform = partial(_transform_params_to_json_serializable, convert_list_to_dict=convert_list_to_dict)

    if dataclasses.is_dataclass(x):
        return _transform(dataclasses.asdict(x))
    if isinstance(x, dict):
        return {k: _transform(v) for k, v in x.items()}
    if isinstance(x, list):
        if convert_list_to_dict:
            return {"list_len": len(x)} | {f"{i}": _transform(v) for i, v in enumerate(x)}
        else:
            return [_transform(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value

    return x


def _flatten_dict(raw: dict[str, Any], *, sep: str) -> dict[str, Any]:
    import pandas as pd

    ans = pd.json_normalize(raw, sep=sep).to_dict(orient="records")[0]
    assert isinstance(ans, dict)
    return ans


@dataclasses.dataclass
class ValidationGenerationsLogger:
    project_name: str = None
    experiment_name: str = None

    def log(self, loggers, samples, step):
        if "wandb" in loggers:
            self.log_generations_to_wandb(samples, step)
        if "swanlab" in loggers:
            self.log_generations_to_swanlab(samples, step)
        if "mlflow" in loggers:
            self.log_generations_to_mlflow(samples, step)

        if "clearml" in loggers:
            self.log_generations_to_clearml(samples, step)
        if "tensorboard" in loggers:
            self.log_generations_to_tensorboard(samples, step)

        if "vemlp_wandb" in loggers:
            self.log_generations_to_vemlp_wandb(samples, step)

    def log_generations_to_vemlp_wandb(self, samples, step):
        from volcengine_ml_platform import wandb as vemlp_wandb

        self._log_generations_to_wandb(samples, step, vemlp_wandb)

    def log_generations_to_wandb(self, samples, step):
        import wandb

        self._log_generations_to_wandb(samples, step, wandb)

    def _log_generations_to_wandb(self, samples, step, wandb):
        """将验证样本记录到 WandB，支持文本与图像。"""

        columns = ["step", "input", "output", "score"]
        has_images = any(len(sample) >= 4 and sample[3] is not None for sample in samples)
        if has_images:
            columns.append("image")

        if not hasattr(self, "validation_table") or getattr(self.validation_table, "columns", []) != columns:
            self.validation_table = wandb.Table(columns=columns)

        new_table = wandb.Table(columns=columns, data=list(getattr(self.validation_table, "data", [])))

        for idx, sample in enumerate(samples):
            input_text, output_text, score = sample[0], sample[1], sample[2]
            row = [step, input_text, output_text, score]
            if has_images:
                image_entry = sample[3] if len(sample) >= 4 else None
                row.append(self._prepare_wandb_image(image_entry, wandb, caption=f"sample_{idx}"))
            new_table.add_data(*row)

        wandb.log({"val/generations": new_table}, step=step)
        self.validation_table = new_table

    def log_generations_to_swanlab(self, samples, step):
        """Log samples to swanlab as text and images"""
        import swanlab

        swanlab_table = swanlab.echarts.Table()

        display_headers = ["step", "input", "output", "score"]
        has_images = any(len(sample) >= 4 and sample[3] is not None for sample in samples)
        if has_images:
            display_headers.append("image")

        swanlab_row_list = []
        for idx, sample in enumerate(samples):
            input_text, output_text, score = sample[0], sample[1], sample[2]
            row = [step, input_text, output_text, score]
            if has_images:
                image_entry = sample[3] if len(sample) >= 4 else None
                row.append(self._prepare_swanlab_image(image_entry, swanlab, caption=f"sample_{idx}"))
            swanlab_row_list.append(row)

        swanlab_table.add(headers=display_headers, rows=swanlab_row_list)

        # Log to swanlab
        swanlab.log({"val/generations": swanlab_table}, step=step)

    def _prepare_swanlab_image(self, image: Any, swanlab, _caption: str | None = None):
        if image is None:
            return None

        from PIL import Image

        processed_image, _ = self._normalize_image_payload(image)

        if processed_image is None:
            return None

        if isinstance(processed_image, Image.Image):
            return swanlab.Image(processed_image)

        if isinstance(processed_image, str):
            return swanlab.Image(processed_image)

        return swanlab.Image(processed_image)

    def log_generations_to_mlflow(self, samples, step):
        """Log validation generation to mlflow as artifacts"""
        # https://mlflow.org/docs/latest/api_reference/python_api/mlflow.html?highlight=log_artifact#mlflow.log_artifact

        import json
        import tempfile

        import mlflow

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                validation_gen_step_file = Path(tmp_dir, f"val_step{step}.json")
                row_data = []
                for sample in samples:
                    data = {"input": sample[0], "output": sample[1], "score": sample[2]}
                    row_data.append(data)
                with open(validation_gen_step_file, "w") as file:
                    json.dump(row_data, file)
                mlflow.log_artifact(validation_gen_step_file)
        except Exception as e:
            print(f"WARNING: save validation generation file to mlflow failed with error {e}")

    def log_generations_to_clearml(self, samples, step):
        """Log validation generation to clearml as table"""

        import clearml
        import pandas as pd

        task: clearml.Task | None = clearml.Task.current_task()
        if task is None:
            return

        table = [
            {
                "step": step,
                "input": sample[0],
                "output": sample[1],
                "score": sample[2],
            }
            for sample in samples
        ]

        logger = task.get_logger()
        logger.report_table(
            series="Validation generations",
            title="Validation",
            table_plot=pd.DataFrame.from_records(table),
            iteration=step,
        )

    def _prepare_wandb_image(self, image: Any, wandb, caption: str | None = None):
        if image is None:
            return None

        processed_image, resolved_caption = self._normalize_image_payload(image)

        if processed_image is None:
            return None

        if isinstance(processed_image, wandb.Image):
            if caption and getattr(processed_image, "caption", None) in (None, ""):
                processed_image.caption = caption
            return processed_image

        if isinstance(processed_image, str):
            return wandb.Image(processed_image, caption=resolved_caption or caption)

        return wandb.Image(processed_image, caption=resolved_caption or caption)

    def _normalize_image_payload(self, image: Any) -> tuple[Any | None, str | None]:
        caption: str | None = None

        if isinstance(image, dict):
            caption = image.get("caption")
            if "image" in image:
                image = image["image"]
            elif "path" in image:
                image = image["path"]
            elif "bytes" in image:
                image = self._bytes_to_pil(image["bytes"])
            elif "b64_json" in image:
                image = self._decode_base64_image(image["b64_json"])
            elif "array" in image:
                image = self._array_to_pil(image["array"])
            elif "pil" in image:
                image = image["pil"]

        if hasattr(image, "to") and callable(getattr(image, "to", None)):
            image = self._tensor_to_pil(image)

        normalized = self._coerce_to_supported_image(image)
        return normalized, caption

    def _coerce_to_supported_image(self, image: Any):
        if image is None:
            return None

        from PIL import Image

        if isinstance(image, Image.Image):
            return image

        if isinstance(image, (bytes, bytearray)):
            return self._bytes_to_pil(image)

        if isinstance(image, str):
            if os.path.exists(image):
                return image
            decoded = self._decode_base64_image(image)
            if decoded is not None:
                return decoded
            return None

        if "numpy" in str(type(image)):
            return self._array_to_pil(image)

        return image

    def _tensor_to_pil(self, tensor):
        try:
            import torch
        except ImportError:
            return None

        if not isinstance(tensor, torch.Tensor):
            return None

        tensor = tensor.detach().cpu()
        if tensor.ndim == 4:
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3):
            tensor = tensor.permute(1, 2, 0)

        return self._array_to_pil(tensor.numpy())

    def _array_to_pil(self, array_like):
        try:
            import numpy as np
        except ImportError:
            return None

        from PIL import Image

        array = np.array(array_like)

        if array.ndim == 4:
            array = array[0]
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        if array.ndim == 3 and array.shape[0] in (1, 3):
            array = np.transpose(array, (1, 2, 0))

        if array.dtype != np.uint8:
            array = np.clip(array, 0, 1)
            array = (array * 255).astype(np.uint8)

        return Image.fromarray(array)

    def _bytes_to_pil(self, image_bytes: bytes | bytearray):
        from PIL import Image

        try:
            buffer = BytesIO(image_bytes)
            return Image.open(buffer).convert("RGB")
        except Exception:
            return None

    def _decode_base64_image(self, data: str):
        if not isinstance(data, str):
            return None

        try:
            if data.startswith("data:image"):
                data = data.split(",", 1)[1]
            return self._bytes_to_pil(base64.b64decode(data))
        except Exception:
            return None

    def _dict_to_pil(self, payload: dict):
        image, _ = self._normalize_image_payload(payload)
        return image

    def log_generations_to_tensorboard(self, samples, step):
        """Log samples to tensorboard as text"""
        # Initialize tensorboard writer if not exists
        if not hasattr(self, "writer"):
            from torch.utils.tensorboard import SummaryWriter

            # Use the same directory structure as _TensorboardAdapter
            if self.project_name and self.experiment_name:
                default_dir = os.path.join("tensorboard_log", self.project_name, self.experiment_name)
            else:
                default_dir = "tensorboard_log"

            tensorboard_dir = os.environ.get("TENSORBOARD_DIR", default_dir)
            os.makedirs(tensorboard_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_dir)

        # Format the samples data into readable text
        text_content = f"**Generation Results - Step {step}**\n\n"

        for i, sample in enumerate(samples):
            text_content += f"### Sample {i + 1}\n"

            # Assuming sample contains [input, output, score]
            if len(sample) >= 3:
                input_text, output_text, score = sample[0], sample[1], sample[2]

                text_content += f"**Input:** {input_text}\n\n"
                text_content += f"**Output:** {output_text}\n\n"
                text_content += f"**Score:** {score}\n\n"
            else:
                # Handle cases where sample format might be different
                text_content += f"**Data:** {sample}\n\n"

            text_content += "---\n\n"

        # Log to tensorboard as text
        self.writer.add_text("val/generations", text_content, step)
        # Flush to ensure data is written
        self.writer.flush()
