"""Streaming face-valence estimates from a ResNet50 + LSTM model.

The model is the TorchScript release of Elena Ryumina et al.'s
EMO-AffectNetModel. The CNN produces a 512-value feature vector per face;
the temporal head consumes ten consecutive features and predicts seven
expression probabilities. Those probabilities are converted to a signed
valence proxy for the live UI.

Incoming face crops and temporal features are kept in memory only. Model
checkpoints are cached locally under ~/Library/Caches/Subtext.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import io
import logging
import os
import queue
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any


logger = logging.getLogger("subtext.face_valence")

_BACKBONE_FOLDER = "https://drive.google.com/drive/folders/1Z53O_5OF3pf2Y3oEQn1hW3f4TLJFB3nH"
_LSTM_FOLDER = "https://drive.google.com/drive/folders/1IuGhp76DlusdNxM5iJzqOf44fpXk70zP"
_SEQUENCE_LENGTH = 10
_FEATURE_SIZE = 512

# Expected value of the seven-class classifier on a negative-to-positive
# display scale. Surprise is intentionally near neutral because its valence
# depends on context.
_CLASS_VALENCE = (0.0, 1.0, -0.8, 0.05, -0.75, -0.85, -1.0)


class StreamingValenceModel:
    """Keep a ten-frame feature window for each temporary face track."""

    def __init__(self) -> None:
        self.status = "not_loaded"
        self.detail: str | None = None
        self._jobs: queue.Queue[tuple[str, dict[str, Any], Any] | None] = queue.Queue(maxsize=1)
        self._worker: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._publish: Any = None
        self._session_id = ""
        self._predictor: _CNNLSTMPredictor | None = None
        self._status_lock = threading.Lock()
        self._metrics_lock = threading.Lock()
        self._metrics: dict[str, Any] = {
            "frames_submitted": 0,
            "frames_processed": 0,
            "frames_dropped_queue_full": 0,
            "frames_skipped_model_error": 0,
            "frames_with_crops": 0,
            "last_crop_count": 0,
            "last_input_at": None,
            "last_processed_at": None,
            "last_inference_duration_s": None,
            "last_scores_count": 0,
            "inference_errors": 0,
            "last_error": None,
        }

    def diagnostics_snapshot(self) -> dict[str, Any]:
        with self._metrics_lock:
            metrics = copy.deepcopy(self._metrics)
        with self._status_lock:
            state = self.status
            detail = self.detail
        return {
            **metrics,
            "state": state,
            "detail": detail,
            "worker_alive": bool(self._worker and self._worker.is_alive()),
            "queue_size": self._jobs.qsize(),
            "queue_capacity": self._jobs.maxsize,
        }

    def submit(
        self,
        message: dict[str, Any],
        *,
        session_id: str,
        publish: Any,
    ) -> None:
        """Queue the newest frame without delaying audio or cue ingestion."""
        if self.status == "error":
            with self._metrics_lock:
                self._metrics["frames_skipped_model_error"] += 1
            return
        crop_count = sum(
            isinstance(subject.get("face_crop_jpeg"), str)
            for subject in message.get("subjects", [])
            if isinstance(subject, dict)
        )
        with self._metrics_lock:
            self._metrics["frames_submitted"] += 1
            self._metrics["last_input_at"] = time.time()
            self._metrics["last_crop_count"] = crop_count
            self._metrics["frames_with_crops"] += int(crop_count > 0)
        self._loop = asyncio.get_running_loop()
        self._publish = publish
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(
                target=self._consume,
                name="subtext-face-valence",
                daemon=True,
            )
            self._worker.start()

        job = (session_id, message, publish)
        try:
            self._jobs.put_nowait(job)
        except queue.Full:
            try:
                self._jobs.get_nowait()
            except queue.Empty:
                pass
            try:
                self._jobs.put_nowait(job)
            except queue.Full:
                pass
            with self._metrics_lock:
                self._metrics["frames_dropped_queue_full"] += 1
                dropped = self._metrics["frames_dropped_queue_full"]
            if dropped == 1 or dropped % 50 == 0:
                logger.warning(
                    "Face-model queue full; dropped_stale_frames_total=%d queue_size=%d",
                    dropped,
                    self._jobs.qsize(),
                )

    def reset(self) -> None:
        """Forget sequence state at the next captured frame."""
        self._session_id = ""
        if self.status == "error":
            self.status = "not_loaded"
            self.detail = None
        if self._worker is not None and self._worker.is_alive():
            reset_job = ("", {"_reset_sequence": True}, self._publish)
            try:
                self._jobs.put_nowait(reset_job)
            except queue.Full:
                try:
                    self._jobs.get_nowait()
                    self._jobs.put_nowait(reset_job)
                except queue.Empty:
                    pass

    def close(self) -> None:
        try:
            self._jobs.put_nowait(None)
        except queue.Full:
            try:
                self._jobs.get_nowait()
                self._jobs.put_nowait(None)
            except queue.Empty:
                pass

    def _consume(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            session_id, message, publish = job
            if message.get("_reset_sequence"):
                self._session_id = ""
                if self._predictor is not None:
                    self._predictor.reset()
                continue
            if session_id != self._session_id:
                self._session_id = session_id
                if self._predictor is not None:
                    self._predictor.reset()
            try:
                if self._predictor is None:
                    self._set_status("loading", "Loading the local face model and its weights…", publish)
                    logger.info("Loading face model checkpoints from %s", _CNNLSTMPredictor._cache_root())
                    load_started = time.monotonic()
                    self._predictor = _CNNLSTMPredictor()
                    logger.info(
                        "Face model ready; device=%s load_seconds=%.2f",
                        self._predictor.device,
                        time.monotonic() - load_started,
                    )
                    self._set_status("ready", "", publish)
                inference_started = time.monotonic()
                scores = self._predictor.add_frame(message.get("subjects", []))
                elapsed = time.monotonic() - inference_started
                with self._metrics_lock:
                    self._metrics["frames_processed"] += 1
                    self._metrics["last_processed_at"] = time.time()
                    self._metrics["last_inference_duration_s"] = round(elapsed, 4)
                    self._metrics["last_scores_count"] = sum(
                        score.get("valence") is not None for score in scores
                    )
                    self._metrics["last_error"] = None
                    processed = self._metrics["frames_processed"]
                if processed == 1 or processed % 50 == 0:
                    logger.info(
                        "Face inference progress; processed_frames=%d crops=%d scores=%d elapsed_s=%.3f",
                        processed,
                        sum(
                            isinstance(subject.get("face_crop_jpeg"), str)
                            for subject in message.get("subjects", [])
                            if isinstance(subject, dict)
                        ),
                        self._metrics["last_scores_count"],
                        elapsed,
                    )
                self._emit(
                    {
                        "type": "valence_update",
                        "timestamp": float(message.get("timestamp", time.time())),
                        "scores": scores,
                    },
                    publish,
                )
            except Exception as exc:
                logger.exception("Face-valence inference failed")
                with self._metrics_lock:
                    self._metrics["inference_errors"] += 1
                    self._metrics["last_error"] = str(exc)
                self._predictor = None
                self._set_status(
                    "error",
                    f"Face model unavailable: {exc}",
                    publish,
                )

    def _set_status(self, state: str, detail: str, publish: Any) -> None:
        with self._status_lock:
            previous = self.status
            self.status = state
            self.detail = detail or None
        if state == "error":
            logger.error("Face-model state changed; %s -> %s detail=%s", previous, state, detail)
        elif previous != state:
            logger.info("Face-model state changed; %s -> %s", previous, state)
        self._emit(
            {"type": "valence_status", "state": state, "detail": detail},
            publish,
        )

    def _emit(self, event: dict[str, Any], publish: Any) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return

        def schedule() -> None:
            asyncio.create_task(publish(event))

        loop.call_soon_threadsafe(schedule)


class _CNNLSTMPredictor:
    def __init__(self) -> None:
        try:
            import numpy as np
            import torch
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Install the local model packages by rebuilding the Subtext Python environment."
            ) from exc

        self.np = np
        self.torch = torch
        self.Image = Image
        self.device = self._device_for(torch)
        self.backbone, self.temporal = self._load_models(torch)
        self.features: dict[str, deque[Any]] = {}
        self.smoothed_scores: dict[str, float] = {}

    @staticmethod
    def _device_for(torch: Any) -> Any:
        requested = os.environ.get("SUBTEXT_FACE_MODEL_DEVICE", "auto").lower()
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "mps":
            return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        if requested == "auto" and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _cache_root() -> Path:
        override = os.environ.get("SUBTEXT_FACE_MODEL_CACHE")
        if override:
            return Path(override).expanduser()
        return Path.home() / "Library" / "Caches" / "Subtext" / "AffectNetLSTM"

    def _load_models(self, torch: Any) -> tuple[Any, Any]:
        root = self._cache_root()
        backbone_dir = root / "backbone"
        lstm_dir = root / "lstm"
        backbone_override = os.environ.get("SUBTEXT_FACE_BACKBONE_PATH")
        lstm_override = os.environ.get("SUBTEXT_FACE_LSTM_PATH")
        backbone_files = (
            [Path(backbone_override).expanduser()]
            if backbone_override
            else self._torchscript_files(backbone_dir)
        )
        lstm_files = (
            [Path(lstm_override).expanduser()]
            if lstm_override
            else self._torchscript_files(lstm_dir)
        )
        if not backbone_files or not lstm_files:
            self._download_model_folders(root)
            backbone_files = backbone_files or self._torchscript_files(backbone_dir)
            lstm_files = lstm_files or self._torchscript_files(lstm_dir)
        if not backbone_files or not lstm_files:
            raise RuntimeError(
                "The public ResNet50/LSTM checkpoints were not found after download. "
                "See the model setup notes in README.md."
            )

        backbone = self._find_module(
            torch,
            backbone_files,
            lambda module: self._has_output_size(
                module,
                torch.zeros((1, 3, 224, 224)),
                _FEATURE_SIZE,
                method="extract_features",
            ),
            "512-value CNN feature extractor",
        )

        # Prefer the authors' cross-corpus checkpoint evaluated on Aff-Wild2.
        lstm_files.sort(
            key=lambda path: (
                0 if "affwild" in path.name.lower().replace("-", "") else 1,
                path.name.lower(),
            )
        )
        temporal = self._find_module(
            torch,
            lstm_files,
            lambda module: self._has_output_size(
                module,
                torch.zeros((1, _SEQUENCE_LENGTH, _FEATURE_SIZE)),
                7,
            ),
            "ten-frame, seven-class LSTM",
        )
        return backbone.eval(), temporal.eval()

    def _download_model_folders(self, root: Path) -> None:
        try:
            import gdown
        except ImportError as exc:
            raise RuntimeError(
                "The model weights are missing and gdown is not installed. Rebuild the Python environment."
            ) from exc
        root.mkdir(parents=True, exist_ok=True)
        for folder_name, url in (("backbone", _BACKBONE_FOLDER), ("lstm", _LSTM_FOLDER)):
            target = root / folder_name
            target.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading %s model weights into %s", folder_name, target)
            downloaded = gdown.download_folder(
                url=url,
                output=str(target),
                quiet=True,
                remaining_ok=True,
            )
            if not downloaded:
                raise RuntimeError(
                    f"Could not download the public {folder_name} checkpoint folder. Check your internet connection."
                )

    @staticmethod
    def _torchscript_files(directory: Path) -> list[Path]:
        if not directory.exists():
            return []
        allowed = {".pt", ".pth", ".ts", ".torchscript"}
        return sorted(path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in allowed)

    def _find_module(
        self,
        torch: Any,
        files: list[Path],
        predicate: Any,
        description: str,
    ) -> Any:
        failures: list[str] = []
        for path in files:
            try:
                module = torch.jit.load(str(path), map_location="cpu").eval()
                if predicate(module):
                    logger.info("Loaded %s from %s", description, path)
                    return module.to(self.device)
            except Exception as exc:
                failures.append(f"{path.name}: {exc}")
        detail = "; ".join(failures[:3])
        raise RuntimeError(f"Could not locate the {description} in the downloaded model files. {detail}")

    def _output_shape(
        self,
        module: Any,
        values: Any,
        *,
        method: str | None = None,
    ) -> tuple[int, ...] | None:
        try:
            with self.torch.inference_mode():
                output = self._first_tensor(
                    getattr(module, method)(values) if method else module(values)
                )
            return tuple(int(value) for value in output.shape)
        except Exception:
            return None

    def _has_output_size(
        self,
        module: Any,
        values: Any,
        output_size: int,
        *,
        method: str | None = None,
    ) -> bool:
        shape = self._output_shape(module, values, method=method)
        return bool(shape and shape[0] == 1 and self.np.prod(shape[1:]) == output_size)

    @staticmethod
    def _first_tensor(value: Any) -> Any:
        if isinstance(value, (tuple, list)):
            if not value:
                raise ValueError("Model returned an empty sequence")
            return _CNNLSTMPredictor._first_tensor(value[0])
        if isinstance(value, dict):
            for item in value.values():
                if hasattr(item, "shape"):
                    return item
            raise ValueError("Model returned no tensor")
        if not hasattr(value, "shape"):
            raise ValueError("Model did not return a tensor")
        return value

    def reset(self) -> None:
        self.features.clear()
        self.smoothed_scores.clear()

    def add_frame(self, subjects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not subjects:
            self.reset()
            return []

        active_tracks = {
            str(subject.get("track_id") or "window_face_1")
            for subject in subjects
            if isinstance(subject.get("face_crop_jpeg"), str)
        }
        for stale_track in set(self.features) - active_tracks:
            self.features.pop(stale_track, None)
            self.smoothed_scores.pop(stale_track, None)

        results: list[dict[str, Any]] = []
        for subject in subjects[:8]:
            track_id = str(subject.get("track_id") or "window_face_1")
            encoded = subject.get("face_crop_jpeg")
            if not isinstance(encoded, str) or not encoded:
                continue
            image_tensor = self._decode_face(encoded)
            with self.torch.inference_mode():
                feature = self._first_tensor(
                    self.backbone.extract_features(image_tensor.to(self.device))
                )
                feature = feature.reshape(1, -1)
            if feature.shape[-1] != _FEATURE_SIZE:
                raise RuntimeError(
                    f"CNN feature model returned {feature.shape[-1]} values; expected {_FEATURE_SIZE}."
                )

            history = self.features.setdefault(
                track_id,
                deque(maxlen=_SEQUENCE_LENGTH),
            )
            history.append(feature[0].detach())
            if len(history) < _SEQUENCE_LENGTH:
                results.append(
                    {
                        "track_id": track_id,
                        "valence": None,
                        "frames_seen": len(history),
                    }
                )
                continue

            sequence = self.torch.stack(tuple(history)).unsqueeze(0).to(self.device)
            with self.torch.inference_mode():
                output = self._first_tensor(self.temporal(sequence))
                probabilities = output.reshape(-1, 7)[-1]
                if (probabilities < 0).any() or not self.torch.isclose(
                    probabilities.sum(), self.torch.tensor(1.0, device=probabilities.device), atol=0.05
                ):
                    probabilities = self.torch.softmax(probabilities, dim=-1)
                probabilities = probabilities / probabilities.sum().clamp_min(1e-8)
                class_valence = self.torch.tensor(
                    _CLASS_VALENCE,
                    dtype=probabilities.dtype,
                    device=probabilities.device,
                )
                raw_score = float((probabilities * class_valence).sum().clamp(-1.0, 1.0).item())

            previous = self.smoothed_scores.get(track_id, raw_score)
            score = previous * 0.25 + raw_score * 0.75
            self.smoothed_scores[track_id] = score
            results.append(
                {
                    "track_id": track_id,
                    "valence": round(score, 4),
                    "frames_seen": _SEQUENCE_LENGTH,
                }
            )
        return results

    def _decode_face(self, encoded: str) -> Any:
        from PIL import Image

        try:
            image_bytes = base64.b64decode(encoded, validate=True)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
        except Exception as exc:
            raise RuntimeError(f"Invalid face crop: {exc}") from exc

        # The released ResNet50 was trained with the VGGFace preprocessing:
        # RGB input is reversed to BGR and the channel means are removed.
        values = self.np.asarray(image, dtype=self.np.float32)[:, :, ::-1].copy()
        values -= self.np.asarray((91.4953, 103.8827, 131.0912), dtype=self.np.float32)
        return self.torch.from_numpy(values).permute(2, 0, 1).unsqueeze(0)


streaming_valence = StreamingValenceModel()
