"""Predict-only transformer memory for frozen close data.

The package ships the trained checkpoint produced by the notebook.  Runtime
inference only needs the transformer architecture, the frozen date index stored
inside the checkpoint, and arithmetic to turn predicted digits back into log_px.
"""

from __future__ import annotations

import datetime as _dt
import math
from importlib import resources
from typing import Any

import torch
import torch.nn as nn


MODEL_DTYPE = torch.float32
LOG_PX_SCALE = 10_000_000_000
Y_SCALE = 1_000_000
FIXED_DIGITS = 11
FIXED_PLACE_VALUES = tuple(10**power for power in range(FIXED_DIGITS - 1, -1, -1))
DEFAULT_CHECKPOINT_NAME = "mem_digit.pt"
RANGE_BATCH_SIZE = 2048

_CHATPX_CACHE: dict[str, "ChatPX"] = {}


class DateTransformerDigitMemorizer(nn.Module):
    """Transformer encoder that memorizes fixed-point log_px digits by date.

    Input tokens are date_id, shifted year, month, and day.  The output is 11
    ten-way digit logits for the fixed-point integer round(log_px * 1e10).
    """

    def __init__(self, min_year, max_year, num_dates, d_model=256, nhead=8, num_layers=4, dropout=0.0):
        """Build date-part embeddings, transformer layers, and digit heads.

        The architecture must match the training notebook exactly so the saved
        state_dict can be loaded without the original training package.
        """
        super().__init__()
        assert d_model % nhead == 0

        self.min_year = int(min_year)
        self.max_year = int(max_year)
        self.num_dates = int(num_dates)
        self.d_model = int(d_model)
        self.model_kwargs = {
            "d_model": int(d_model),
            "nhead": int(nhead),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
        }

        n_years = self.max_year - self.min_year + 1
        self.date_embedding = nn.Embedding(num_dates, d_model)
        self.year_embedding = nn.Embedding(n_years, d_model)
        self.month_embedding = nn.Embedding(13, d_model)
        self.day_embedding = nn.Embedding(32, d_model)
        self.field_embedding = nn.Embedding(4, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.shared_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.digit_head = nn.Linear(d_model, FIXED_DIGITS * 10)

    def forward(self, x):
        """Run [date_id, yyyy_idx, mm, dd] through the model.

        The returned tensor has shape [batch, 11, 10], one ten-way classifier
        for each fixed-point digit.
        """
        x = x.long()
        date_token = self.date_embedding(x[:, 0])
        year_token = self.year_embedding(x[:, 1])
        month_token = self.month_embedding(x[:, 2])
        day_token = self.day_embedding(x[:, 3])

        tokens = torch.stack([date_token, year_token, month_token, day_token], dim=1)
        field_ids = torch.arange(4, device=x.device).unsqueeze(0)
        tokens = tokens + self.field_embedding(field_ids)

        encoded = self.encoder(tokens)
        pooled = encoded[:, 0]
        hidden = self.shared_head(pooled)
        return self.digit_head(hidden).view(-1, FIXED_DIGITS, 10)


def get_device() -> torch.device:
    """Return CUDA when this Torch build sees it, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def digits_to_fixed(digits) -> int:
    """Combine 11 predicted base-10 digits into one fixed-point integer."""
    if len(digits) != FIXED_DIGITS:
        raise ValueError(f"expected {FIXED_DIGITS} digits, got {len(digits)}")
    return int(sum(int(digit) * place for digit, place in zip(digits, FIXED_PLACE_VALUES)))


def fixed_to_log_px(fixed: int) -> float:
    """Convert the fixed integer round(log_px * 1e10) back to log_px."""
    fixed = int(fixed)
    x = fixed // Y_SCALE
    y = fixed % Y_SCALE
    return float(x * 1e-4 + y * 1e-10)


def _torch_load(path, device: torch.device) -> dict[str, Any]:
    """Load a trusted checkpoint, compatible with older and newer Torch releases."""
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _default_checkpoint_resource():
    """Return the package-data checkpoint resource."""
    return resources.files("chat_spx").joinpath("data").joinpath(DEFAULT_CHECKPOINT_NAME)


def _load_payload(device: torch.device) -> dict[str, Any]:
    """Load the bundled checkpoint from package data."""
    checkpoint = _default_checkpoint_resource()
    with resources.as_file(checkpoint) as resource_path:
        return _torch_load(resource_path, device)


def _prepare_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Validate checkpoint metadata and cache the date-to-id map in memory."""
    if int(metadata.get("fixed_digits", FIXED_DIGITS)) != FIXED_DIGITS:
        raise ValueError(f"checkpoint fixed_digits does not match {FIXED_DIGITS}")

    date_index = metadata.get("date_index")
    if not date_index:
        raise ValueError("checkpoint metadata is missing date_index")

    if "_date_to_id" not in metadata:
        metadata["_date_to_id"] = {date_key: idx for idx, date_key in enumerate(date_index)}
    return metadata


def load_model_checkpoint(device: str | torch.device | None = None):
    """Load model weights and metadata from the bundled checkpoint.

    Returns (model, metadata).  Only load checkpoints you trust because this uses
    torch.load, matching normal PyTorch checkpoint behavior.
    """
    torch_device = get_device() if device is None else torch.device(device)
    payload = _load_payload(torch_device)
    metadata = _prepare_metadata(dict(payload["metadata"]))

    model = DateTransformerDigitMemorizer(
        metadata["min_year"],
        metadata["max_year"],
        metadata["num_dates"],
        **metadata.get("model_kwargs", {}),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device=torch_device, dtype=MODEL_DTYPE)
    model.eval()
    return model, metadata


def clear_model_cache() -> None:
    """Clear loaded ChatPX instances from the process-local cache."""
    _CHATPX_CACHE.clear()


def _normalize_date_key(value) -> str:
    """Normalize supported date objects and strings to YYYY-MM-DD."""
    if isinstance(value, _dt.datetime):
        return value.date().isoformat()
    if isinstance(value, _dt.date):
        return value.isoformat()
    if isinstance(value, str):
        text = value.strip()
        try:
            return _dt.date.fromisoformat(text[:10]).isoformat()
        except ValueError as exc:
            raise ValueError(f"date must be parseable as YYYY-MM-DD, got {value!r}") from exc
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m-%d")
    raise TypeError(f"unsupported date value: {value!r}")


def _date_key(date_or_year, month=None, day=None) -> str:
    """Support either one date-like object or a yyyy, mm, dd triple."""
    if month is None and day is None:
        return _normalize_date_key(date_or_year)
    if month is None or day is None:
        raise ValueError("month and day must be provided together")
    return _dt.date(int(date_or_year), int(month), int(day)).isoformat()


def make_feature_for_date(date_or_year, metadata: dict[str, Any], month=None, day=None) -> torch.Tensor:
    """Turn one frozen trading date into [date_id, yyyy_idx, mm, dd]."""
    metadata = _prepare_metadata(metadata)
    date_key = _date_key(date_or_year, month, day)
    date_to_id = metadata["_date_to_id"]
    if date_key not in date_to_id:
        first_date = metadata["date_index"][0]
        last_date = metadata["date_index"][-1]
        raise KeyError(f"{date_key} is not in the frozen trading-date index {first_date}..{last_date}")

    parsed = _dt.date.fromisoformat(date_key)
    return torch.as_tensor(
        [[date_to_id[date_key], parsed.year - int(metadata["min_year"]), parsed.month, parsed.day]],
        dtype=torch.long,
    )


def _date_range_keys(start_date, end_date, metadata: dict[str, Any]) -> list[str]:
    """Return frozen trading-date keys between start_date and end_date, inclusive."""
    start_key = _date_key(start_date)
    end_key = _date_key(end_date)
    if start_key > end_key:
        raise ValueError(f"start_date {start_key} must be on or before end_date {end_key}")
    return [date_key for date_key in metadata["date_index"] if start_key <= date_key <= end_key]


def _make_features_for_date_keys(date_keys: list[str], metadata: dict[str, Any]) -> torch.Tensor:
    """Turn ordered date keys into model feature rows."""
    metadata = _prepare_metadata(metadata)
    min_year = int(metadata["min_year"])
    date_to_id = metadata["_date_to_id"]
    rows = []
    for date_key in date_keys:
        parsed = _dt.date.fromisoformat(date_key)
        rows.append([date_to_id[date_key], parsed.year - min_year, parsed.month, parsed.day])
    return torch.as_tensor(rows, dtype=torch.long)


@torch.no_grad()
def predict_log_px_for_date(
    model: DateTransformerDigitMemorizer,
    date_or_year,
    metadata: dict[str, Any],
    month=None,
    day=None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Predict the memorized fixed-point log_px record for one trading date."""
    torch_device = get_device() if device is None else torch.device(device)
    x = make_feature_for_date(date_or_year, metadata, month, day).to(torch_device)

    model.eval()
    model.to(device=torch_device, dtype=MODEL_DTYPE)
    logits = model(x)
    digits = [int(digit) for digit in logits.argmax(dim=2).detach().cpu()[0].tolist()]
    fixed = digits_to_fixed(digits)
    date_key = _date_key(date_or_year, month, day)
    parsed = _dt.date.fromisoformat(date_key)
    log_px = fixed_to_log_px(fixed)

    return {
        "date": date_key,
        "date_id": int(x.detach().cpu()[0, 0].item()),
        "yyyy": parsed.year,
        "mm": parsed.month,
        "dd": parsed.day,
        "pred_digits": digits,
        "pred_log_px_fixed": fixed,
        "pred_log_px_x": int(fixed // Y_SCALE),
        "pred_log_px_y": int(fixed % Y_SCALE),
        "pred_log_px": log_px,
        "pred_px": float(math.exp(log_px)),
    }


class ChatPX:
    """Reusable predictor that keeps the transformer checkpoint loaded.

    Create one instance when asking for many dates.  The model and date index are
    loaded during initialization and reused for every method call.
    """

    def __init__(self, device: str | torch.device | None = None):
        """Load the internal model checkpoint onto the requested Torch device."""
        self.device = get_device() if device is None else torch.device(device)
        self.model, self.metadata = load_model_checkpoint(self.device)

    def predict(self, date_or_year, month=None, day=None) -> dict[str, Any]:
        """Return the full predicted record for one trading date."""
        return predict_log_px_for_date(self.model, date_or_year, self.metadata, month, day, device=self.device)

    def get_log_px(self, date_or_year, month=None, day=None) -> float:
        """Return only the memorized log_px value for one trading date."""
        return float(self.predict(date_or_year, month, day)["pred_log_px"])

    def get_px(self, date_or_year, month=None, day=None) -> float:
        """Return exp(log_px), the recovered close price for one trading date."""
        return float(math.exp(self.get_log_px(date_or_year, month, day)))

    @torch.no_grad()
    def get_px_by_date_range(self, start_date, end_date) -> list[float]:
        """Return recovered close prices for frozen trading dates in an inclusive range.

        Calendar dates without memorized rows, such as weekends and holidays, are skipped.
        """
        date_keys = _date_range_keys(start_date, end_date, self.metadata)
        if not date_keys:
            return []

        features = _make_features_for_date_keys(date_keys, self.metadata)
        out = []
        self.model.eval()
        self.model.to(device=self.device, dtype=MODEL_DTYPE)

        for start in range(0, len(features), RANGE_BATCH_SIZE):
            batch = features[start : start + RANGE_BATCH_SIZE].to(self.device)
            logits = self.model(batch)
            digit_rows = logits.argmax(dim=2).detach().cpu().tolist()
            for digits in digit_rows:
                fixed = digits_to_fixed(digits)
                out.append(float(math.exp(fixed_to_log_px(fixed))))
        return out


def _default_chatpx(device: str | torch.device | None = None) -> ChatPX:
    """Return one lazily loaded ChatPX instance per Torch device."""
    torch_device = get_device() if device is None else torch.device(device)
    key = str(torch_device)
    if key not in _CHATPX_CACHE:
        _CHATPX_CACHE[key] = ChatPX(torch_device)
    return _CHATPX_CACHE[key]


def predict(date_or_year, month=None, day=None, *, device=None):
    """Return the full predicted record for one trading date."""
    return _default_chatpx(device).predict(date_or_year, month, day)


def get_log_px(date_or_year, month=None, day=None, *, device=None) -> float:
    """Return only the memorized log_px value for one trading date."""
    return _default_chatpx(device).get_log_px(date_or_year, month, day)


def get_px(date_or_year, month=None, day=None, *, device=None) -> float:
    """Return exp(log_px), the recovered close price for one trading date."""
    return _default_chatpx(device).get_px(date_or_year, month, day)


def get_px_by_date_range(start_date, end_date, *, device=None) -> list[float]:
    """Return recovered close prices for frozen trading dates in an inclusive range."""
    return _default_chatpx(device).get_px_by_date_range(start_date, end_date)
