"""Predict-only transformer memory for frozen close data.

The package ships the trained checkpoint produced by the notebook.  Runtime
inference only needs the transformer architecture, the checkpoint weights, and
arithmetic to turn predicted digits back into log_px.  The trading-date index is
reconstructed from the checkpoint by querying the memorized D and M prefixes.
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
FIXED_DIGITS = 11
FIXED_PLACE_VALUES = tuple(10**power for power in range(FIXED_DIGITS - 1, -1, -1))
DEFAULT_CHECKPOINT_NAME = "mem_digit.pt"
RANGE_BATCH_SIZE = 2048

PRICE_PREFIX = "P"
DATE_PREFIX = "D"
MAX_DATE_ID_PREFIX = "M"
PREFIX_TO_ID = {PRICE_PREFIX: 0, DATE_PREFIX: 1, MAX_DATE_ID_PREFIX: 2}

_CHATPX_CACHE: dict[str, "ChatPX"] = {}


class DateTransformerDigitMemorizer(nn.Module):
    """Transformer encoder that memorizes digit outputs by prefix and date_id.

    Input rows are [prefix_id, date_id].  P returns log_px_fixed digits, D
    returns YYYYMMDD digits, and M with date_id 0 returns max(date_id).
    """

    def __init__(self, num_dates, d_model=144, nhead=6, num_layers=3, dropout=0.0):
        """Build prefix/date embeddings, transformer layers, and digit heads.

        The architecture must match the training notebook exactly so the saved
        state_dict can be loaded without the original training package.
        """
        super().__init__()
        assert d_model % nhead == 0

        self.num_dates = int(num_dates)
        self.d_model = int(d_model)
        self.model_kwargs = {
            "d_model": int(d_model),
            "nhead": int(nhead),
            "num_layers": int(num_layers),
            "dropout": float(dropout),
        }

        self.prefix_embedding = nn.Embedding(len(PREFIX_TO_ID), d_model)
        self.date_embedding = nn.Embedding(self.num_dates, d_model)
        self.field_embedding = nn.Embedding(2, d_model)

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
        """Run [prefix_id, date_id] through the model.

        The returned tensor has shape [batch, 11, 10], one ten-way classifier
        for each fixed-point digit.
        """
        x = x.long()
        prefix_token = self.prefix_embedding(x[:, 0])
        date_token = self.date_embedding(x[:, 1])

        tokens = torch.stack([prefix_token, date_token], dim=1)
        field_ids = torch.arange(2, device=x.device).unsqueeze(0)
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
    return float(int(fixed) / LOG_PX_SCALE)


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
    """Validate checkpoint metadata shared by the bundled predictor."""
    if int(metadata.get("fixed_digits", FIXED_DIGITS)) != FIXED_DIGITS:
        raise ValueError(f"checkpoint fixed_digits does not match {FIXED_DIGITS}")

    prefix_to_id = metadata.get("prefix_to_id", PREFIX_TO_ID)
    for prefix, expected in PREFIX_TO_ID.items():
        if int(prefix_to_id.get(prefix, expected)) != expected:
            raise ValueError(f"checkpoint prefix id for {prefix!r} does not match {expected}")

    return metadata


def _prefix_to_id(prefix: str, metadata: dict[str, Any] | None = None) -> int:
    """Convert P, D, or M into the checkpoint prefix id."""
    prefix_map = PREFIX_TO_ID if metadata is None else metadata.get("prefix_to_id", PREFIX_TO_ID)
    key = prefix.upper()
    if key not in prefix_map:
        raise KeyError(f"unknown prefix {prefix!r}; expected one of {sorted(prefix_map)}")
    return int(prefix_map[key])


def make_features_for_date_ids(
    date_ids,
    prefix: str = PRICE_PREFIX,
    metadata: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Build feature rows shaped as [prefix_id, date_id]."""
    prefix_id = _prefix_to_id(prefix, metadata)
    rows = [[prefix_id, int(date_id)] for date_id in date_ids]
    return torch.as_tensor(rows, dtype=torch.long)


def _yyyymmdd_to_date_key(value: int) -> str:
    """Convert integer YYYYMMDD into an ISO date string."""
    value = int(value)
    year = value // 10_000
    month = (value // 100) % 100
    day = value % 100
    return _dt.date(year, month, day).isoformat()


@torch.no_grad()
def _predict_digits_for_date_ids(
    model: DateTransformerDigitMemorizer,
    date_ids,
    prefix: str,
    metadata: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
    batch_size: int = RANGE_BATCH_SIZE,
) -> list[list[int]]:
    """Predict 11 output digits for explicit date_id values using one prefix."""
    torch_device = get_device() if device is None else torch.device(device)
    date_id_list = [int(date_id) for date_id in date_ids]
    if not date_id_list:
        return []

    features = make_features_for_date_ids(date_id_list, prefix=prefix, metadata=metadata)
    rows: list[list[int]] = []
    model.eval()
    model.to(device=torch_device, dtype=MODEL_DTYPE)
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size].to(torch_device)
        logits = model(batch)
        rows.extend(logits.argmax(dim=2).detach().cpu().tolist())
    return [[int(digit) for digit in row] for row in rows]


@torch.no_grad()
def predict_max_date_id(
    model: DateTransformerDigitMemorizer,
    metadata: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Predict the memorized max(date_id) from the [M, 0] query."""
    digits = _predict_digits_for_date_ids(model, [0], MAX_DATE_ID_PREFIX, metadata=metadata, device=device)[0]
    return {
        "prefix": MAX_DATE_ID_PREFIX,
        "date_id": 0,
        "pred_digits": digits,
        "pred_max_date_id": digits_to_fixed(digits),
    }


@torch.no_grad()
def predict_yyyymmdd_for_date_id(
    model: DateTransformerDigitMemorizer,
    date_id: int,
    metadata: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
) -> dict[str, Any]:
    """Predict the memorized YYYYMMDD record for one date_id."""
    digits = _predict_digits_for_date_ids(model, [int(date_id)], DATE_PREFIX, metadata=metadata, device=device)[0]
    yyyymmdd = digits_to_fixed(digits)
    return {
        "prefix": DATE_PREFIX,
        "date_id": int(date_id),
        "pred_digits": digits,
        "pred_yyyymmdd": yyyymmdd,
        "pred_date": _yyyymmdd_to_date_key(yyyymmdd),
    }


@torch.no_grad()
def construct_date_index(
    model: DateTransformerDigitMemorizer,
    metadata: dict[str, Any] | None = None,
    device: str | torch.device | None = None,
    batch_size: int = RANGE_BATCH_SIZE,
) -> list[str]:
    """Rebuild the frozen trading-date index by querying M, then D for all ids."""
    torch_device = get_device() if device is None else torch.device(device)
    max_date_id = predict_max_date_id(model, metadata=metadata, device=torch_device)["pred_max_date_id"]
    expected_max_date_id = int(model.num_dates) - 1
    if int(max_date_id) != expected_max_date_id:
        raise ValueError(f"M predicted max_date_id={max_date_id}, expected {expected_max_date_id}")

    date_ids = list(range(int(max_date_id) + 1))
    digit_rows = _predict_digits_for_date_ids(
        model,
        date_ids,
        DATE_PREFIX,
        metadata=metadata,
        device=torch_device,
        batch_size=batch_size,
    )
    return [_yyyymmdd_to_date_key(digits_to_fixed(digits)) for digits in digit_rows]


def _prepare_date_index_metadata(
    model: DateTransformerDigitMemorizer,
    metadata: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Cache the reconstructed date index and date-to-id map in metadata."""
    if "date_index" not in metadata:
        metadata["date_index"] = construct_date_index(model, metadata=metadata, device=device)
    if "_date_to_id" not in metadata:
        metadata["_date_to_id"] = {date_key: idx for idx, date_key in enumerate(metadata["date_index"])}
    metadata["max_date_id"] = len(metadata["date_index"]) - 1
    return metadata


def load_model_checkpoint(device: str | torch.device | None = None):
    """Load model weights and raw metadata from the bundled checkpoint.

    Date-index reconstruction is a separate step performed by ChatPX. Only load
    checkpoints you trust because this uses torch.load, matching normal PyTorch
    checkpoint behavior.
    """
    torch_device = get_device() if device is None else torch.device(device)
    payload = _load_payload(torch_device)
    metadata = _prepare_metadata(dict(payload["metadata"]))
    state_dict = payload["state_dict"]
    num_dates = int(metadata.get("num_dates", state_dict["date_embedding.weight"].shape[0]))

    model = DateTransformerDigitMemorizer(
        num_dates,
        **metadata.get("model_kwargs", {}),
    )
    model.load_state_dict(state_dict)
    model.to(device=torch_device, dtype=MODEL_DTYPE)
    model.eval()
    metadata["num_dates"] = num_dates
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
    """Turn one frozen trading date into [P, date_id]."""
    metadata = _prepare_metadata(metadata)
    if "_date_to_id" not in metadata:
        raise ValueError("metadata has no reconstructed date index; ChatPX prepares it during initialization")
    date_key = _date_key(date_or_year, month, day)
    date_to_id = metadata["_date_to_id"]
    if date_key not in date_to_id:
        first_date = metadata["date_index"][0]
        last_date = metadata["date_index"][-1]
        raise KeyError(f"{date_key} is not in the frozen trading-date index {first_date}..{last_date}")

    return make_features_for_date_ids([date_to_id[date_key]], prefix=PRICE_PREFIX, metadata=metadata)


def _date_range_keys(start_date, end_date, metadata: dict[str, Any]) -> list[str]:
    """Return frozen trading-date keys between start_date and end_date, inclusive."""
    if "date_index" not in metadata:
        raise ValueError("metadata has no reconstructed date index; ChatPX prepares it during initialization")
    start_key = _date_key(start_date)
    end_key = _date_key(end_date)
    if start_key > end_key:
        raise ValueError(f"start_date {start_key} must be on or before end_date {end_key}")
    return [date_key for date_key in metadata["date_index"] if start_key <= date_key <= end_key]


def _make_features_for_date_keys(date_keys: list[str], metadata: dict[str, Any]) -> torch.Tensor:
    """Turn ordered date keys into model feature rows."""
    metadata = _prepare_metadata(metadata)
    if "_date_to_id" not in metadata:
        raise ValueError("metadata has no reconstructed date index; ChatPX prepares it during initialization")
    date_to_id = metadata["_date_to_id"]
    return make_features_for_date_ids([date_to_id[date_key] for date_key in date_keys], prefix=PRICE_PREFIX, metadata=metadata)


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
        "date_id": int(x.detach().cpu()[0, 1].item()),
        "yyyy": parsed.year,
        "mm": parsed.month,
        "dd": parsed.day,
        "pred_digits": digits,
        "pred_log_px_fixed": fixed,
        "pred_log_px": log_px,
        "pred_px": float(math.exp(log_px)),
    }


class ChatPX:
    """Reusable predictor that keeps the transformer checkpoint loaded.

    Create one instance when asking for many dates.  The model and date index are
    loaded during initialization and reused for every method call.
    """

    def __init__(self, device: str | torch.device | None = None):
        """Load the checkpoint, then reconstruct the date index from the model."""
        self.device = get_device() if device is None else torch.device(device)
        self.model, self.metadata = load_model_checkpoint(self.device)
        _prepare_date_index_metadata(self.model, self.metadata, self.device)

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
