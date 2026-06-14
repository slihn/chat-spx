# chat-spx

This small AI package predicts SPX Close data from a PyTorch transformer model.


This package is predict-only.  The bundled checkpoint contains the model
weights and the trading-date index through `5/29/2026`.
Before `12/30/1927`, the data is rescaled from the Industrial close since `5/26/1896`.

Disclaimer: There is no guarrantee of data precision since the AI model is prone to error. Use with your own risk.


## Install

From this repository:

```bash
pip install .
```

For editable development:

```bash
pip install -e .
```

## Usage

```python
from chat_spx import get_px, get_log_px, predict

px = get_px("1927-12-30")
log_px = get_log_px("1927-12-30")
record = predict("1990-01-05")

print(f"{px:.6f}")  # 17.660000
print(log_px)
print(record["pred_log_px_fixed"])
```

Date ranges return a list of floats for trading dates in the inclusive range:

```python
from chat_spx import get_px_by_date_range

pxs = get_px_by_date_range("2026-05-18", "2026-05-21")
print([round(px, 2) for px in pxs])  # [7403.05, 7353.61, 7432.97, 7445.72]
```

For repeated calls, create one `ChatPX` instance so the model is loaded once:

```python
from chat_spx import ChatPX

chatpx = ChatPX()

print(f"{chatpx.get_px('1927-12-30'):.6f}")  # 17.660000
print(f"{chatpx.get_px('1990-01-05'):.6f}")  # 352.200000
print([round(px, 2) for px in chatpx.get_px_by_date_range("2026-05-18", "2026-05-21")])
```

You can also pass date parts:

```python
from chat_spx import get_px

px = get_px(1990, 1, 5)
print(f"{px:.6f}")  # 352.200000
```

The command-line entry point prints one price by default:

```bash
chat-spx 1990-01-05
chat-spx --price 1990-01-05
```

For a trading-date range:

```bash
chat-spx --range 2026-05-18 2026-05-21
```

## Notes

- `get_px(date)` returns `exp(log_px)`.
- `get_px_by_date_range(start_date, end_date)` returns prices for frozen trading
  dates in the inclusive range.
- `get_log_px(date)` returns the memorized fixed-point `log_px` value.
- `predict(date)` returns the date id, predicted digits, fixed integer, `log_px`,
  and `px`.
- The date must be in the frozen trading-date index.  Weekends, holidays,
  and dates after `5/29/2026` raise `KeyError`.

## Tests

```bash
pip install -e ".[test]"
pytest
```

## Publishing

Build with:

```bash
python -m build
```

Upload with:

```bash
python -m twine upload dist/*
```

The package includes the model weights, so the distribution is much
larger than a typical pure-Python package. It is about 46MB.
