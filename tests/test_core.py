import torch

from chat_spx import ChatPX, get_px, get_px_by_date_range, load_model_checkpoint, make_feature_for_date, predict_max_date_id
from chat_spx.core import _prepare_date_index_metadata


def test_get_px_1927_12_30_rounds_to_expected_close():
    assert round(get_px("1927-12-30"), 6) == 17.660000


def test_get_px_1990_01_05_rounds_to_expected_close():
    assert round(get_px("1990-01-05"), 6) == 352.200000


def test_chatpx_reuses_loaded_model_for_multiple_dates():
    chatpx = ChatPX()
    assert round(chatpx.get_px("1927-12-30"), 6) == 17.660000
    assert round(chatpx.get_px("1990-01-05"), 6) == 352.200000


def test_checkpoint_reconstructs_date_index_from_model():
    chatpx = ChatPX()
    assert chatpx.metadata["date_index"][0] == "1896-05-26"
    assert chatpx.metadata["date_index"][-1] == "2026-05-29"


def test_load_model_checkpoint_returns_raw_metadata_without_date_index():
    _, metadata = load_model_checkpoint(device="cpu")
    assert "date_index" not in metadata
    assert "_date_to_id" not in metadata


def test_prepare_date_index_metadata_adds_date_lookup_and_feature_date_id():
    model, metadata = load_model_checkpoint(device="cpu")
    _prepare_date_index_metadata(model, metadata, torch.device("cpu"))

    assert metadata["date_index"][0] == "1896-05-26"
    assert metadata["date_index"][-1] == "2026-05-29"
    assert metadata["_date_to_id"]["1896-05-26"] == 0
    assert metadata["_date_to_id"]["2026-05-29"] == 32555

    feature = make_feature_for_date("1990-01-05", metadata)
    assert feature.shape == (1, 2)
    assert int(feature[0, 1]) == metadata["_date_to_id"]["1990-01-05"]


def test_m_prefix_predicts_max_date_id():
    chatpx = ChatPX()
    assert predict_max_date_id(chatpx.model, chatpx.metadata, device=chatpx.device)["pred_max_date_id"] == 32555


def test_predict_uses_direct_fixed_point_output():
    record = ChatPX().predict("1990-01-05")
    assert "pred_log_px_fixed" in record
    assert "pred_log_px_x" not in record
    assert "pred_log_px_y" not in record


def test_get_px_by_date_range_returns_floats_for_inclusive_trading_dates():
    pxs = get_px_by_date_range("2026-05-18", "2026-05-21")
    assert isinstance(pxs, list)
    assert all(isinstance(px, float) for px in pxs)
    assert [round(px, 2) for px in pxs] == [7403.05, 7353.61, 7432.97, 7445.72]


def test_chatpx_get_px_by_date_range_returns_same_values():
    chatpx = ChatPX()
    pxs = chatpx.get_px_by_date_range("2026-05-18", "2026-05-21")
    assert [round(px, 2) for px in pxs] == [7403.05, 7353.61, 7432.97, 7445.72]
