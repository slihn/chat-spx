from chat_spx import ChatPX, get_px, get_px_by_date_range


def test_get_px_1927_12_30_rounds_to_expected_close():
    assert round(get_px("1927-12-30"), 6) == 17.660000


def test_get_px_1990_01_05_rounds_to_expected_close():
    assert round(get_px("1990-01-05"), 6) == 352.200000


def test_chatpx_reuses_loaded_model_for_multiple_dates():
    chatpx = ChatPX()
    assert round(chatpx.get_px("1927-12-30"), 6) == 17.660000
    assert round(chatpx.get_px("1990-01-05"), 6) == 352.200000


def test_get_px_by_date_range_returns_floats_for_inclusive_trading_dates():
    pxs = get_px_by_date_range("2026-05-18", "2026-05-21")
    assert isinstance(pxs, list)
    assert all(isinstance(px, float) for px in pxs)
    assert [round(px, 2) for px in pxs] == [7403.05, 7353.61, 7432.97, 7445.72]


def test_chatpx_get_px_by_date_range_returns_same_values():
    chatpx = ChatPX()
    pxs = chatpx.get_px_by_date_range("2026-05-18", "2026-05-21")
    assert [round(px, 2) for px in pxs] == [7403.05, 7353.61, 7432.97, 7445.72]
