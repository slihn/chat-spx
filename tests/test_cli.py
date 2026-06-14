import json

from chat_spx.__main__ import main


def test_main_price_option_prints_one_price(capsys):
    main(["--price", "1927-12-30"])
    out = capsys.readouterr().out.strip()
    assert round(float(out), 6) == 17.660000


def test_main_positional_date_defaults_to_price(capsys):
    main(["1990-01-05"])
    out = capsys.readouterr().out.strip()
    assert round(float(out), 6) == 352.200000


def test_main_range_option_prints_price_list(capsys):
    main(["--range", "2026-05-18", "2026-05-21"])
    out = capsys.readouterr().out.strip()
    assert [round(px, 2) for px in json.loads(out)] == [7403.05, 7353.61, 7432.97, 7445.72]
