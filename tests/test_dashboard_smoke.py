from pathlib import Path


def test_dashboard_pages_exist():
    expected = [
        "Overview.py",
        "pages/1X2.py",
        "pages/Over_Under_2_5.py",
        "pages/BTTS.py",
        "pages/Correct_Score.py",
        "pages/Asian_Handicap.py",
        "pages/Fixture_Drilldown.py",
        "pages/Backtest_Lab.py",
        "pages/Experiments.py",
        "pages/Run_Control.py",
        "pages/History.py",
    ]
    for rel in expected:
        assert (Path("app") / rel).exists()
