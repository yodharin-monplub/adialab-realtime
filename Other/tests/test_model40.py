from models.model40.run_model40 import CONFIG


def test_model40_fixes_exactly_two_smoothing_passes():
    assert CONFIG["smoothing_passes"] == 2

