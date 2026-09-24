from models.model41.run_model41 import CONFIG


def test_model41_gate_is_pilot_frozen():
    assert CONFIG["pilot_folds"] == [0, 1]
    assert CONFIG["confirmation_folds"] == [2, 3, 4]
    assert CONFIG["correction_before_time"] == 128

