from models.model42.run_model42 import CONFIG, transform_spec


def test_model42_transform_and_split_are_frozen():
    spec = transform_spec()
    assert spec.kind == "logit_ema_trend" and spec.half_life == 8 and spec.correction_weight == -0.1
    assert CONFIG["pilot_folds"] == [0, 1]
    assert CONFIG["confirmation_folds"] == [2, 3, 4]

