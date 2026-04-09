from __future__ import annotations

import numpy as np

from tchhmrl.constraints.dual_layer import DualLayer


def test_dual_layer_vector_penalty_and_update():
    dual = DualLayer.from_meta_cfg(
        {
            "dual_names": ["qos", "temp_anchor", "temp_boost1", "temp_boost2"],
            "dual_lrs": [0.1, 0.2, 0.2, 0.2],
            "dual_target_costs": [0.0, 0.1, 0.1, 0.1],
            "dual_max_lambdas": [10.0, 10.0, 10.0, 10.0],
        },
        n_tx=3,
    )

    mean_cost = np.asarray([0.5, 0.2, 0.0, 0.4], dtype=np.float32)
    lambda_mean = dual.update(mean_cost)
    penalty = dual.penalty(mean_cost)

    assert dual.n_constraints == 4
    assert lambda_mean > 0.0
    assert penalty > 0.0
    assert set(dual.as_dict().keys()) == {
        "lambda_qos",
        "lambda_temp_anchor",
        "lambda_temp_boost1",
        "lambda_temp_boost2",
    }
