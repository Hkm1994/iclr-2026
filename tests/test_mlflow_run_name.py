from training.mlflow_run_name import make_mlflow_run_name


def test_make_mlflow_run_name_unique_and_shape():
    a = make_mlflow_run_name("strong_mlp", {}, {})
    b = make_mlflow_run_name("strong_mlp", {}, {})
    assert a != b
    assert "strong_mlp" in a
    assert len(a) <= 200


def test_make_mlflow_run_name_prefix():
    n = make_mlflow_run_name(
        "mlp",
        {"mlflow_run_name_prefix": "sweep_a"},
        {},
    )
    assert n.startswith("sweep_a-mlp-")
