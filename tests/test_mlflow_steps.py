from training.mlflow_steps import new_stream_counters


def test_stream_counters_independent():
    c = new_stream_counters()
    c.train_batch[0] += 2
    c.val_batch[0] += 5
    assert c.train_batch[0] == 2
    assert c.val_batch[0] == 5
    assert c.test_batch[0] == 0
