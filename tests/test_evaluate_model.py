from scripts.evaluate_model import build_prediction_row


def test_prediction_row_records_timing_and_raw_log_probabilities():
    score_map = {"0": -2.5, "1": -0.25}

    result = build_prediction_row(
        row={"sample_id": "sample-1", "output": "1"},
        score_map=score_map,
        prompt_text="prompt",
        was_truncated=False,
        inference_seconds=0.125,
        batch_inference_seconds=0.5,
        inference_batch_size=4,
    )

    assert result["prediction"] == "1"
    assert result["raw_label_log_probabilities"] == score_map
    assert result["selected_label_log_probability"] == -0.25
    assert result["inference_seconds"] == 0.125
    assert result["batch_inference_seconds"] == 0.5
    assert result["inference_batch_size"] == 4
