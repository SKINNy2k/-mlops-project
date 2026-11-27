import pytest
from promotion_gate import check_promotion

def test_good_model_is_approved():
    result = check_promotion({"accuracy": .95, "f1_score": .95})
    assert result["approved"] is True and result["status"] == "APPROVED"

def test_bad_model_is_rejected():
    result = check_promotion({"accuracy": .70, "f1_score": .70})
    assert result["approved"] is False and result["status"] == "REJECTED"

def test_model_without_improvement_is_rejected():
    result = check_promotion({"accuracy": .95, "f1_score": .95}, baseline_f1=.95)
    assert result["approved"] is False

def test_model_with_improvement_is_approved():
    result = check_promotion({"accuracy": .96, "f1_score": .96}, baseline_f1=.94)
    assert result["approved"] is True

def test_missing_metrics_raise_error():
    with pytest.raises(ValueError): check_promotion({"accuracy": .95})

def test_out_of_range_metrics_raise_error():
    with pytest.raises(ValueError): check_promotion({"accuracy": 1.2, "f1_score": .95})
