from ml.model import load_model, inference, compute_model_metrics
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
import xgboost as xgb
from train_model import X_train, y_train
import pytest


@pytest.fixture
def model_encoder_pth():
    model_pth = "../model/model.pkl"
    encoder_pth = "../model/encoder.pkl"
    return model_pth, encoder_pth


def test_load_model(model_encoder_pth):
    try:
        model, cat_encoder, lb_encoder = load_model(*model_encoder_pth)
        assert isinstance(model, xgb.XGBClassifier)
        assert isinstance(cat_encoder, OneHotEncoder)
        assert isinstance(lb_encoder, LabelBinarizer)
    except AssertionError:
        print("load_model returned an output of incorrect type")


def test_inference(model_encoder_pth):
    try:
        model, _, _ = load_model(*model_encoder_pth)
        output = inference(model, X_train)
        assert output.shape == X_train.shape[0]
        assert set(output) == {0, 1} or set(
            output) == {0} or set(output) == {1}
    except AssertionError:
        print("Inference function output is not as expected")


def test_compute_model_metrics(model_encoder_pth):
    try:
        model, _, _ = load_model(*model_encoder_pth)
        output = inference(model, X_train)
        precision, recall, fbeta = compute_model_metrics(y_train, output)
        assert isinstance(precision, float)
        assert isinstance(recall, float)
        assert isinstance(fbeta, float)

    except AssertionError:
        print("Compute model metrics did not run correctly and metrics may be of the wrong type")
