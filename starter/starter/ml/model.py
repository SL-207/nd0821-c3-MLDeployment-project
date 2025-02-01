import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
import xgboost as xgb
import numpy as np


def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model = xgb.XGBClassifier(objective="binary:logistic")
    model.fit(X_train, y_train)
    return model


def save_model(
        model,
        model_pth: str,
        cat_encoder,
        label_encoder,
        encoder_pth: str):
    """Saves model as pickle file given a model and its path

    Inputs
    ------
    model : xgboost.XGBModel
        Trained model.
    model_pth : str
        Path of pkl file for model to be saved as.
    cat_encoder : OneHotEncoder
        Categorical encoder.
    lb_encoder : LabelEncoder
        Label encoder.
    encoder_pth : str
        Path of pkl file for encoder to be saved as.
    """
    with open(model_pth, "wb") as file:
        pickle.dump(model, file)

    with open(encoder_pth, "wb") as file:
        pickle.dump({"cat": cat_encoder, "lb": label_encoder}, file)


def load_model(model_pth: str, encoder_pth):
    """Loads a pickle model given its path

    Inputs
    ------
    model_pth : str
        Path of pkl file of model.

    Return
    ------
    model : xgboost.XGBModel
    cat_encoder : OneHotEncoder
    lb_encoder : LabelEncoder
    """
    with open(model_pth, 'rb') as file:
        model = pickle.load(file)

    with open(encoder_pth, 'rb') as file:
        encoder = pickle.load(file)
    return model, encoder["cat"], encoder["lb"]


def compute_slice_metrics(model, train, X_train, y_train, category):
    """Compute and print metrics for each possible slice of a given category.

    Inputs
    ------
    model : xgboost.XGBModel
        Trained model.
    train : pd.DataFrame
        unprocessed train dataframe.
    X_train : np.array
    y_train: np.array
    category : str
        Category or column in dataframe for slice
    """
    with open("slice_output.txt", "w") as file:
        for value in train[category].unique():
            print(category, " = ", value)
            condition = train[category] == value
            indices = train.index[condition]
            filtered_x = X_train[indices]
            filtered_y = y_train[indices]
            y_preds = model.predict(filtered_x)
            precision, recall, fbeta = compute_model_metrics(
                filtered_y, y_preds)
            file.write(f"{category} = {value}\n")
            file.write(
                f"<< Precision: {precision:.3f} | Recall: {recall:.3f} | fbeta: {fbeta:.3f}>>\n")


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    ------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : xgboost.XGBModel
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)
