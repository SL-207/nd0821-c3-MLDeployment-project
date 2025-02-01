# Script to train machine learning model
from ml.data import process_data
from ml.model import train_model, save_model, load_model, compute_model_metrics, inference
import pandas as pd
import os

from sklearn.model_selection import train_test_split


current_dir = os.path.dirname(os.path.abspath(__file__))
proj_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
data_pth = os.path.join(proj_root, "starter", "data", "census.csv")

data = pd.read_csv(data_pth)

train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

model_pth = os.path.join(proj_root, "starter", "model", "model.pkl")
encoder_pth = os.path.join(proj_root, "starter", "model", "encoder.pkl")
model = train_model(X_train, y_train)
save_model(model, model_pth, encoder, lb, encoder_pth)
loaded_model, encoder, lb = load_model(model_pth, encoder_pth)
y_preds = inference(loaded_model, X_train)
precision, recall, fbeta = compute_model_metrics(y_train, y_preds)
print(
    f"<< Precision: {precision:.3f} | Recall: {recall:.3f} | fbeta: {fbeta:.3f}>>")
