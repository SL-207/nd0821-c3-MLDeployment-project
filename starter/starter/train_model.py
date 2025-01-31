# Script to train machine learning modelimport pandas as pd
from ml.data import process_data
from ml.model import train_model, save_model, load_model, compute_model_metrics, inference
import numpy as n
import pandas as pd

from sklearn.model_selection import train_test_split

# Add code to load in the data.
data = pd.read_csv("../data/census.csv")
# Optional enhancement, use K-fold cross validation instead of a train-test split.
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

model_pth = "../model/model3.pkl"
encoder_pth = "../model/encoder.pkl"
model = train_model(X_train, y_train)
save_model(model, model_pth, encoder, lb, encoder_pth)
loaded_model, encoder, lb = load_model(model_pth, encoder_pth)
y_preds = inference(loaded_model, X_train)
precision, recall, fbeta = compute_model_metrics(y_train, y_preds)
print(f"<< Precision: {precision:.3f} | Recall: {recall:.3f} | fbeta: {fbeta:.3f}>>")
