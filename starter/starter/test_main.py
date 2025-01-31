from fastapi.testclient import TestClient
import json

from starter.starter.main import app

client = TestClient(app)

def test_greeting():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"greeting": "Greetings! The following app makes predictions on income range based on demographic and census-related factors"}
    
def test_inference_under_fifty():
    data = json.dumps({
        "age": 39,
        "workclass": "State-gov",
        "fnlgt": 77516,
        "education": "Bachelors",
        "education_num": 13,
        "marital_status": "Never-married",
        "occupation": "Adm-clerical",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Male",
        "capital_gain": 2174,
        "capital_loss": 0,
        "hours_per_week": 40,
        "native_country": "United-States"
    })
    r = client.post("/inference", data=data)
    assert r.status_code == 200
    assert r.json() == {"output": "Salary is less than or equal to 50k"}
    
def test_inference_over_fifty():
    data = json.dumps({
        "age": 43,
        "workclass": "Federal-gov",
        "fnlgt": 410867,
        "education": "Doctorate",
        "education_num": 16,
        "marital_status": "Never-married",
        "occupation": "Prof-specialty",
        "relationship": "Not-in-family",
        "race": "White",
        "sex": "Female",
        "capital_gain": 0,
        "capital_loss": 0,
        "hours_per_week": 50,
        "native_country": "United-States"
    })
    r = client.post("/inference", data=data)
    assert r.status_code == 200
    assert r.json() == {"output": "Salary is over 50k"}