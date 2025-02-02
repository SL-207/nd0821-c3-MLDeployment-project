import requests
import json

URL = "https://nd0821-c3-mldeployment-project.onrender.com/inference"
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

r = requests.post(URL, data=data)
if r.status_code == 200:
    print("Status Code: ", r.status_code)
    print(r.json())
else:
    print(r.status_code, ": ", r.text)