"""Send a sample request to the API."""
import requests
import json

SAMPLE_APPLICANT = {
    "features": {
        "AMT_INCOME_TOTAL": 180000.0,
        "AMT_CREDIT": 500000.0,
        "AMT_ANNUITY": 25000.0,
        "AMT_GOODS_PRICE": 450000.0,
        "DAYS_BIRTH": -12000,
        "DAYS_EMPLOYED": -2000,
        "EXT_SOURCE_2": 0.55,
        "EXT_SOURCE_3": 0.42,
        "NAME_EDUCATION_TYPE": "Higher education",
        "NAME_CONTRACT_TYPE": "Cash loans",
        "CODE_GENDER": "F",
        "CNT_CHILDREN": 0,
        "CNT_FAM_MEMBERS": 2.0
    }
}

response = requests.post("http://localhost:8000/predict", json=SAMPLE_APPLICANT)
print(f"Status: {response.status_code}")
print(json.dumps(response.json(), indent=2))