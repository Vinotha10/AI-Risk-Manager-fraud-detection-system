import joblib
import pandas as pd
import json

base_model = joblib.load('models/base_model.pkl')
print("Loaded without warning")
test_final = pd.read_pickle('data/paysim_test_features.pkl')

with open('models/feature_columns.json') as f:
    feature_columns = json.load(f)

print("Model loaded:", base_model)
print("Feature columns:", feature_columns)
print("Test shape:", test_final.shape)