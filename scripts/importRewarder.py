import joblib
import os

model_path = os.path.join(os.path.dirname(__file__), "../rewarder/xgb_deathcode_predictor.pkl")
model = joblib.load(model_path)
#y_new = model.predict(new_data[features])