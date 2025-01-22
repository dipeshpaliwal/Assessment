import pandas as pd
import numpy as np

np.random.seed(42)

data = {
    "Machine_ID": [f"M{i}" for i in range(1, 101)],
    "Temperature": np.random.uniform(60, 100, 100),
    "Run_Time": np.random.uniform(50, 500, 100),
    "Downtime_Flag": np.random.choice([0, 1], size=100, p=[0.8, 0.2]),
}

df = pd.DataFrame(data)
df.to_csv("data/synthetic_data.csv", index=False)
print("Synthetic dataset saved to 'data/synthetic_data.csv'")


from fastapi import FastAPI, UploadFile, File
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import pickle
import os

# Initialize FastAPI app
app = FastAPI()

# Global variables for dataset, model, and scaler
df = None
model = None
scaler = StandardScaler()

# Load the model if it exists
if os.path.exists("models/model.pkl"):
    with open("models/model.pkl", "rb") as f:
        model, scaler = pickle.load(f)

# Endpoint 1: Upload Dataset
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    global df
    if file.content_type != "text/csv":
        return {"error": "Invalid file format. Please upload a CSV file."}

    df = pd.read_csv(file.file)
    return {"message": "File uploaded successfully!", "columns": list(df.columns)}

# Endpoint 2: Train Model
@app.post("/train")
def train_model():
    global model, scaler
    if df is None:
        return {"error": "No data uploaded. Please upload a dataset first."}

    # Extract features and target
    X = df[["Temperature", "Run_Time"]]
    y = df["Downtime_Flag"]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalize data
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train the model
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Save the model
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump((model, scaler), f)

    return {"accuracy": accuracy, "f1_score": f1}

# Endpoint 3: Predict
@app.post("/predict")
def predict(input_data: dict):
    global model, scaler
    if model is None:
        return {"error": "Model not trained. Please train the model first."}

    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)[0]
    confidence = max(model.predict_proba(input_scaled)[0])

    return {"Downtime": "Yes" if prediction == 1 else "No", "Confidence": round(confidence, 2)}


