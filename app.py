import torch
import torch.nn as nn
import pandas as pd
import joblib
import json
from fastapi import FastAPI
from pydantic import BaseModel

# Initialize the App
app = FastAPI()

# --- THE BLUEPRINT ---
# We must paste the EXACT same class here so PyTorch knows the architecture
class LogisticRegressionPyTorch(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionPyTorch, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        outputs = self.linear(x)
        return torch.sigmoid(outputs)

# Global variables to hold the "Brain" (Model), "Ruler" (Scaler), and "Map" (Columns)
model = None
scaler = None
model_columns = None

@app.on_event("startup")
def load_artifacts():
    global model, scaler, model_columns
    
    # 1. Load the "Map" (So we know the order of columns: Income, Age, etc.)
    with open('model_columns.json', 'r') as f:
        model_columns = json.load(f)
    
    # 2. Load the "Ruler" (To scale 50000 -> 0.5)
    scaler = joblib.load('scaler.pkl')
    
    # 3. Load the "Brain"
    # We tell it: "Expect inputs equal to the number of columns in our map"
    model = LogisticRegressionPyTorch(input_dim=len(model_columns))
    
    # Actually load the weights
    model.load_state_dict(torch.load('credit_model.pth'))
    
    # Switch to "Test Mode" (turns off training specific features)
    model.eval()
    
    print("System Online: Model Loaded Successfully.")

# --- 3. DEFINE INPUT DATA (THE PROFESSIONAL WAY) ---
class LoanApplication(BaseModel):
    # 1. Define the Key Features (Strictly Validated)
    # The user MUST send these, and they MUST be the right type.
    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float
    DAYS_EMPLOYED: int  # Must be an integer
    NAME_CONTRACT_TYPE: str # Must be text (e.g., "Cash loans")
    
    # 2. Allow other columns without typing them all
    # This configuration tells Pydantic: "If the user sends extra fields 
    # (like CNT_CHILDREN) that I didn't list above, just accept them."
    class Config:
        extra = "allow"

@app.post("/predict")
def predict(loan: LoanApplication):
    # 1. Convert JSON to DataFrame
    # We wrap it in [] because DataFrame expects a list of rows
    df = pd.DataFrame([loan.model_dump()])
    
    # 2. Data Cleaning (Mini-Version)
    # We must do the exact same cleaning we did in training
    if 'DAYS_EMPLOYED' in df.columns:
        df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, 0)
        # Note: In production, we'd use np.nan, but 0 is safer for a quick API fix
    
    # 3. Encoding (Text -> Numbers)
    df = pd.get_dummies(df)
    
    # 4. ALIGNMENT (The Fix for the Ghost Column Problem)
    # "Force this dataframe to have exactly the same columns as our training data"
    # If a column is missing, fill it with 0.
    df = df.reindex(columns=model_columns, fill_value=0)
    
    # 5. Scaling
    # Use the saved ruler to shrink the numbers
    df_scaled = scaler.transform(df)
    
    # 6. Predict
    # Convert to Tensor -> Feed to Model -> Get Number
    input_tensor = torch.tensor(df_scaled.astype('float32'))
    
    with torch.no_grad():
        prediction = model(input_tensor)
        
    # Return the result as JSON
    return {"default_probability": prediction.item()}