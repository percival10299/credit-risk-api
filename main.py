import pandas as pd
import numpy as np

# 1. Load Data
print("Loading data...")
df = pd.read_csv('data/application_train.csv')
print(f"Original Shape: {df.shape}")


# --- DATA CLEANING ---

print(df.describe().T)
print(df.head())
values = df['DAYS_EMPLOYED'].value_counts()
print(values)

# Fix the anomaly (365243 days employed -> NaN)
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

# Drop columns that are mostly empty (optional, but speeds things up)
# If >30% of the data is missing, we drop the column for this baseline
df = df.dropna(thresh=0.7*len(df), axis=1)

# Fill Missing Values (Imputation)
# Select only numeric columns to calculate median
numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill Categorical Missing Values
# We fill text columns with "Unknown" so they become their own category
categorical_cols = df.select_dtypes(include=['object']).columns
df[categorical_cols] = df[categorical_cols].fillna("Unknown")

# Encode Categorical Data (Text -> Numbers)
print("Encoding categorical data...")
df = pd.get_dummies(df)

# --- END CLEANING ---

print("-" * 30)
print("Data Cleaned Successfully!")
print(f"Final Shape: {df.shape}")
print(f"Remaining NaNs: {df.isnull().sum().sum()}") # Should be 0

# --- PREPARATION FOR MODELING ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# --- STEP 1: PREPARE DATA ---
print("Preparing X and y...")

# 'TARGET' is what we want to predict (y)
y = df['TARGET']

# 'X' is everything else (drop TARGET)
X = df.drop(columns=['TARGET'])

# Free up memory (optional, but good for big datasets)
del df

# --- STEP 2: SPLIT DATA ---
# 80% for training, 20% for testing
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- STEP 3: SCALE DATA ---
# Logistic Regression requires scaling to work well
print("Scaling features...")
scaler = MinMaxScaler()

# Learn the scale from training data ONLY, then apply to both
# (Never learn scale from test data, that is "Data Leakage")
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score

# --- 1. CONVERT DATA TO TENSORS ---
# PyTorch requires 'FloatTensors' instead of numpy arrays
# We use .astype(np.float32) to ensure precision matches
X_train_tensor = torch.tensor(X_train_scaled.astype(np.float32))
y_train_tensor = torch.tensor(y_train.values.astype(np.float32)).view(-1, 1)

X_test_tensor = torch.tensor(X_test_scaled.astype(np.float32))
y_test_tensor = torch.tensor(y_test.values.astype(np.float32)).view(-1, 1)

# --- 2. DEFINE THE MODEL ---
class CreditNeuralNet(nn.Module):
    def __init__(self, input_dim):
        super(CreditNeuralNet, self).__init__()
        
        # 1. First "Hidden" Layer: Expands the 168 inputs to 64 patterns
        self.hidden1 = nn.Linear(input_dim, 64)
        
        # 2. ReLU Activation: The "Magic" that allows non-linearity
        self.relu = nn.ReLU()
        
        # 3. Output Layer: Compresses the 64 patterns into 1 risk score
        self.output = nn.Linear(64, 1)
        
        # 4. Final Sigmoid: Squashes result to a 0-1 probability
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Data flows: Linear -> ReLU -> Linear -> Sigmoid
        x = self.hidden1(x)
        x = self.relu(x)
        x = self.output(x)
        x = self.sigmoid(x)
        return x

# Initialize Model
input_features = X_train_tensor.shape[1] # Should be ~168
model = CreditNeuralNet(input_features)

# --- 3. DEFINE LOSS & OPTIMIZER ---
# A. Calculate the Weight (SAME AS BEFORE)
num_pos = y_train.sum()
num_neg = len(y_train) - num_pos
# Approx 11.5x multiplier for defaults
weight_multiplier = num_neg / num_pos 

# B. Create a Weight Tensor for EVERY row (THE FIX)
# Start by giving everyone a weight of 1.0
weights = torch.ones_like(y_train_tensor)
# Find the Defaulters (where y == 1) and change their weight to 11.5
weights[y_train_tensor == 1] = weight_multiplier

# C. Plug it into the Criterion
# Now BCELoss knows exactly how important each specific student is
criterion = nn.BCELoss(weight=weights)

optimizer = optim.Adam(model.parameters(), lr=0.001)

# --- 4. THE TRAINING LOOP ---
print("Starting PyTorch Training (Weighted)...")
epochs = 400

for epoch in range(epochs):
    # A. Forward Pass
    outputs = model(X_train_tensor)
    
    # B. Calculate Loss
    # The 'criterion' now automatically applies the 11.5x multiplier 
    # to any row that is a Defaulter
    loss = criterion(outputs, y_train_tensor)
    
    # C. Backward Pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

# --- 5. EVALUATION ---
print("Evaluating...")
model.eval() # Set to evaluation mode
with torch.no_grad():
    # Get probabilities
    y_pred_probs = model(X_test_tensor).numpy()
    
    # Calculate Score
    score = roc_auc_score(y_test, y_pred_probs)
    print("-" * 30)
    print(f"PyTorch ROC AUC Score: {score:.4f}")
    print("-" * 30)

import joblib

# --- 6. SAVE ARTIFACTS FOR PHASE 2 ---
print("Saving model and scaler...")

# A. Save the PyTorch Model Weights
torch.save(model.state_dict(), 'credit_model.pth')
print("Model saved to 'credit_model.pth'")

# B. Save the Scaler (to scale API inputs later)
joblib.dump(scaler, 'scaler.pkl')
print("Scaler saved to 'scaler.pkl'")

# C. Save the Feature Columns (Optional but helpful for API)
# The API needs to know which 168 columns to expect
import json
columns_list = list(X.columns)
with open('model_columns.json', 'w') as f:
    json.dump(columns_list, f)
print("Column names saved to 'model_columns.json'")