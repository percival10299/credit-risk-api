import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import json
import warnings
warnings.filterwarnings('ignore')

print("1. Loading main training data...")
df = pd.read_csv('data/application_train.csv')

# --- PHASE 1: RELATIONAL DATA AGGREGATION ---

print("2. Aggregating historical bureau.csv data...")
bureau = pd.read_csv('data/bureau.csv')
# Group by applicant and calculate statistical summaries of their past loans
bureau_agg = bureau.groupby('SK_ID_CURR').agg({
    'SK_ID_BUREAU': 'count',              # Total number of past loans
    'DAYS_CREDIT': ['min', 'max', 'mean'],# How recently they applied for credit
    'AMT_CREDIT_SUM': ['sum', 'mean'],    # Total amount they have borrowed in the past
    'AMT_CREDIT_SUM_DEBT': ['sum', 'mean']# Total amount they currently owe elsewhere
})
# Flatten the multi-level columns created by .agg()
bureau_agg.columns = ['BUREAU_' + '_'.join(c).strip().upper() for c in bureau_agg.columns.values]
# Merge into main dataframe
df = df.merge(bureau_agg, on='SK_ID_CURR', how='left')
del bureau, bureau_agg # Explicitly free RAM

print("3. Aggregating previous_application.csv data...")
prev = pd.read_csv('data/previous_application.csv')
prev_agg = prev.groupby('SK_ID_CURR').agg({
    'SK_ID_PREV': 'count',                # Total previous applications at THIS bank
    'AMT_APPLICATION': ['mean', 'max'],   # How much they asked for previously
    'CNT_PAYMENT': ['mean', 'sum']        # Length of their previous loan terms
})
prev_agg.columns = ['PREV_' + '_'.join(c).strip().upper() for c in prev_agg.columns.values]
df = df.merge(prev_agg, on='SK_ID_CURR', how='left')
del prev, prev_agg # Explicitly free RAM

print("3.5 Aggregating installments_payments.csv data (Behavioral Alpha)...")
installments = pd.read_csv('data/installments_payments.csv')

# 1. Did they pay late? (Positive days = late)
installments['DAYS_PAST_DUE'] = installments['DAYS_ENTRY_PAYMENT'] - installments['DAYS_INSTALMENT']
installments['DAYS_PAST_DUE'] = installments['DAYS_PAST_DUE'].apply(lambda x: x if x > 0 else 0)

# 2. Did they underpay?
installments['PAYMENT_FRACTION'] = installments['AMT_PAYMENT'] / installments['AMT_INSTALMENT']

# THE FIX: Replace 'inf' and '-inf' (caused by division by zero) with NaN
installments.replace([np.inf, -np.inf], np.nan, inplace=True)

# Group by applicant and extract their payment behavior
inst_agg = installments.groupby('SK_ID_CURR').agg({
    'SK_ID_PREV': 'count',                  
    'DAYS_PAST_DUE': ['max', 'mean', 'sum'],
    'PAYMENT_FRACTION': ['mean', 'min'],    
    'AMT_PAYMENT': ['mean', 'sum']          
})

# Flatten columns and merge
inst_agg.columns = ['INSTAL_' + '_'.join(c).strip().upper() for c in inst_agg.columns.values]
df = df.merge(inst_agg, on='SK_ID_CURR', how='left')
del installments, inst_agg # Free RAM

print("3.7 Aggregating credit_card_balance.csv data...")
cc = pd.read_csv('data/credit_card_balance.csv')
cc_agg = cc.groupby('SK_ID_CURR').agg({
    'AMT_BALANCE': ['mean', 'max'],           # Are they carrying high balances?
    'AMT_DRAWINGS_CURRENT': ['sum', 'max'],   # Are they pulling cash out constantly?
    'SK_DPD': ['max', 'sum']                  # Credit Card Days Past Due
})
cc_agg.columns = ['CC_' + '_'.join(c).strip().upper() for c in cc_agg.columns.values]
df = df.merge(cc_agg, on='SK_ID_CURR', how='left')
del cc, cc_agg # Crucial for saving RAM

print("3.8 Aggregating POS_CASH_balance.csv data...")
pos = pd.read_csv('data/POS_CASH_balance.csv')
pos_agg = pos.groupby('SK_ID_CURR').agg({
    'SK_DPD': ['max', 'mean'],                # Point of Sale Days Past Due
    'CNT_INSTALMENT_FUTURE': ['mean', 'max']  # How many future installments do they owe?
})
pos_agg.columns = ['POS_' + '_'.join(c).strip().upper() for c in pos_agg.columns.values]
df = df.merge(pos_agg, on='SK_ID_CURR', how='left')
del pos, pos_agg # Crucial for saving RAM
# --- PHASE 2: CORE FEATURE ENGINEERING ---

print("4. Engineering financial ratios...")
df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)

# Financial Pressure Ratios
df['CREDIT_TO_INCOME_RATIO'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
df['ANNUITY_TO_INCOME_RATIO'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
df['CREDIT_TO_ANNUITY_RATIO'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
df['DAYS_EMPLOYED_PERCENT'] = df['DAYS_EMPLOYED'] / df['DAYS_BIRTH']

# Family Dynamics Ratios
df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (1 + df['CNT_CHILDREN'])
df['CREDIT_PER_CHILD'] = df['AMT_CREDIT'] / (1 + df['CNT_CHILDREN'])

# Drop empty columns
df = df.dropna(thresh=0.7*len(df), axis=1)

# --- PHASE 3: CATEGORICAL OPTIMIZATION ---

print("5. Converting text to native categories...")
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].astype('category')

# --- PHASE 4: PREPARATION & TRAINING ---

print("6. Preparing matrices...")
y = df['TARGET']
# SK_ID_CURR is an ID, not a predictive feature. We must drop it so the model doesn't learn noise.
X = df.drop(columns=['TARGET', 'SK_ID_CURR'])
del df 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

print("7. Starting Advanced XGBoost Training...")
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,
    learning_rate=0.03,         
    n_estimators=1500,          
    max_depth=6,                
    subsample=0.8,              
    colsample_bytree=0.8,       
    tree_method='hist',         
    enable_categorical=True,    
    early_stopping_rounds=50,   
    random_state=42,
    n_jobs=-1                   
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=50  # Print progress every 50 trees to watch the climb
)

# --- PHASE 5: EVALUATION & EXPORT ---

print("8. Evaluating final architecture...")
y_pred_probs = model.predict_proba(X_test)[:, 1]
score = roc_auc_score(y_test, y_pred_probs)

print("-" * 40)
print(f"ULTIMATE PIPELINE ROC AUC Score: {score:.4f}")
print("-" * 40)

model.save_model('credit_xgboost_model.json')
with open('model_columns.json', 'w') as f:
    json.dump(list(X.columns), f)