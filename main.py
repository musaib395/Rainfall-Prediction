import numpy as np
import pandas as pd
import pickle
import os

# Get correct folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "rainfall in india 1901-2015.csv")

# Load dataset
data = pd.read_csv(csv_path)

# Fill missing values
data = data.fillna(data.mean(numeric_only=True))

# Melt full dataset (IMPORTANT: no filtering)
df = data.melt(
    id_vars=['SUBDIVISION', 'YEAR'],
    var_name='Month',
    value_name='Avg_Rainfall'
)

# Convert Month to number
month_map = {
    'JAN':1,'FEB':2,'MAR':3,'APR':4,'MAY':5,'JUN':6,
    'JUL':7,'AUG':8,'SEP':9,'OCT':10,'NOV':11,'DEC':12
}
df['Month'] = df['Month'].map(month_map)

# Encode states
df['SUBDIVISION'] = df['SUBDIVISION'].astype('category')
df['State_Code'] = df['SUBDIVISION'].cat.codes

# Clean rainfall column
df['Avg_Rainfall'] = pd.to_numeric(df['Avg_Rainfall'], errors='coerce')
df = df.dropna()

# Features and target
X = df[['State_Code', 'YEAR', 'Month']].values
y = df['Avg_Rainfall'].values

# Train model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X, y)

# Save model + mapping
mapping = df[['SUBDIVISION','State_Code']].drop_duplicates()

model_path = os.path.join(BASE_DIR, "model.pkl")
with open(model_path, "wb") as f:
    pickle.dump((model, mapping), f)

print("✅ Model with state created successfully")