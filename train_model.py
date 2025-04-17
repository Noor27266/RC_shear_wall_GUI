import pandas as pd
import pickle
from xgboost import XGBRegressor

# Load the Excel file with original (raw) data
df = pd.read_excel("F:/GUI/Damping Ratio.xlsx")
df.columns = df.columns.str.strip()
df = df.drop(columns=["lw", "hw"])  # Drop normalized columns
df.rename(columns={"lw.1": "lw", "hw.1": "hw", "Ductility Ratio ": "Ductility Ratio"}, inplace=True)

# Split features and target
X = df.drop(columns=["DampingRatio"])
y = df["DampingRatio"]

# Train the model
model = XGBRegressor()
model.fit(X, y)

# Save it to model.pkl
with open("F:/GUI/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
streamlit run f:/GUI/GUI.py