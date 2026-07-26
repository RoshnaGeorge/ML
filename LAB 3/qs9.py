import pandas as pd
import numpy as np
from qs8 import mean, std

df = pd.read_excel(
    "Lab Session Data.xlsx",
    sheet_name="marketing_campaign"
)

df = df.select_dtypes(include="number")

nm = np.mean(df, axis=0)
ns = np.std(df, axis=0)

for i, col in enumerate(df.columns):
    print(f"\nFeature: {col}")

    print("Own Mean      :", mean(df[col]))
    print("NumPy Mean    :", nm.iloc[i])

    print("Own Std Dev   :", std(df[col]))
    print("NumPy Std Dev :", ns.iloc[i])
