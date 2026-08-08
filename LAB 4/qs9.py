import pandas as pd
import numpy as np
from qs8 import mean, std

df = pd.read_excel("Lab Session Data.xlsx",sheet_name="marketing_campaign")
df = df.select_dtypes(include="number")
nm = np.mean(df, axis=0)
ns = np.std(df, axis=0)
for i, col in enumerate(df.columns):
    print("my mean:", mean(df[col]))
    print("numpys:", nm.iloc[i])
    print("my std dev:", std(df[col]))
    print("numpys:", ns.iloc[i])
