import pandas as pd
from qs2 import label_encode, one_hot_encode

df = pd.read_excel(
    "Lab Session Data.xlsx",
    sheet_name="marketing_campaign"
)

df.columns = df.columns.str.strip()

edf = df.copy()

ee, mp = label_encode(edf["Education"])
edf["Education"] = ee

print("Education Mapping:")
print(mp)

me, cats = one_hot_encode(edf["Marital_Status"])

mdf = pd.DataFrame(
    me,
    columns=[f"Marital_Status_{c}" for c in cats]
)

edf.drop(columns=["Marital_Status"], inplace=True)

edf = pd.concat([edf, mdf], axis=1)

print("\nOriginal Dataset Shape :", df.shape)
print("Encoded Dataset Shape  :", edf.shape)

print("\nOriginal Number of Features :", df.shape[1])
print("Encoded Number of Features  :", edf.shape[1])
