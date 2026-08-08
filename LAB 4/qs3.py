import pandas as pd
import numpy as np


d = pd.read_excel("Lab_Session_Data.xlsx", sheet_name="marketing_campaign")
def label(s):
    mp = {}
    
    for x in s:
        if x not in mp:
            mp[x] = len(mp)
    
    return s.map(mp), mp  # code with chatgpt


def onehot(d, col):
    n = d.copy()
    vals = d[col]
    mp = {}
    
    for x in vals:
        if x not in mp:
            mp[x] = len(mp)
    
    for x in mp:
        row = (vals == x).astype(int).tolist()  # code with chatgpt
        n[str(x)] = row  # code with chatgpt
    
    return n


w = d.drop(columns=["ID", "Dt_Customer"]).copy()
m = {}
m["Basic"] = 0
m["2n Cycle"] = 1
m["Graduation"] = 2
m["Master"] = 3
m["PhD"] = 4
e = w.copy()
e["Education"] = e["Education"].map(m)
e = onehot(e, "Marital_Status")
print("Before encoding:", len(w.columns), "columns")
print("After encoding:", len(e.columns), "columns")
print(e.head())