import pandas as pd
from scipy.spatial import distance
from qs4 import minkowski_distance
df = pd.read_excel( "Lab Session Data.xlsx",sheet_name="marketing_campaign")
df = df.select_dtypes(include="number")
a = df.iloc[0] #iloc takes full row
b = df.iloc[1]
p = int(input("p:"))
d1 = minkowski_distance(a, b, p)
d2 = distance.minkowski(a, b, p)#using skipy
print("My own :", d1)
print("Scipys :", d2)
if abs(d1 - d2) < 1e-6:
    print("both equal.")
else:
    print("no")
