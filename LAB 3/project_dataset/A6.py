import pandas as pd
from scipy.spatial import distance
from A4 import minkowski_distance

df = pd.read_csv(
    "..\\simulation_500.csv"
)

df = df.select_dtypes(include="number")

a = df.iloc[0]
b = df.iloc[1]

p = int(input("Enter order (p): "))

d1 = minkowski_distance(a, b, p)
d2 = distance.minkowski(a, b, p)

print("Own Function Distance      :", d1)
print("Scipy Package Distance     :", d2)

if abs(d1 - d2) < 1e-6:
    print("Result: Both distances are equal.")
else:
    print("Result: Distances are different.")
