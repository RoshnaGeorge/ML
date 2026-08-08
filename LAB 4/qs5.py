import pandas as pd
import matplotlib.pyplot as plt
from qs4 import minkowski_distance
df = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
df = df.select_dtypes(include="number")
a = df.iloc[0]
b = df.iloc[1]
dists = []
for p in range(1, 11):
    dists.append(minkowski_distance(a, b, p))
print(dists)
plt.plot(range(1, 11), dists, marker='o')
plt.xlabel("p-order:")
plt.ylabel("Minkowski:")
#extra
plt.grid(True)
plt.show()
