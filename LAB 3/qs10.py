import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel(
    "Lab Session Data.xlsx",
    sheet_name="marketing_campaign"
)

f = df["Income"].dropna()

hist, bins = np.histogram(f, bins=10)

print("Histogram Counts:", hist)
print("Bin Edges:", bins)

print("Mean:", np.mean(f))
print("Variance:", np.var(f))

plt.hist(f, bins=10)
plt.title("Histogram of Income")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()
