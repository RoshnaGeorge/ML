import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_excel("Lab Session Data.xlsx",sheet_name="marketing_campaign")
f = df["Income"].dropna() #na also removes NaN values.
hist, bins = np.histogram(f, bins=10) 
print("hist count:", hist)
print("Bin Edges:", bins)
print("mean:", np.mean(f))
print("var:", np.var(f))
#extra
plt.hist(f, bins=10)
plt.xlabel("income")
plt.ylabel("freq")
plt.show()
