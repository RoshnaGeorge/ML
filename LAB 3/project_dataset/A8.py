import pandas as pd


def mean(data):
    t = 0
    for v in data:
        t += v
    return t / len(data)

def variance(data):
    m = mean(data)
    t = 0
    for v in data:
        t += (v - m) ** 2
    return t / len(data)

def std(data):
    return variance(data) ** 0.5

def ds_stats(df):
    for col in df.columns:
        print(f"\nFeature : {col}")
        print("Mean              :", mean(df[col]))
        print("Variance          :", variance(df[col]))
        print("Standard Deviation:", std(df[col]))


df = pd.read_csv(
    "..\\simulation_500.csv"
)

df = df.select_dtypes(include="number")

ds_stats(df)
