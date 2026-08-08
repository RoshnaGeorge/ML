import pandas as pd

def mean(data):#code with chatgpt
    return sum(data) / len(data)

def variance(data):#code with chatgpt
    m = mean(data)
    return sum((x - m) ** 2 for x in data) / len(data)

def std(data):#code with chatgpt
    return variance(data) ** 0.5

def ds_stats(df): #predefined
    for col in df.columns:
        print("mean:", mean(df[col]))
        print("var:", variance(df[col]))
        print("std dev:", std(df[col]))

if __name__ == "__main__":
    df = pd.read_excel("Lab Session Data.xlsx", sheet_name="marketing_campaign")
    df = df.select_dtypes(include="number")
    ds_stats(df)