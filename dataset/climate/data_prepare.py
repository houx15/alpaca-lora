import pandas as pd
import numpy as np


LABEL_SET = ["irrelevant or no opinion on climate change", "strongly disagree with climate change", "slightly disagree with climate change", "neutral to climate change", "slightly agree with climate change", "strongly agree with climate change"]


def union_mtscale(row):
    if row["michael"] in LABEL_SET:
        return row["michael"]
    if row["label-scale"] in LABEL_SET:
        return row["label-scale"]
    if row["tevin"] in LABEL_SET:
        return row["tevin"]
    return np.nan

def union_mt(row):
    if row["michael"] in LABEL_SET:
        return row["michael"]
    if row["tevin"] in LABEL_SET:
        return row["tevin"]
    return np.nan

def union_am(row):
    if row["michael"] in LABEL_SET:
        return row["michael"]
    if row["allan"] in LABEL_SET:
        return row["allan"]
    return np.nan

def merge_all(row):
    row = row[["allan", "michael", "shameek", "tevin", "label-scale"]]
    count = row.value_counts()
    if count.size == 0:
        return np.nan
    if count[0] > 1:
        return count.index[0]
    return np.nan

def merge():
    scale_df = pd.read_csv("2023-02-17-climate-change-scale.csv", header=0, usecols=["tweet_id", "scale"])
    scale_df = scale_df.rename(columns={"scale": "label-scale"})
    replaced_labels = ["Irrelevant", "Strongly disagree with climate change", "Slightly disagree with climate change", "Neutral to climate change", "Slightly agree with climate change", "Strongly agree with climate change"]
    scale_df["label-scale"].replace(replaced_labels, LABEL_SET, inplace=True)
    
    previous_df = pd.read_csv("2023-03-03-climate-change-aggregated.csv", header=0)
    new_df = previous_df.merge(scale_df, how="left", on="tweet_id", sort=False)
    
    print(new_df, new_df.columns)
    
    # new_df["label-mscale"] = new_df.apply(lambda x: x["michael"] if x["michael"] == x["label-scale"] else np.nan, axis=1)
    # new_df["label-tscale"] = new_df.apply(lambda x: x["tevin"] if x["tevin"] == x["label-scale"] else np.nan, axis=1)
    new_df["label-mt-union"] = new_df.apply(union_mt, axis=1)
    new_df["label-am-union"] = new_df.apply(union_am, axis=1)
    new_df["label-mtscale"] = new_df.apply(union_mtscale, axis=1)
    new_df["label-all-scale"] = new_df.apply(merge_all, axis=1)
    print(new_df)
    print(new_df["label-mtscale"].count())
    print(new_df["label-all-scale"].count())
    new_df.to_csv("2023-03-04-climate-change-aggregated.csv")
    new_df.to_excel("2023-03-04-climate-change-aggregated.xlsx")
    

if __name__ =="__main__":
    merge()