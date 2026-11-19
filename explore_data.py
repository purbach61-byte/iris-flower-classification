import pandas as pd
from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
df = iris.frame
df["species"] = df["target"].map(dict(enumerate(iris.target_names)))

print("===== IRIS FLOWER DATASET =====")
print("Total records:", len(df))
print("Dataset shape:", df.shape)
print("Features:", iris.feature_names)
print("Species:", [str(s) for s in iris.target_names])
print("\nFirst five records:")
print(df.head())

print("\nDataset info:")
df.info()

print("\nStatistical summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

print("\nRecords per species:")
print(df["species"].value_counts())