import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/image/segmentation.data'
df = pd.read_csv(url, header=4, on_bad_lines='skip')

print(100*'-')
print(df.head())

print(100*'-')
print("Columns")
print(100*'-')
print(df.columns)

print(100*'-')
print("Type")
print(100*'-')
print(df.dtypes)

print(100*'-')
print("describe")
print(100*'-')
print(df.describe())


df2 = df.dropna()

sns.displot(df2, bins=30)
plt.show()
