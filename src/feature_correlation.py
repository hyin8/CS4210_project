from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import seaborn as sns

#Input: The name of the col you want removed in a list []
#Output: DataFrame with the col removed
def remove_col_by_name(df, cols):
	for col in cols:
		df = df.drop(col,axis=1)
	return df

def correlation_heatmap(train):
    correlations = train.corr()

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show()

col_remove = ["X","Y"]
df = pd.read_csv("../resource/forestfires.csv")
df = remove_col_by_name(df,col_remove)

#export to file
with open('../export/pearson_cof.csv', 'w') as f:
	with redirect_stdout(f):
		print(df.corr("pearson").to_csv())

with open('../export/spearman_cof.csv', 'w') as f:
	with redirect_stdout(f):
		print(df.corr("spearman").to_csv())
		
with open('../export/kendall_cof.csv', 'w') as f:
	with redirect_stdout(f):
		print(df.corr("kendall").to_csv())

correlation_heatmap(df)