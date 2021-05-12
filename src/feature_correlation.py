from scipy.stats import pearsonr
import numpy as np
import pandas as pd
from contextlib import redirect_stdout
import matplotlib.pyplot as plt
import seaborn as sns

def correlation_heatmap(train):
    correlations = train.corr()

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.show()

df = pd.read_csv("../resource/forestfires.csv")

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