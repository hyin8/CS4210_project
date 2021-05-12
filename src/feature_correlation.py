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

#Input : DataFrame
#Ouput : DataFrame with month and day change to numbers
def translateMonthAndDay(df):
	month_dict = {'jan':1,'feb':2,'mar':3,'apr':4,'may':5,'jun':6,
              'jul':7,'aug':8,'sep':9,'oct':10,'nov':11,'dec':12,}
	day_dict = {'mon':1,'tue':2,'wed':3,'thu':4,'fri':5,'sat':6,'sun':7}
	df.month = df.month.map(month_dict)
	df.day = df.day.map(day_dict)
	return df

def correlation_heatmap(train, method):
	correlations = train.corr(method)
	fig, ax = plt.subplots(figsize=(10,10))
	sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
	plt.title(method)
	plt.show()

col_remove = []
df = pd.read_csv("../resource/forestfires.csv")
df = remove_col_by_name(df,col_remove)
df = translateMonthAndDay(df)
print(df)
#export to file
# with open('../export/pearson_cof.csv', 'w') as f:
# 	with redirect_stdout(f):
# 		print(df.corr("pearson").to_csv())

# with open('../export/spearman_cof.csv', 'w') as f:
# 	with redirect_stdout(f):
# 		print(df.corr("spearman").to_csv())
		
# with open('../export/kendall_cof.csv', 'w') as f:
# 	with redirect_stdout(f):
# 		print(df.corr("kendall").to_csv())

methods = ["pearson","spearman","kendall"]

for m in methods:
	correlation_heatmap(df,m)