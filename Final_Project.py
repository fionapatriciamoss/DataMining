from pandas.tools.plotting import scatter_matrix
from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn

'''Data Loading'''
region_data = pd.ExcelFile("University of Wyoming Risk Reports.xlsx")
region= region_data.parse('Denver')
region = pd.DataFrame(region)
target_columns = region[['Cashier No', 'StoreNo', 'Total Risk Factor']].copy()
region = region.drop(['Cashier No', 'StoreNo', 'Total Risk Factor'], axis = 1)
region_corr= region.corr()
columns_heading = region_corr.columns.tolist()

'''Heat Map of correlation Matrix'''
plt.figure(figsize=(25,25))
cmapbhelix = sbn.cubehelix_palette(16,as_cmap=True)
sbn.heatmap(region_corr, vmax=1, square=True,annot=True,cmap=cmapbhelix)
plt.title('Feature Correlation')

'''Dropping Highly Correlated columns'''
drop_columns=[]
indices = np.where(region_corr > 0.7)
for x, y in zip(*indices):
    if x != y and x < y:
        drop_columns.append(region.columns[x])
        indices = [(region.columns[x], region.columns[y])]
        
region= region.drop(drop_columns, axis=1)

no_of_clusters = 5

'''Mean Shift Clustering'''
# X = StandardScaler().fit_transform(region)
# bandwidth = cluster.estimate_bandwidth(X, quantile=0.3)
# models = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)

'''ScatterPlot Matrix for important features'''
scatter_matrix(region, figsize=(50,50))

'''K-Means Clustering'''
models= KMeans(n_clusters=no_of_clusters)
models.fit(region)

'''Accuracy Measure'''
total_risk_factor = target_columns['Total Risk Factor'].tolist()
no_of_items = len(total_risk_factor)/no_of_clusters

true_value_list = [] 
for cluster in range(no_of_clusters-1,-1,-1): 
    temp_list = [] 
    for item_index in range(0, no_of_items): 
        true_value_list.append(cluster) 
    total_risk_factor = total_risk_factor[no_of_items:]

y_true = true_value_list 
y_pred = models.labels_ 

print "Confusion Matrix:\n", confusion_matrix(y_true, y_pred)
print "Accuracy on the basis of confusion matrix:", sum(np.diag(confusion_matrix(y_true, y_pred))),"%"
print "Accuracy Score:", accuracy_score(y_true, y_pred)

