import os
import pandas as pd
import numpy  as np
from scipy import stats
from sklearn.svm import SVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

import pyarrow.parquet as pq
path = os.getcwd()
newpath = os.path.join(path, 'dataset.parquet')
os.chdir(newpath)

data_raw = pq.read_table('part-00000-tid-6056519413201330783-119422a4-5c84-4461-ba64-4a5e2a1d4acd-29-c000.snappy.parquet').to_pandas()
data_raw.info()

selector = VarianceThreshold(threshold = 0.2)
## remove column with zero <= 0.2, which means variance <= 0.04 ##
data0 = data_raw.loc[:, data_raw.std() > 0.2]
data0.info()
data_raw['label'].value_counts()
data0 = pd.concat([data_raw['label'], data0], axis = 1)
data0.describe()
del(data_raw)  ## delete the hugy buddy

## column_scale = data0.std().sort_values(ascending=False)[0:15].index
scaler = MinMaxScaler()
## since the detail of features are not given, the scaling is applied to all the features here
## but the categorical features which coded in hot-keys should not be scaled
data0.iloc[:,1:] = (scaler.fit_transform(data0.iloc[:,1:]))
data0.describe()

## pd.read_parquet('part-00000-tid-6056519413201330783-119422a4-5c84-4461-ba64-4a5e2a1d4acd-29-c000.snappy.parquet', engine='fastparquet')
pca = PCA(n_components=0.95)
data_predictor_new = pd.DataFrame( pca.fit_transform(data0.iloc[:,1:]) )
data1 = pd.concat([data0['label'], data_predictor_new], axis = 1)
data1.info()
print(pca.explained_variance_ratio_)
print(pca.singular_values_)

data2 = pd.concat([data1[data1.label == 0].sample(n = 1000), data1[data1.label == 1]])
data2.info()

from sklearn.linear_model import LogisticRegression, SGDClassifier
## from multiprocessing import Pool
## clf = SGDClassifier(max_iter=1000)
clf = SVC(gamma = 'auto')
## X_sample = data0.iloc[:,1:]
## Y_sample = data0.iloc[:,0]
X_sample1 = data2.iloc[:,1:]
Y_sample1 = data2.iloc[:,0]
from imblearn.over_sampling import SMOTE, ADASYN
## sm = SMOTE(random_state=42)
## X_ADsample, Y_ADsample = ADASYN().fit_sample(X_sample, Y_sample)
X_ADsample, Y_ADsample = ADASYN().fit_sample(X_sample1, Y_sample1)
pd.value_counts(Y_ADsample == 1)
sample_weight = np.where(Y_ADsample == 0, 1, 0.6)

## svm_weights = np.concatenate((np.repeat(1,5000), np.repeat(0.80,5000)), axis = 0)
## pool = Pool()
## def solve1(X_ADsample, Y_ADsample):
##     return(clf.fit(X_ADsample,Y_ADsample))
## def solve2(X_ADsample, Y_ADsample):
##     return(clf1.fit(X_ADsample,Y_ADsample))
## SGD_SVC_Result = pool.apply_async(clf.fit,[X_ADsample, Y_ADsample])
## Full_SVC_Result = pool.apply_async(clf2.fit,[X_ADsample, Y_ADsample])
clf.fit(X_ADsample, Y_ADsample, sample_weight = sample_weight)
## clf2.fit(X_ADsample, Y_ADsample, sample_weight= sample_weight)

## clf3 = LogisticRegression(random_state = 0, solver = 'lbfgs').fit(X_ADsample, Y_ADsample)
## clf3.get_params(deep=True)
from sklearn.metrics import classification_report, confusion_matrix
X_test = data1.iloc[:,1:]
Y_test = data1.iloc[:,0]
Y_pred = clf.predict(X_test)
## Y_pred1 = clf2.predict(X_test)

print(classification_report(Y_test, Y_pred))
report = classification_report(Y_test, Y_pred)
with open("classification_report.csv", 'wb') as f:
     f.write(report.encode())
print(confusion_matrix(Y_test, Y_pred))
confusion_matrix = confusion_matrix(Y_test, Y_pred)
np.savetxt("confusion_matrix.csv",confusion_matrix, delimiter = ',')


##  Feature Selection ##

from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV, LogisticRegressionCV
clf4 = LogisticRegressionCV(penalty = 'l2',cv=5, solver = 'lbfgs')
featureSelect = SelectFromModel(clf4, threshold = 0.9)
data5 = pd.concat([data0[data0.label == 0].sample(n = 1000), data0[data0.label == 1]])

X_forFeatureSelect = data5.iloc[:,1:]
Y_forFeatureSelect = data5.iloc[:,0]
while n_features>2:
    featureSelect.threshold += 0.1
    featureSelect.fit(X_forFeatureSelect,Y_forFeatureSelect)
    n_features = featureSelect.transform(X_forFeatureSelect).shape[1]
    n_features
np.array(X_forFeatureSelect.columns)[featureSelect.get_support(indices = True)]
featureSelect.threshold
