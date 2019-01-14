import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

ph = pd.read_csv("ph_tq.csv").dropna(subset = ["rank"])
sr = pd.read_csv("sr_tq.csv").dropna(subset = ["rank"])
vt = pd.read_csv("vt_tq.csv").dropna(subset = ["rank"])
pb = pd.read_csv("pb_tq.csv").dropna(subset = ["rank"])
hb = pd.read_csv("hb_tq.csv").dropna(subset = ["rank"])
fx = pd.read_csv("fx_tq.csv").dropna(subset = ["rank"])
ph = ph.assign(E_Penalty=lambda ph:ph.TotalScore.astype(str).astype(float)-ph.DScore)
sr = sr.assign(E_Penalty=lambda sr:sr.TotalScore.astype(str).astype(float)-sr.DScore)
vt = vt.assign(E_Penalty=lambda vt:vt.TotalScore.astype(str).astype(float)-vt.DScore)
vt = vt.assign(E_Penalty2=lambda vt:vt.TotalScore2.astype(str).astype(float)-vt.DScore2)
pb = pb.assign(E_Penalty=lambda pb:pb.TotalScore.astype(str).astype(float)-pb.DScore)
hb = hb.assign(E_Penalty=lambda hb:hb.TotalScore.astype(str).astype(float)-hb.DScore)
fx = fx.assign(E_Penalty=lambda fx:fx.TotalScore.astype(str).astype(float)-fx.DScore)

plt.scatter(ph.DScore, ph.E_Penalty)
bins = np.linspace(ph[['DScore','E_Penalty']].min().min()-0.1,ph[['DScore','E_Penalty']].max().max()+0.1,50)
plt.hist(ph.DScore,bins, alpha = 0.5, label='ph.DScore')
plt.hist(ph.E_Penalty,bins, alpha = 0.5, label='ph.E-Penalty')
plt.legend(loc='upper left')
plt.xlabel("DScore")
plt.ylabel("Frequncy")
plt.title(r'Pommel Horse TQ Histogram')
plt.savefig('ph_TQ_Hist.png',dpi=300)
plt.boxplot(ph.DScore)
plt.boxplot(sr.DScore)
plt.boxplot(pb.DScore)
plt.boxplot(hb.DScore)
plt.boxplot(fx.DScore)
plt.boxplot(vt.DScore)
plt.boxplot(vt.DScore2)
plt.legend() ##loc='right'

total_data = pd.concat([ph.assign(event=lambda ph:'ph')
                        ,sr.assign(event=lambda sr:'sr')
                        ,pb.assign(event=lambda pb:'pb')
                        ,hb.assign(event=lambda hb:'hb')
                        ,fx.assign(event=lambda fx:'fx')],sort = False)
total_data.to_csv("TQData.csv", index=False)

fig, ax = plt.subplots(1,2)
sns.boxplot(x='event',y = 'DScore', data = total_data, ax = ax[0],palette = 'Set3',fliersize = 3)
sns.boxplot(x='event',y = 'E_Penalty', data = total_data, ax = ax[1],palette= 'Set3',fliersize = 3)
plt.suptitle("Compare D&E Score across 5 Events")
plt.savefig("5EventsCompare.png", dpi = 300)
total_data.groupby('event')[['DScore','E_Penalty']].describe().to_csv("TQSummary.csv",index=True)

bins = np.linspace(sr[['DScore','E_Penalty']].min().min()-0.1,sr[['DScore','E_Penalty']].max().max()+0.1,50)
plt.hist(sr.DScore,bins, alpha = 0.5, label='sr.DScore')
plt.hist(sr.E_Penalty,bins, alpha = 0.5, label='sr.E-Penalty')
plt.legend(loc='upper left')
plt.xlabel("DScore")
plt.ylabel("Frequncy")
plt.title(r'Ring TQ Histogram')
plt.savefig('sr_TQ_Hist.png',dpi=300)


bins = np.linspace(pb[['DScore','E_Penalty']].min().min()-0.1,pb[['DScore','E_Penalty']].max().max()+0.1,50)
plt.hist(pb.DScore,bins, alpha = 0.5, label='pb.DScore')
plt.hist(pb.E_Penalty,bins, alpha = 0.5, label='pb.E-Penalty')
plt.legend(loc='upper left')
plt.xlabel("DScore")
plt.ylabel("Frequncy")
plt.title(r'Parallel Bar TQ Histogram')
plt.savefig('pb_TQ_Hist.png',dpi=300)

bins = np.linspace(hb[['DScore','E_Penalty']].min().min()-0.1,hb[['DScore','E_Penalty']].max().max()+0.1,50)
plt.hist(hb.DScore,bins, alpha = 0.5, label='hb.DScore')
plt.hist(hb.E_Penalty,bins, alpha = 0.5, label='hb.E-Penalty')
plt.legend(loc='upper left')
plt.xlabel("DScore")
plt.ylabel("Frequncy")
plt.title(r'Horizontal Bar TQ Histogram')
plt.savefig('hb_TQ_Hist.png',dpi=300)

bins = np.linspace(vt[['DScore','E_Penalty']].min().min()-0.1,vt[['DScore','E_Penalty']].max().max()+0.1,50)
plt.hist(vt.DScore,bins, alpha = 0.3, label='vt.DScore')
plt.hist(vt.DScore2,bins, alpha = 0.3, label='vt.DScore2')
plt.hist(vt.E_Penalty,bins, alpha = 0.3, label='vt.E-Penalty')
plt.hist(vt.E_Penalty2,bins, alpha = 0.3, label='vt.E-Penalty2')
plt.legend(loc='upper right')
plt.xlabel("DScore")
plt.ylabel("Frequncy")
plt.title(r'Vault Histogram')
plt.savefig('vt_TQ_Hist.png',dpi=300)

bins = np.linspace(fx[['DScore','E_Penalty']].min().min()-0.1,fx[['DScore','E_Penalty']].max().max()+0.1,50)
plt.hist(fx.DScore,bins, alpha = 0.5, label='fx.DScore')
plt.hist(fx.E_Penalty,bins, alpha = 0.5, label='fx.E-Penalty')
plt.legend(loc='upper left')
plt.xlabel("DScore")
plt.ylabel("Frequncy")
plt.title(r'Floor Histogram')
plt.savefig('fx_TQ_Hist.png',dpi=300)


from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import LogisticRegressionCV as LRCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
MinorData = pd.read_csv("/home/yang/GymClean/MinorData.csv")
X_ohkey = pd.get_dummies(X)
y_ohkey = pd.get_dummies(y)
y_ohkey
clf = LRCV(cv = 5, random_state=0).fit(X_ohkey,y_ohkey['Yes'])
clf1 = LR().fit(X_ohkey,y_ohkey['Yes'])
y_pred = clf.predict(X_ohkey)
y_pred1 = clf1.predict(X_ohkey)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_ohkey['Yes'], y_pred)
confusion_matrix(y_ohkey['Yes'], y_pred1)
clf.fit
clf.coef_
clf.get_params()
X_ohkey.columns
MinorCoeff = pd.DataFrame([list(X_ohkey.columns), list(clf.coef_.tolist()[0])], index=['indicator', 'coefficients']).transpose()


MajorData = pd.read_csv("/home/yang/GymClean/MajorData.csv")
X,y = MajorData.iloc[:,1:], MajorData.iloc[:,0]
X_ohkey = pd.get_dummies(X)
y_ohkey = pd.get_dummies(y)
clf = LRCV(cv = 5, random_state=0).fit(X_ohkey,y_ohkey['Yes'])
clf1 = LR().fit(X_ohkey,y_ohkey['Yes'])
y_pred = clf.predict(X_ohkey)
y_pred1 = clf1.predict(X_ohkey)
confusion_matrix(y_ohkey['Yes'], y_pred)
confusion_matrix(y_ohkey['Yes'], y_pred1)
X_ohkey.columns
clf.coef_
MajorCoeff = pd.DataFrame([list(X_ohkey.columns), list(clf.coef_.tolist()[0])], index=['indicator', 'coefficients']).transpose()
MajorCoeff
