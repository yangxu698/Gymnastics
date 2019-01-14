from tabula import read_pdf
import pandas as pd
import numpy as np

data_aa = read_pdf("/home/yang/Lucid/C73G_All-Around_Individual_Results_MenSenC1.pdf", pages = "all")
data_ph_1 = read_pdf("/home/yang/Lucid/C73I_Pommel Horse_Results_MenSenC1.pdf", pages = "1-4")
data_ph_20.columns = data_ph_10.columns
data_ph = data_ph_10.append(data_ph_20)
data_ph.columns = ['rank', 'ID', 'Full Name','NOC','DScore', 'EScore','Penalty', 'TotalScore']
data_ph = data_ph.reset_index(drop=True)
data_ph.to_csv("ph_tq.csv")
data_ph = pd.read_csv("ph_tq.csv")
data_ph.isnull().sum()
data_ph.describe()
index = data_ph['NOC'].isnull()
index.describe()
data_ph.loc[index,'NOC'] = data_ph.loc[index,"Full Name"].str.split(" ").str[-1]
data_ph.loc[index,'Full Name'] = data_ph.loc[index,"Full Name"].str[:-4]
data_ph.loc[index,]
rank_array = pd.DataFrame(data_ph.loc[index,['rank','ID']]`.str.split(' ').tolist(), columns = ['rank', 'ID'])
rank_array
data_temp = data_ph
data_temp.loc[index,'NOC'] = data_temp.loc[index,'Full Name'].str.split(" ").str[-1]
data_ph.loc[index,['rank', 'ID']] = rank_array

data_fx = read_pdf("/home/yang/Lucid/C73I_Floor Exercise_Results_MenSenC1.pdf", pages = "all")
data_fx.columns = ['rank', 'ID', 'Full Name','NOC','DScore', 'EScore','Penalty', 'TotalScore']
data_fx = data_fx.dropna(subset=['DScore'])
data_fx.to_csv("fx_tq.csv", index = False)



data_sr = read_pdf("/home/yang/Lucid/C73I_Rings_Results_MenSenC1.pdf", pages = "all")
data_sr.columns = ['rank', 'ID', 'Full Name','NOC','DScore', 'EScore','Penalty', 'TotalScore']
data_sr = data_sr.dropna(subset=['DScore'])
data_sr.to_csv("sr_tq.csv", index = False)
data_sr = pd.read_csv("sr_tq.csv")
index = data_sr['NOC'].isnull()
index.describe()
data_sr.loc[index,'NOC'] = data_sr.loc[index,"Full Name"].str.split(" ").str[-1]
data_sr.loc[index,'Full Name'] = data_sr.loc[index,"Full Name"].str[:-4]
data_sr.loc[index,]



data_vt = read_pdf("/home/yang/Lucid/C73J_Both_Vault_Results_MenSenC1.pdf", pages = "all")
data_vt.head(10)
data_vt.columns = ['rank', 'ID', 'Full Name','NOC','Vault','DScore', 'EScore','Penalty', 'TotalScore','FinalScore']
data_vt = data_vt.dropna(subset=['DScore'])
data_vt.to_csv("vt_tq.csv", index = False)
data_vt = pd.read_csv("vt_tq.csv")
data_vt

index = np.arange(0,73,2)
temp = data_vt.loc[index+1,['Vault', 'DScore', 'EScore', 'Penalty','TotalScore']]
temp.columns = ['Vault2', 'DScore2', 'EScore2', 'Penalty2','TotalScore2']
temp = temp.reset_index(drop= True)
temp
data_vt = pd.concat([data_vt.loc[index,].reset_index(drop = True), temp],axis = 1)
data_vt
data_vt.to_csv("vt_tq.csv", index = False)


data_pb = read_pdf("/home/yang/Lucid/C73I_Parallel Bars_Results_MenSenC1.pdf", pages = "all")
data_pb.columns = ['rank', 'ID', 'Full Name','NOC','DScore', 'EScore','Penalty', 'TotalScore']
data_pb = data_pb.dropna(subset=['DScore'])
data_pb.to_csv("pb_tq.csv", index = False)
data_pb = pd.read_csv("pb_tq.csv")
index = data_pb['NOC'].isnull()
index.describe()
data_pb.loc[index,'NOC'] = data_pb.loc[index,"Full Name"].str.split(" ").str[-1]
data_pb.loc[index,'Full Name'] = data_pb.loc[index,"Full Name"].str[:-4]
data_pb.to_csv("pb_tq.csv", index = False)
data_pb


data_hb = read_pdf("/home/yang/Lucid/C73I_Horizontal Bar_Results_MenSenC1.pdf", pages = "all")
data_hb.columns = ['rank', 'ID', 'Full Name','NOC','DScore', 'EScore','Penalty', 'TotalScore']
data_hb = data_hb.dropna(subset=['DScore'])
data_hb.to_csv("hb_tq.csv", index = False)
data_hb = pd.read_csv("hb_tq.csv")
index = data_hb['NOC'].isnull()
index.describe()
data_hb.loc[index,'NOC'] = data_hb.loc[index,"Full Name"].str.split(" ").str[-1]
data_hb.loc[index,'Full Name'] = data_hb.loc[index,"Full Name"].str[:-4]
data_hb.to_csv("hb_tq.csv", index = False)

data_tf = read_pdf("/home/yang/Lucid/C73C_Results_Team_MenSenC4.pdf", pages = "all",head = False)
data_tf.drop(index=[[1,2,3]])
data_tf.iloc[3:,:]
writer = pd.ExcelWriter('TQ.xlsx')
data_tf.to_excel(writer, 'Sheet1')
writer.save()
