#### Created as part of the Metro21 Fire Risk Analysis project
#### In partnership with the City of Pittsburgh's Department of Innovation and Performance, and the Pittsburgh Bureau of Fire

# Authors:
#   Bhavkaran Singh
#   Michael Madaio
#   Qianyi Hu
#   Nathan Kuo
#   Palak Narang
#   Jeffrey Chen
#   Fangyan Chen


#importing relevant libraries
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import sqlalchemy as sa
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pandas as pd
from sklearn import datasets, linear_model, cross_validation, grid_search
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.grid_search import GridSearchCV
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier
import datetime
from dateutil.relativedelta import relativedelta
import os



# Turn off pandas chained assignment warning
pd.options.mode.chained_assignment = None  # default='warn'

# =============================#1: CLEAN PLI & PITT DATA========================
# create directory paths for opening files
curr_path = os.path.dirname(os.path.realpath(__file__))
dataset_path = os.path.join(curr_path, "datasets/")

# Reading plidata
plidata = pd.read_csv(os.path.join(dataset_path, "pli.csv"),encoding = 'utf-8',dtype={'STREET_NUM':'str','STREET_NAME':'str'}, low_memory=False)
#Reading city of Pittsburgh dataset
pittdata = pd.read_csv(os.path.join(dataset_path, "pittdata.csv"),dtype={'PROPERTYADDRESS':'str','PROPERTYHOUSENUM':'str','CLASSDESC':'str'}, low_memory=False)


#removing all properties outside Pittsburgh, Wilkinsburg, and Ingram
#pittdata = pittdata[(pittdata.PROPERTYCITY == 'Pittsburgh') & (pittdata.PROPERTYCITY == 'Wilkinsburg') & (pittdata.PROPERTYCITY == 'Ingram')]
pittdata = pittdata[(pittdata.PROPERTYCITY == 'PITTSBURGH')]# & (pittdata.PROPERTYCITY == 'WILKINSBURG') & (pittdata.PROPERTYCITY == 'INGRAM')]

#removing extra whitespaces
plidata['STREET_NAME'] = plidata['STREET_NAME'].str.strip()
plidata['STREET_NUM'] = plidata['STREET_NUM'].str.strip()

#removing residential data
pittdata = pittdata[pittdata.CLASSDESC!='RESIDENTIAL']
pittdata = pittdata[pittdata.PROPERTYHOUSENUM!= '0']
pittdata = pittdata[pittdata.PROPERTYADDRESS!= '']



#dropping columns with less than 15% data
pittdata = pittdata.dropna(thresh=4000, axis=1)
pittdata = pittdata.rename(columns={pittdata.columns[0]:'PARID'})
pittdata = pittdata.drop_duplicates()

#merging pli with city of pitt
plipca = pd.merge(pittdata, plidata[['PARCEL','INSPECTION_DATE','INSPECTION_RESULT','VIOLATION']], how = 'left', left_on =['PARID'], right_on = ['PARCEL'] )
plipca = plipca.drop_duplicates()


#dropping nas
newpli = plipca.dropna(subset =['PARCEL','INSPECTION_DATE','INSPECTION_RESULT','VIOLATION'] )
newpli = newpli.reset_index()
newpli = newpli.drop(['index','PARID','index',
    u'PROPERTYCITY', u'PROPERTYSTATE', u'PROPERTYUNIT', u'PROPERTYZIP',
    u'MUNICODE', u'MUNIDESC', u'SCHOOLCODE', u'SCHOOLDESC', u'LEGAL1',
    u'LEGAL2', u'LEGAL3', u'NEIGHCODE',
    u'TAXCODE', u'TAXDESC',
    u'OWNERCODE', u'OWNERDESC', u'CLASS',
    u'CLASSDESC', u'USECODE', u'USEDESC', u'LOTAREA', u'SALEDATE',
    u'SALEPRICE', u'SALECODE', u'SALEDESC', u'DEEDBOOK', u'DEEDPAGE',
    u'CHANGENOTICEADDRESS1', u'CHANGENOTICEADDRESS2',
    u'CHANGENOTICEADDRESS3', u'CHANGENOTICEADDRESS4', u'COUNTYBUILDING',
    u'COUNTYLAND', u'COUNTYTOTAL', u'COUNTYEXEMPTBLDG', u'LOCALBUILDING',
    u'LOCALLAND', u'LOCALTOTAL', u'FAIRMARKETBUILDING', u'FAIRMARKETLAND',
    u'FAIRMARKETTOTAL', u'PARCEL'], axis=1)

newpli = newpli.drop_duplicates()

#converting to datetime
newpli.INSPECTION_DATE = pd.to_datetime(newpli.INSPECTION_DATE)
newpli['violation_year'] = newpli['INSPECTION_DATE'].map(lambda x: x.year)

plipca.SALEPRICE = plipca.SALEPRICE.replace('NaN',0)

#Groups by address and replaces LOTAREA','SALEPRICE','FAIRMARKETLAND','FAIRMARKETBUILDING' by mean
numerical = plipca.groupby( [ "PROPERTYHOUSENUM", "PROPERTYADDRESS"] , as_index=False)[['LOTAREA','SALEPRICE',
    'FAIRMARKETLAND',
    'FAIRMARKETBUILDING']].mean()

# Following blocks of code group by address and get the category with maximum count for each given categorical columns
temp = pd.DataFrame({'count' : plipca.groupby( [ "PROPERTYHOUSENUM", "PROPERTYADDRESS"] ).CLASSDESC.value_counts()}).reset_index()
idx = temp.groupby([ "PROPERTYHOUSENUM", "PROPERTYADDRESS"])['count'].transform(max) == temp['count']
result1 = temp[idx]
result1 = result1.drop_duplicates(subset=[ "PROPERTYHOUSENUM", "PROPERTYADDRESS"], keep = 'last')
del result1['count']

temp = pd.DataFrame({'count' : plipca.groupby( [ "PROPERTYHOUSENUM", "PROPERTYADDRESS"] ).CLASSDESC.value_counts()}).reset_index()
temp.groupby([ "PROPERTYHOUSENUM", "PROPERTYADDRESS"])['count'].transform(max)
idx = temp.groupby([ "PROPERTYHOUSENUM", "PROPERTYADDRESS"])['count'].transform(max) == temp['count']
result1 = temp[idx]
result1 = result1.drop_duplicates(subset=[ "PROPERTYHOUSENUM", "PROPERTYADDRESS"], keep = 'last')
del result1['count']

temp = pd.DataFrame({'count' : plipca.groupby( [ "PROPERTYHOUSENUM", "PROPERTYADDRESS"] ).SCHOOLDESC.value_counts()}).reset_index()
idx = temp.groupby([ "PROPERTYHOUSENUM", "PROPERTYADDRESS"])['count'].transform(max) == temp['count']
result2 = temp[idx]
result2 = result2.drop_duplicates(subset=[ "PROPERTYHOUSENUM", "PROPERTYADDRESS"], keep = 'last')
del result2['count']

temp = pd.DataFrame({'count' : plipca.groupby( [ "PROPERTYHOUSENUM", "PROPERTYADDRESS"] ).OWNERDESC.value_counts()}).reset_index()
idx = temp.groupby([ "PROPERTYHOUSENUM", "PROPERTYADDRESS"])['count'].transform(max) == temp['count']
result3 = temp[idx]
result3 = result3.drop_duplicates(subset=[ "PROPERTYHOUSENUM", "PROPERTYADDRESS"], keep = 'last')
del result3['count']

temp = pd.DataFrame({'count' : plipca.groupby( [ "PROPERTYHOUSENUM", "PROPERTYADDRESS"] ).MUNIDESC.value_counts()}).reset_index()
idx = temp.groupby([ "PROPERTYHOUSENUM", "PROPERTYADDRESS"])['count'].transform(max) == temp['count']
result4 = temp[idx]
result4 = result4.drop_duplicates(subset=[ "PROPERTYHOUSENUM", "PROPERTYADDRESS"], keep = 'last')
del result4['count']

temp = pd.DataFrame({'count' : plipca.groupby( [ "PROPERTYHOUSENUM", "PROPERTYADDRESS"] ).INSPECTION_RESULT.value_counts()}).reset_index()
idx = temp.groupby([ "PROPERTYHOUSENUM", "PROPERTYADDRESS"])['count'].transform(max) == temp['count']
result5 = temp[idx]
result5 = result5.drop_duplicates(subset=[ "PROPERTYHOUSENUM", "PROPERTYADDRESS"], keep = 'last')
del result5['count']

temp = pd.DataFrame({'count' : plipca.groupby( [ "PROPERTYHOUSENUM", "PROPERTYADDRESS"] ).NEIGHCODE.value_counts()}).reset_index()
idx = temp.groupby([ "PROPERTYHOUSENUM", "PROPERTYADDRESS"])['count'].transform(max) == temp['count']
result6 = temp[idx]
result6 = result6.drop_duplicates(subset=[ "PROPERTYHOUSENUM", "PROPERTYADDRESS"], keep = 'last')
del result6['count']

temp = pd.DataFrame({'count' : plipca.groupby( [ "PROPERTYHOUSENUM", "PROPERTYADDRESS"] ).TAXDESC.value_counts()}).reset_index()
idx = temp.groupby([ "PROPERTYHOUSENUM", "PROPERTYADDRESS"])['count'].transform(max) == temp['count']
result7 = temp[idx]
result7 = result7.drop_duplicates(subset=[ "PROPERTYHOUSENUM", "PROPERTYADDRESS"], keep = 'last')
del result7['count']

temp = pd.DataFrame({'count' : plipca.groupby( [ "PROPERTYHOUSENUM", "PROPERTYADDRESS"] ).USEDESC.value_counts()}).reset_index()
idx = temp.groupby([ "PROPERTYHOUSENUM", "PROPERTYADDRESS"])['count'].transform(max) == temp['count']
result8 = temp[idx]
result8 = result8.drop_duplicates(subset=[ "PROPERTYHOUSENUM", "PROPERTYADDRESS"], keep = 'last')
del result8['count']

dfs = [result1,result2,result3,result4,result6,result7,result8,numerical]

pcafinal = reduce(lambda left,right: pd.merge(left,right,on= [ "PROPERTYHOUSENUM", "PROPERTYADDRESS"] ), dfs)

plipca1 = pd.merge(pcafinal, newpli, how = 'left', left_on =[ "PROPERTYHOUSENUM", "PROPERTYADDRESS"], right_on = [ "PROPERTYHOUSENUM", "PROPERTYADDRESS"] )
# ============#1 DONE, ^this is the cleaned dataframe of pli + pitt ============


# =====================#2 CLEAN FIRE INCIDENT DATA====================
#loading fire incidents csvs
fire_pre14 = pd.read_csv(os.path.join(dataset_path, "Fire_Incidents_Pre14.csv"),encoding = 'latin-1',dtype={'street':'str','number':'str'}, low_memory=False)
fire_new = pd.read_csv(os.path.join(dataset_path, "Fire_Incidents_New.csv"),encoding = 'utf-8',dtype={'street':'str','number':'str'}, low_memory=False)

#cleaning columns of fire_pre14
fire_pre14['full.code'] = fire_pre14['full.code'].str.replace('  -',' -')
fire_pre14['st_type'] = fire_pre14['st_type'].str.strip()
fire_pre14['street'] = fire_pre14['street'].str.strip()
fire_pre14['number'] = fire_pre14['number'].str.strip()
fire_pre14['st_type'] = fire_pre14['st_type'].str.replace('AV','AVE')
fire_pre14['street'] = fire_pre14['street'].str.strip() +' ' +fire_pre14['st_type'].str.strip()

# drop irrelevant columns
pre14_drop = ['PRIMARY_UNIT','MAP_PAGE','alm_dttm','arv_dttm','XCOORD','YCOORD',
              'inci_id','inci_type','alarms','st_prefix','st_suffix','st_type','CALL_NO']
for col in pre14_drop:
    del fire_pre14[col]
pre14_drop = [0, 4]
fire_pre14.drop(fire_pre14.columns[pre14_drop], axis=1, inplace=True)

post14_drop = ['alm_dttm','arv_dttm','XCOORD','YCOORD','alarms','inci_type','CALL_NO']
for col in post14_drop:
    del fire_new[col]

#joining both the fire incidents file together
fire_new = fire_new.append(fire_pre14, ignore_index=True)

# removing events that are not fire related
fire_new['descript'] = fire_new['descript'].str.strip()
remove_descript = ['System malfunction, Other',
                   # 'Smoke detector activation, no fire - unintentional']
                   # 'Alarm system activation, no fire - unintentional']
                   'Detector activation, no fire - unintentional', 'Smoke detector activation due to malfunction',
                   'Dispatched & cancelled en route', 'Dispatched & cancelled on arrival',
                   'EMS call, excluding vehicle accident with injury', 'Medical assist, assist EMS crew',
                   'Emergency medical service, other', 'Good intent call, Other', 'Rescue, EMS incident, other',
                   'Medical Alarm Activation (No Medical Service Req)', 'Motor Vehicle Accident with no injuries',
                   'No Incident found on arrival at dispatch address', 'Unintentional transmission of alarm, Other',
                   'Motor vehicle accident with injuries', 'Vehicle accident, general cleanup', 'Power line down',
                   'Person in distress, Other', 'Cable/Telco Wires Down', 'Service Call, other',
                   'Vehicle Accident canceled en route', 'Lock-out', 'False alarm or false call, Other',
                   'Assist police or other governmental agency', 'Special type of incident, Other',
                   'Alarm system sounded due to malfunction', 'Motor vehicle/pedestrian accident (MV Ped)',
                   'Assist invalid ', 'Malicious, mischievous false call, Other', 'Accident, potential accident, Other',
                   'Assist invalid', 'EMS call, party transported by non-fire agency', 'Rescue or EMS standby',
                   'Public service assistance, Other', 'Police matter', 'Lock-in (if lock out , use 511 )',
                   'Sprinkler activation, no fire - unintentional', 'Wrong location',
                   'Local alarm system, malicious false alarm', 'Authorized controlled burning',
                   'Water problem, Other',
                   # 'Smoke or odor removal']
                   'Passenger vehicle fire', 'CO detector activation due to malfunction',
                   'Authorized controlled burning', 'Steam, vapor, fog or dust thought to be smoke', 'Overheated motor',
                   'Local alarm system, malicious false alarm', 'Central station, malicious false alarm',
                   'Public service',
                   # 'Building or structure weakened or collapsed'
                   'Heat detector activation due to malfunction', 'Citizen complaint',
                   'Municipal alarm system, malicious false alarm', 'Sprinkler activation due to malfunction',
                   'Severe weather or natural disaster, Other', 'Water evacuation', 'Breakdown of light ballast',
                   'Extrication of victim(s) from vehicle', 'Flood assessment', 'Telephone, malicious false alarm',
                   'Cover assignment, standby, moveup', 'Road freight or transport vehicle fire']

for descript in remove_descript:
    fire_new = fire_new[fire_new.descript != descript]
fire_new = fire_new[fire_new['full.code'].str.strip()  != '540 - Animal problem, Other']
fire_new = fire_new[fire_new['full.code'].str.strip()  != '5532 - Public Education (Station Visit)']
fire_new = fire_new[fire_new['full.code'].str.strip()  != '353 - Removal of victim(s) from stalled elevator']

#correcting problems with the street column
fire_new['street'] = fire_new['street'].replace(to_replace=', PGH', value='', regex=True)
fire_new['street'] = fire_new['street'].replace(to_replace=', P', value='', regex=True)
fire_new['street'] = fire_new['street'].replace(to_replace=',', value='', regex=True)
fire_new['street'] = fire_new['street'].replace(to_replace='#.*', value='', regex=True)
fire_new['street'] = fire_new['street'].str.strip()
fire_new['number'] = fire_new['number'].str.strip()

#converting to date time and extracting year
fireDate, fireTime = fire_new['CALL_CREATED_DATE'].str.split(' ', 1).str
fire_new['CALL_CREATED_DATE']= fireDate
fire_new['CALL_CREATED_DATE'] = pd.to_datetime(fire_new['CALL_CREATED_DATE'])
fire_new['fire_year'] = fire_new['CALL_CREATED_DATE'].map(lambda x: x.year)

#removing all codes with less than 20 occurences
for col,val in fire_new['full.code'].value_counts().iteritems():
    if val <20 and col[0]!= '1':
        fire_new = fire_new[fire_new['full.code'] != col]

fire_new = fire_new.drop_duplicates()
# ============#2 DONE, ^this is the cleaned data frame of fire incident ============


# =====================#3 JOIN TWO DATASETS AND FINAL CLEANING =====================
#joining plipca with fireincidents
pcafire = pd.merge(plipca1, fire_new, how = 'left', left_on =['PROPERTYADDRESS','PROPERTYHOUSENUM'],
        right_on = ['street','number'])

# making the fire column with all type 100s as fires
pcafire['fire'] = pcafire['full.code'].astype(str).str[0]
pcafire.loc[pcafire.fire == '1', 'fire'] = 'fire'
pcafire.loc[pcafire.fire != 'fire', 'fire'] = 'No fire'
pcafire['full.code'][pcafire['fire'] == 'fire'] = None

#Fire occured after inspection
pcafire1 = pcafire[(pcafire.CALL_CREATED_DATE >= pcafire.INSPECTION_DATE )]
pcafire1 = pcafire[(pcafire.CALL_CREATED_DATE >= pcafire.INSPECTION_DATE )]
pcafire1 = pcafire1[pd.notnull(pcafire1.INSPECTION_DATE)]

#checking if violation is in the same year as the fire and keeping only those
pcafire2 = pcafire1[(pcafire1.violation_year == pcafire1.fire_year)]

#joining all rows with no pli violations
fire_nopli = pd.concat([fire_new, pcafire2[['number','street','CALL_CREATED_DATE','full.code','response_time','fire_year']], pcafire2[['number','street','CALL_CREATED_DATE','full.code','response_time','fire_year']]]).drop_duplicates(keep=False)
pcafire_nopli = pd.merge(pcafinal, fire_nopli, how = 'left', left_on =['PROPERTYADDRESS','PROPERTYHOUSENUM'],
        right_on = ['street','number'])

pcafire_nopli['fire'] = pcafire_nopli['full.code'].astype(str).str[0]
pcafire_nopli.loc[pcafire_nopli.fire == '1', 'fire'] = 'fire'
pcafire_nopli.loc[pcafire_nopli.fire != 'fire', 'fire'] = 'No fire'
pcafire_nopli['full.code'][pcafire_nopli['fire'] == 'fire'] = None

#combined_df is the final file
combined_df  = pcafire_nopli.append(pcafire2, ignore_index=True)
# ================= #3 DONE, ^this is final cleaned dataframe ===================


# =========================== #4 PREPARE DATA =============================
# this part splits the data into training and testing, then coverts the log
# of incident/inspection violation data and convert it to building addresses

#Removing vacant commerical land
combined_df = combined_df[combined_df.USEDESC!= 'VACANT COMMERCIAL LAND']

#converting back to 1 and 0
combined_df['fire'] = combined_df['fire'].map({'fire': 1, 'No fire': 0})

#one hot encoding the features
ohe9 = pd.get_dummies(combined_df['VIOLATION'])
ohe8 = pd.get_dummies(combined_df['full.code'])
ohe10 = pd.get_dummies(combined_df['INSPECTION_RESULT'])

#concatenating the features together
combined_df1 = pd.concat([combined_df[['PROPERTYADDRESS','PROPERTYHOUSENUM','CALL_CREATED_DATE','fire','fire_year']],ohe8,ohe9,ohe10], axis=1)


#PREPARING THE TESTING DATA (final 6 months of data)
cutoff = datetime.datetime.now() - relativedelta(months=6)
cutoffdate = cutoff.strftime("%m/%d/%Y")


testdata = combined_df1[combined_df1.CALL_CREATED_DATE > cutoffdate]
testdata2 = testdata.groupby( [ "PROPERTYHOUSENUM", "PROPERTYADDRESS",'CALL_CREATED_DATE','fire_year'] ).sum().reset_index()
del testdata['CALL_CREATED_DATE']
del testdata['fire_year']
#testdata2 = testdata.groupby( [ "PROPERTYHOUSENUM", "PROPERTYADDRESS"] ).sum().reset_index() #,'CALL_CREATED_DATE','fire_year'
testdata2.loc[testdata2.fire != 0, 'fire'] = 1

nofire2017 = pd.concat([pcafinal[["PROPERTYHOUSENUM","PROPERTYADDRESS"]], testdata2[["PROPERTYHOUSENUM","PROPERTYADDRESS"]],testdata2[["PROPERTYHOUSENUM","PROPERTYADDRESS"]]]).drop_duplicates(keep=False)

testdata2 = testdata2.append(nofire2017, ignore_index=True)
testdata2 = testdata2.fillna(0)

test_data = pd.merge(testdata2,pcafinal, on = ["PROPERTYHOUSENUM", "PROPERTYADDRESS"], how = 'left')
#test_data.fire.value_counts()

#One hot encoding the features for the test set
ohe1 = pd.get_dummies(test_data['CLASSDESC'])
ohe2 = pd.get_dummies(test_data['SCHOOLDESC'])
ohe3 = pd.get_dummies(test_data['OWNERDESC'])
ohe4 = pd.get_dummies(test_data['MUNIDESC'])
ohe5 = pd.get_dummies(test_data['NEIGHCODE'])
ohe6 = pd.get_dummies(test_data['TAXDESC'])
ohe7 = pd.get_dummies(test_data['USEDESC'])

state_desc = test_data['CLASSDESC']
school_desc= test_data['SCHOOLDESC']
owner_desc= test_data['OWNERDESC']
muni_desc= test_data['MUNIDESC']
neigh_desc= test_data['NEIGHCODE']
tax_desc= test_data['TAXDESC']
use_desc= test_data['USEDESC']
address= test_data['PROPERTYADDRESS']
housenum= test_data['PROPERTYHOUSENUM']

#Deleting features not required anymore or already one hot encoded for the model
ohe_del = ['CALL_CREATED_DATE','CLASSDESC','SCHOOLDESC','OWNERDESC','MUNIDESC',
           'NEIGHCODE','TAXDESC','USEDESC','fire_year','PROPERTYADDRESS','PROPERTYHOUSENUM']
for col in ohe_del:
    del test_data[col]

#Concatenating everything back together
encoded_testdata = pd.concat([test_data,ohe1,ohe2,ohe3,ohe4,ohe5,ohe6,ohe7], axis=1)

#PREPARING THE TRAINING DATA

#Everything till final 6-month period is training data
traindata1 = combined_df1[combined_df1.CALL_CREATED_DATE <= cutoffdate]

#Combining multiple instances of an address together
traindata = traindata1.groupby( [ "PROPERTYHOUSENUM", "PROPERTYADDRESS",'CALL_CREATED_DATE','fire_year'] ).sum().reset_index()
#Relabeling them
traindata.loc[traindata.fire != 0, 'fire'] = 1

#concatenating non fire, non pca and fire instances together
nofire_train = pd.concat([pcafinal[["PROPERTYHOUSENUM","PROPERTYADDRESS"]], traindata[["PROPERTYHOUSENUM","PROPERTYADDRESS"]],traindata[["PROPERTYHOUSENUM","PROPERTYADDRESS"]]]).drop_duplicates(keep=False)
traindata = traindata.append(nofire2017, ignore_index=True)
traindata = traindata.fillna(0)
train_data = pd.merge(traindata,pcafinal, on = ["PROPERTYHOUSENUM", "PROPERTYADDRESS"], how = 'left')
#train_data.fire.value_counts()

#creating on hot encoded features for the categorical values
ohe1 = pd.get_dummies(train_data['CLASSDESC'])
ohe2 = pd.get_dummies(train_data['SCHOOLDESC'])
ohe3 = pd.get_dummies(train_data['OWNERDESC'])
ohe4 = pd.get_dummies(train_data['MUNIDESC'])
ohe5 = pd.get_dummies(train_data['NEIGHCODE'])
ohe6 = pd.get_dummies(train_data['TAXDESC'])
ohe7 = pd.get_dummies(train_data['USEDESC'])

#deleting the categories
for col in ohe_del:
    del train_data[col]

#concatenating all the created features together
encoded_traindata = pd.concat([train_data,ohe1,ohe2,ohe3,ohe4,ohe5,ohe6,ohe7], axis=1)

#converting to array and reshaping the data to prep for model
fireVarTrain = encoded_traindata['fire']
del encoded_traindata['fire']
X_train = np.array(encoded_traindata)
y_train = np.reshape(fireVarTrain.values,[fireVarTrain.shape[0],])

#converting to array and reshaping the data to prep for model
fireVarTest = encoded_testdata['fire']
del encoded_testdata['fire']
X_test = np.array(encoded_testdata)
y_test = np.reshape(fireVarTest.values,[fireVarTest.shape[0],])
# ================= #4 DONE, ^final training and testing data ===================


# =========================== #5 MODEL & PREDICTION =============================
#The XG Boost model
#Grid Search was taking too long a time to run hence did hyperparameter tuning manually and arrived
#at the below parameters fiving the most optimal result
model = XGBClassifier( learning_rate =0.13,
        n_estimators=1500,
        max_depth=5,min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective= 'binary:logistic',
        nthread=4,
        seed=27)
model.fit(X_train, y_train)
pred = model.predict(X_test)
real = y_test
cm = confusion_matrix(real, pred)
print confusion_matrix(real, pred)

from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(real, pred)

fpr, tpr, thresholds = metrics.roc_curve(y_test, pred, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

acc = 'Accuracy = {0} \n \n'.format(float(cm[0][0] + cm[1][1])/len(real))
kapp = 'kappa score = {0} \n \n'.format(kappa)
auc = 'AUC Score = {0} \n \n'.format(metrics.auc(fpr, tpr))
recall = 'recall = {0} \n \n'.format(tpr[1])
precis = 'precision = {0} \n \n'.format(float(cm[1][1])/(cm[1][1]+cm[0][1]))

print acc
print kapp
print auc
print recall
print precis



### Write model performance to log file:
log_path = os.path.join(curr_path, "log/")

with open('{0}ModelPerformance_{1}.txt'.format(log_path, datetime.datetime.now()), 'a') as log_file:
    log_file.write("Confusion Matrix: \n \n")
    for item in cm:
        print>>log_file, item
    log_file.write("Model performance metrics: \n \n")
    log_file.write(acc)
    log_file.write(kapp)
    log_file.write(auc)
    log_file.write(recall)
    log_file.write(precis)



#Getting the probability scores
predictions = model.predict_proba(X_test)
print predictions

addresses = housenum +' '+ address

#Addresses with fire and risk score
risk = []
for row in predictions:
    risk.append(row[1])

cols = {"Address": addresses, "Fire":pred,"RiskScore":risk,"state_desc":state_desc,"school_desc":school_desc,
        "owner_desc":owner_desc,"muni_desc":muni_desc,"neigh_desc":neigh_desc,"tax_desc":tax_desc,"use_desc":use_desc}

Results = pd.DataFrame(cols)

#Writing results to the updating Results.csv
Results.to_csv(os.path.join(dataset_path, "Results.csv"))


# Writing results to a log file
Results.to_csv('{0}Results_{1}.csv'.format(log_path, datetime.datetime.now()))


#Plotting the ROC curve
plt.title('Receiver Operating Characteristic')
plt.plot(fpr[1:], tpr[1:], 'b',
        label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
#plt.show()


png_path = os.path.join(curr_path, "images/")
roc_png = "{0}ROC_{1}.png".format(png_path, datetime.datetime.now())
plt.savefig(roc_png, dpi=150)
plt.clf()   # Clear figure

#Tree model for getting features importance
clf = ExtraTreesClassifier()
imputed_fireVarTrain = fireVarTrain.fillna(method="ffill")
imputed_encoded_traindata = encoded_traindata.fillna(method="ffill")

impute_X = np.array(imputed_encoded_traindata)
impute_y = np.reshape(imputed_fireVarTrain.values,[imputed_fireVarTrain.shape[0],])

clf = clf.fit(impute_X, impute_y)


UsedDf = encoded_traindata
important_features = pd.Series(data=clf.feature_importances_,index=UsedDf.columns)
important_features.sort_values(ascending=False,inplace=True)
#top 20 features
print important_features[0:20]

#Plotting the top 20 features
y_pos = np.arange(len(important_features.index[0:20]))

plt.bar(y_pos,important_features.values[0:20], alpha=0.3)
plt.xticks(y_pos, important_features.index[0:20], rotation = (90), fontsize = 11, ha='left')
plt.ylabel('Feature Importance Scores')
plt.title('Feature Importance')


features_png = "{0}FeatureImportancePlot_{1}.png".format(png_path, datetime.datetime.now())
plt.savefig(features_png, dpi=150)
plt.clf()

important_features[0:50].to_csv('{0}FeatureImportanceList_{1}.csv'.format(log_path, datetime.datetime.now()))



#plt.show()
