__author__ = 'mmadaio'

import pandas as pd
import os

# Set input and output folders
output = "/var/www/html/"
root = "/home/linadmin/FirePred/"
dataset_path = "{0}datasets/".format(root)

# Read fire risk data
risk = pd.read_csv(os.path.join(dataset_path, 'Results.csv'), low_memory = False)

# Read City of Pittsburgh property data
pitt = pd.read_csv(os.path.join(dataset_path, "pittdata.csv"), low_memory = False)

# Read City of Pittsburgh parcel data
parcels = pd.read_csv(os.path.join(dataset_path, "parcels.csv"), low_memory = False)

# Format addresses in Pitt property data
pitt['HouseNum'] = pitt['PROPERTYHOUSENUM'].astype(str).map(lambda x: x.rstrip('.0'))
pitt['NewAddress'] = pitt['HouseNum'].astype(str) + " " + pitt['PROPERTYADDRESS']
pitt = pitt[(pitt.PROPERTYCITY == 'PITTSBURGH')]

# Merge risk to pitt
pitt_risk = pd.merge(left=risk,right=pitt, how='left', left_on='Address', right_on='NewAddress')

del pitt_risk['NewAddress']
del pitt_risk['HouseNum']
del pitt_risk['Unnamed: 0']

# Merge pitt_risk to parcels
pitt_risk_parcels = pd.merge(left=pitt_risk,right=parcels, how='left', left_on='PARID', right_on='PIN')

# Filter out everything but properties in Pittsburgh Municipality
pitt_risk_parcels = pitt_risk_parcels[pitt_risk_parcels['MUNIDESC'].str.contains("Ward|Ingram|Wilkinsburg",na=False)]

# Select only needed columns
pitt_risk_parcels_final = pitt_risk_parcels[['PARID','Address','PROPERTYCITY','geo_name_zipcode','muni_desc','Pgh_FireDistrict','geo_name_nhood','use_desc','state_desc','CLASSDESC','RiskScore','x','y']]

# Output to csv
pitt_risk_parcels_final.to_csv(os.path.join(output, "fire_risk_nonres.csv"), index=False)
