__author__ = 'mmadaio'

import pandas as pd

# Set input and output folders
output = "/opt/shiny-server/samples/sample-apps/PBF/Fire_Map/"

# Read fire risk data
risk = pd.read_csv(open('/home/linadmin/FirePred/datasets/Results.csv','rU'), low_memory = False)

# Read City of Pittsburgh property data
pitt = pd.read_csv("/home/linadmin/FirePred/datasets/pittdata.csv", low_memory = False)

# Read City of Pittsburgh parcel data
parcels = pd.read_csv("/home/linadmin/FirePred/datasets/parcels.csv", low_memory = False)

# Format addresses in Pitt property data
pitt['HouseNum'] = pitt['PROPERTYHOUSENUM'].astype(str).map(lambda x: x.rstrip('.0'))
pitt['NewAddress'] = pitt['HouseNum'].astype(str) + " " + pitt['PROPERTYADDRESS']

# Merge risk to pitt
pitt_risk = pd.merge(left=risk,right=pitt, how='left', left_on='Address', right_on='NewAddress')

del pitt_risk['NewAddress']
del pitt_risk['HouseNum']
del pitt_risk['Unnamed: 0']

# Merge pitt_risk to parcels
pitt_risk_parcels = pd.merge(left=pitt_risk,right=parcels, how='left', left_on='PARID', right_on='PIN')

# Output to csv
pitt_risk_parcels.to_csv("{0}fire_risk_nonres.csv".format(output), index=False)