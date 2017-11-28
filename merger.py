__author__ = 'mmadaio'

import pandas as pd

# Set input and output folders
input = "datasets/"
output = "/opt/shiny-server/samples/sample-apps/PBF/Fire_Map/"

# Read fire risk data
risk = pd.read_csv(open('{0}Results_address.csv'.format(input),'rU'), low_memory = False)

# Read City of Pittsburgh property data
pitt = pd.read_csv("{0}pittdata.csv".format(input), encoding='utf-8-sig', low_memory = False)

# Read City of Pittsburgh parcel data
parcels = pd.read_csv("{0}parcels.csv".format(input), low_memory = False)

# Format addresses in Pitt property data
pitt['HouseNum'] = pitt['PROPERTYHOUSENUM'].astype(str).map(lambda x: x.rstrip('.0'))
pitt['NewAddress'] = pitt['HouseNum'].astype(str) + " " + pitt['PROPERTYADDRESS']

# Merge risk to pitt
pitt_risk = pd.merge(left=pitt,right=risk, how='left', left_on='NewAddress', right_on='Address')

del pitt_risk['NewAddress']
del pitt_risk['HouseNum']
del pitt_risk['Unnamed:0']

# Merge pitt_risk to parcels
pitt_risk_parcels = pd.merge(left=parcels,right=pitt_risk, how='left', left_on='PIN', right_on='PARID')

# Output to csv
pitt_risk_parcels.to_csv("{0}fire_risk_nonres.csv".format(input), index=False)