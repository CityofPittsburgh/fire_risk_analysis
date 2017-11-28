__author__ = 'mmadaio'

import pandas as pd

folder = "datasets/"

risk = pd.read_csv(open('{0}Results_address.csv'.format(folder),'rU'), low_memory = False)
pitt = pd.read_csv("{0}pittdata.csv".format(folder), low_memory = False)
parcels = pd.read_csv("{0}parcels.csv".format(folder), low_memory = False)

pitt['HouseNum'] = pitt['PROPERTYHOUSENUM'].astype(str).map(lambda x: x.rstrip('.0'))
print pitt['HouseNum']

pitt['NewAddress'] = pitt['HouseNum'].astype(str) + " " + pitt['PROPERTYADDRESS']
print pitt['NewAddress']

pitt_risk = pd.merge(left=pitt,right=risk, how='left', left_on='NewAddress', right_on='Address')
pitt_risk_parcels = pd.merge(left=parcels,right=pitt_risk, how='left', left_on='PIN', right_on='PARID')

pitt_risk.to_csv("pbf_full.csv", index=False)

## Output pca + risk --> pcafire.csv
## Output pca + risk + inspections --> pitt_risk_inspections.csv