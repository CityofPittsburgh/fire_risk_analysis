# fire_risk_analysis
Modeling structure fire risk to inform fire inspections


## getdata.py

getdata.py scrapes [WPRDC](wprdc.org) for:
* City of Pittsburgh property data ("pittdata.csv")
* City of Pittsburgh parcel data ("parcels.csv")
* Permits, Licenses, and Inspections data ("pli.csv")


## riskmodel.py

riskmodel.py runs the risk prediction model, using:
* the three datasets from WPRDC
* Fire Incident data from PBF

## merger.py

merger.py takes the output of the risk model, and merges each property's risk score with the rest of the property data in pittdata and parcels, sending the output to the BEV directory.