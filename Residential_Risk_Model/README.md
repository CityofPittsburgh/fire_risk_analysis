# fire_risk_residential

This project extends the previous [fire_risk_analysis](https://github.com/CityofPittsburgh/fire_risk_analysis) from predicting fire risk in commercial properties to predicting fire risk in residential properties, in order to inform the community risk reduction efforts.

## How to set up
1. Run `getdata.py` to get the following data:
- City of Pittsburgh property data, provided by [WPRDC](https://www.wprdc.org) ("pittdata.csv")
- City of Pittsburgh parcel data, provided by [WPRDC](https://www.wprdc.org)  ("parcels.csv")
- Permits, Licenses, and Inspections data, provided by [WPRDC](https://www.wprdc.org)  ("pli.csv")
- Tax lien data, provided by [WPRDC](https://www.wprdc.org)  ('tax.csv')
- 2012-2016 American Community Survey 5-Year Estimates, including [household income](https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=ACS_16_5YR_B19001&prodType=table) ("income.csv"), [occupancy status](https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=ACS_16_5YR_B25002&prodType=table) ("occupancy.csv"), [year structure built](https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=ACS_16_5YR_B25034&prodType=table) ("yearBuilt.csv") and [year moved in to](https://factfinder.census.gov/faces/tableservices/jsf/pages/productview.xhtml?pid=ACS_16_5YR_B25038&prodType=table) ("yearMovedIn.csv").

2. One additional dataset is used for predicting fire risk scores:
- Fire Incident data from PBF (public, aggregated version available at WPRDC. However, please note that due to privacy concerns, the most     detailed fire incident data that the model is trained on are not publicly accessible, but the aggregated version of the incident data       is available, at the block-level, instead of the address-level. At the moment, this script is not able to run on the aggregated, block-     level data.

3. Run `risk_model_residential.py`.

## Description of other files included in this branch
1. `columns.json`: The `getdata.py` scraps [WPRDC](https://www.wprdc.org) for tax lien data based on the keys defined in `columns.json`. To include more features, check [WPRDC](https://www.wprdc.org) for the description of the tax lien datasets and add additional keys in `columns.json`.
2. `risk_model_residential.ipynb`: The ipython notebook version of `risk_model_residential.py`, with model outputs such as feature importance ranking and ROC curve shown as inline figures.
3. `Allmodels.py`: All models considered during model development.
