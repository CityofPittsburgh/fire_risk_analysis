# Fire Risk Analysis
This is a set of scripts for a machine learning pipeline to predict structure fire risk and inform fire inspection prioritization decisions. A full technical report can be found [here](http://michaelmadaio.com/Metro21_FireRisk_FinalReport.pdf).

### Authors: 
* Michael Madaio
* Geoffrey Arnold
* Bhavkaran Singh
* Qianyi Hu
* Jason Batts


## Run_Model.sh
Runs all three python scripts listed below in succession.

## getdata.py

Scrapes [WPRDC](https://wprdc.org) for:
* City of Pittsburgh property data ("pittdata.csv")
* City of Pittsburgh parcel data ("parcels.csv")
* Permits, Licenses, and Inspections data ("pli.csv")

## riskmodel.py

Runs the risk prediction model, using:
* the three datasets from WPRDC
* Fire Incident data from PBF (public, aggregated version available at [WPRDC](https://data.wprdc.org/dataset/fire-incidents-in-city-of-pittsburgh).

## merger.py

Takes the output of the risk model, and merges each property's risk score with the rest of the property data in pittdata and parcels, sending the output to the Burgh's Eye View directory for map and dashboard visualization (on a private instance developed for Bureau of Fire inspectors; public version of BEV available [here](https://pittsburghpa.shinyapps.io/BurghsEyeView/?_inputs_&basemap_select=%22OpenStreetMap.Mapnik%22&circumstances_select=null&crash_select=null&dept_select=null&dow_select=null&filter_select=%22%22&fire_desc_select=null&funcarea_select=null&heatVision=0&hier=null&map_bounds=%7B%22north%22%3A40.6035267998859%2C%22east%22%3A-79.5238494873047%2C%22south%22%3A40.290001686076%2C%22west%22%3A-80.4027557373047%7D&map_center=%7B%22lng%22%3A-79.9629625321102%2C%22lat%22%3A40.4467468302211%7D&map_zoom=11&navTab=%22Points%22&offense_select=null&origin_select=null&report_select=%22311%20Requests%22&req.type=null&result_select=null&search=%22%22&status_type=null&times=%5B0%2C24%5D&toggle311=true&toggleArrests=true&toggleBlotter=true&toggleCitations=true&toggleCproj=true&toggleCrashes=false&toggleFires=true&toggleViolations=true&violation_select=null))

## ui.R and server.R

Takes the output of the risk scores, merged with property data, and visualizes them in an R Shiny dashboard, for inspectors and fire chiefs to view property risk levels, by property type, neighborhood, and fire district.

## requirements.txt

All of the packages you'll need to install for the scripts to run
