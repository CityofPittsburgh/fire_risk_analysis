#!/bin/bash
# Script name : Run_Model.sh
# Description : To run the various python scripts which generate the fire scores for commercial buildings
# Author : Geoffrey Arnold
# Date : 12/29/2017
export DISPLAY=:0.0
python ./FirePred/getdata.py && python ./FirePred/riskmodel.py && python ./FirePred/merger.py
cp /opt/shiny-server/samples/sample-apps/PBF/Fire_Map/fire_risk_nonres.csv /opt/shiny-server/samples/sample-apps/PBF/FireRisk_Dashboard/fire_risk_nonres.csv