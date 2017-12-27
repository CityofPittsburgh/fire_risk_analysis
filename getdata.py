__author__ = 'mmadaio'

import os
import csv
import requests

pli_url = "https://data.wprdc.org/datastore/dump/4e5374be-1a88-47f7-afee-6a79317019b4"
property_url = "https://data.wprdc.org/dataset/2b3df818-601e-4f06-b150-643557229491/resource/2514a4e4-5842-4dca-aff6-099bcd68482c/download/assessments.csv"
parcel_url = "https://data.wprdc.org/dataset/2536e5e2-253b-4c58-969d-687828bb94c6/resource/4b68a6dd-b7ea-4385-b88e-e7d77ff0b294/download/parcelcentroidaug102016.csv"

pli_response = requests.get(pli_url)
with open(os.path.join("\\datasets", "pli.csv"), 'wb') as f:
    f.write(pli_response.content)

property_response = requests.get(property_url)
with open(os.path.join("\\datasets/", "pittdata.csv"), 'wb') as f:
    f.write(property_response.content)

parcel_response = requests.get(parcel_url)
with open(os.path.join("\\datasets", "parcels.csv"), 'wb') as f:
    f.write(parcel_response.content)