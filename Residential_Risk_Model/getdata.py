import os
import csv
import requests
import urllib
import json
#import losser.losser as losser

# replace the census_api_key with your own key, which can be requested from https://api.census.gov/data/key_signup.html
census_api_key = "d03d0ccc827db4bb31d2d70eb8409cec154647ea" 

# select variables from American Community Survey data
income_variables= ["B19001_0%.2dE"%i for i in range(1,18)]
income_var_str = ','.join(income_variables)

occupancy_variables= ["B25002_0%.2dE"%i for i in range(1,4)]
occupancy_var_str = ','.join(occupancy_variables)

yearBuilt_variables = ["B25034_0%.2dE"%i for i in range(1,12)]
yearBuilt_variables_str = ','.join(yearBuilt_variables)

yearMovedIn_variables = ["B25038_0%.2dE"%i for i in range(1,16)]
yearMovedIn_variables_str = ','.join(yearMovedIn_variables)


#dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "datasets/")
dir_path = os.path.join(os.path.dirname(os.path.realpath('getdata.py')), "datasets/")
pli_url = "https://data.wprdc.org/datastore/dump/4e5374be-1a88-47f7-afee-6a79317019b4"
property_url = "https://data.wprdc.org/dataset/2b3df818-601e-4f06-b150-643557229491/resource/2514a4e4-5842-4dca-aff6-099bcd68482c/download/assessments.csv"
parcel_url = "https://data.wprdc.org/dataset/2536e5e2-253b-4c58-969d-687828bb94c6/resource/4b68a6dd-b7ea-4385-b88e-e7d77ff0b294/download/parcelcentroidaug102016.csv"
tax_url = "https://data.wprdc.org/api/action/datastore_search_sql?sql=SELECT * from \"65d0d259-3e58-49d3-bebb-80dc75f61245\" WHERE tax_year > 2008"
income_url = "https://api.census.gov/data/2016/acs/acs5?get=%s&for=block%%20group:*&in=state:42%%20county:003&key="%income_var_str + census_api_key
occupancy_url = "https://api.census.gov/data/2016/acs/acs5?get=%s&for=block%%20group:*&in=state:42%%20county:003&key="%occupancy_var_str + census_api_key
yearBuilt_url = "https://api.census.gov/data/2016/acs/acs5?get=%s&for=block%%20group:*&in=state:42%%20county:003&key="%yearBuilt_variables_str + census_api_key
yearMovedIn_url = "https://api.census.gov/data/2016/acs/acs5?get=%s&for=block%%20group:*&in=state:42%%20county:003&key="%yearMovedIn_variables_str + census_api_key


print(dir_path)

print("Getting pli...")
pli_response = requests.get(pli_url)
with open(os.path.join(dir_path, "pli.csv"), 'wb') as f:
    f.write(pli_response.content)

print("Getting pittdata...")
property_response = requests.get(property_url)
with open(os.path.join(dir_path, "pittdata.csv"), 'wb') as f:
    f.write(property_response.content)

print("Getting parcels...")
parcel_response = requests.get(parcel_url)
with open(os.path.join(dir_path, "parcels.csv"), 'wb') as f:
    f.write(parcel_response.content)

print("Getting tax liens...")
tax_response = urllib.urlopen(tax_url)
tax_dict = json.loads(tax_response.read())['result']['records']
columns = "columns.json"
taxdata = losser.table(tax_dict, columns)
with open(os.path.join(dir_path, "tax.csv"), 'wb') as outfile:
    fp = csv.DictWriter(outfile, taxdata[0].keys())
    fp.writeheader()
    fp.writerows(taxdata)

print("Getting income data...")
income_response = requests.get(income_url)
income_response_str = income_response.content.decode("utf-8")
json_data = json.loads(income_response_str)
geo = ["state","county","tract","block group"]
income_var_url = ["https://api.census.gov/data/2016/acs/acs5/variables/" + i for i in income_variables]
income_names = [json.loads(requests.get(var).content.decode("utf-8"))['label'] for var in income_var_url]+geo
with open(os.path.join(dir_path, "income.csv"), 'w') as outfile:
    fp = csv.writer(outfile)
    fp.writerow(income_names)
    for row in json_data:
        fp.writerow(row)

print("Getting occupancy data...")
occupancy_response = requests.get(occupancy_url)
occupancy_response_str = occupancy_response.content.decode("utf-8")
json_data = json.loads(occupancy_response_str)
occupancy_var_url = ["https://api.census.gov/data/2016/acs/acs5/variables/" + i for i in occupancy_variables]
occupancy_names = [json.loads(requests.get(var).content.decode("utf-8"))['label'] for var in occupancy_var_url]+geo
with open(os.path.join(dir_path, "occupancy.csv"), 'w') as outfile:
    fp = csv.writer(outfile)
    fp.writerow(occupancy_names)
    for row in json_data:
        fp.writerow(row)

print("Getting year built data...")
yearBuilt_response = requests.get(yearBuilt_url)
yearBuilt_response_str = yearBuilt_response.content.decode("utf-8")
json_data = json.loads(yearBuilt_response_str)
yearBuilt_var_url = ["https://api.census.gov/data/2016/acs/acs5/variables/" + i for i in yearBuilt_variables]
yearBuilt_names = [json.loads(requests.get(var).content.decode("utf-8"))['label'] for var in yearBuilt_var_url]+geo
with open(os.path.join(dir_path, "yearBuilt.csv"), 'w') as outfile:
    fp = csv.writer(outfile)
    fp.writerow(yearBuilt_names)
    for row in json_data:
        fp.writerow(row)

print("Getting year moved in data...")
yearMovedIn_response = requests.get(yearMovedIn_url)
yearMovedIn_response_str = yearMovedIn_response.content.decode("utf-8")
json_data = json.loads(yearMovedIn_response_str)
yearMovedIn_var_url = ["https://api.census.gov/data/2016/acs/acs5/variables/" + i for i in yearMovedIn_variables]
yearMovedIn_names = [json.loads(requests.get(var).content.decode("utf-8"))['label'] for var in yearMovedIn_var_url]+geo
with open(os.path.join(dir_path, "yearMovedIn.csv"), 'w') as outfile:
    fp = csv.writer(outfile)
    fp.writerow(yearMovedIn_names)
    for row in json_data:
        fp.writerow(row)
        
