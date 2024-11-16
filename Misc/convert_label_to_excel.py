import pandas as pd
import json
import os


os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../")

with open("./Data/all_metadata.json", 'r') as f:
    metadata = json.load(f)['datainfo']
    
df = pd.DataFrame(columns=['REC', 'LNM', 'TD', 'TI', 'CE', 'PI'])

for item in metadata:
    REC = item['REC']
    LNM = item['LNM']
    TD = item['TD']
    TI = item['TI']
    CE = item['CE']
    PI = item['PI']
    df = df.append({'REC': REC, 'LNM': LNM, 'TD': TD, 'TI': TI, 'CE': CE, 'PI': PI}, ignore_index=True)

df.to_excel("./Data/all_metadata_excel.xlsx", index=False)


