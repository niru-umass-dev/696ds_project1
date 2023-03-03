import json
import pandas as pd
import os

src = json.load(open("combined_data.json", 'r'))

headers = []
unique_headers = []
records = []
for example in src[20:]:
    for key, value in example.items():
        if "uid" not in key:
            if isinstance(value, list):
                for idx, item in enumerate(value):
                    headers.append(f"{key}_{idx+1}")
                unique_headers.append(key)
            else:
                unique_headers.append(key)
                headers.append(key)
    break
print(headers)
print(unique_headers)

for example in src:
    record = []
    for header in unique_headers:
        if header in example:
            value = example[header]
            if isinstance(value, list):
                for idx, item in enumerate(value):
                    record.append(item)
            else:
                record.append(value)
        else:
            record.append("")
    records.append(tuple(record))

df = pd.DataFrame.from_records(data=records,
                               columns=headers)
df.to_csv("combined_data.csv", index=False)
