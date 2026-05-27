import pandas as pd
import os

input_csv = "full_predictions_ensemble.csv"
output_csv = "borderline_cases.csv"

print("Reading predictions...")
chunksize = 50000
chunks = []
for chunk in pd.read_csv(input_csv, chunksize=chunksize):
    borderline = chunk[(chunk['adj_prob'] >= 0.65) & (chunk['adj_prob'] < 0.70)]
    chunks.append(borderline)

df = pd.concat(chunks, ignore_index=True)
df = df.sort_values(by='adj_prob', ascending=False)
df.to_csv(output_csv, index=False)
print(f"Extracted {len(df)} borderline predictions to {output_csv}.")
