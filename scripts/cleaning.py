import pandas as pd
import os
from tqdm import tqdm

# File paths
input_path = 'complaints.csv'
output_path = 'complaints_cleaned.csv'

# Set chunk size
chunk_size = 500000

# Columns to lowercase
cols_to_lower = [
    'company', 'state', 'submitted via', 'product',
    'sub-product', 'issue', 'sub-issue'
]

# Remove existing output file if rerunning script
if os.path.exists(output_path):
    os.remove(output_path)

# First pass to estimate number of chunks
print("Estimating total number of chunks...")
total_chunks = sum(1 for _ in pd.read_csv(
    input_path,
    chunksize=chunk_size,
    dtype={"Consumer disputed?": "str"},
    parse_dates=['Date received', 'Date sent to company']
))

print(" Cleaning data in chunks and writing to output file...")
for i, chunk in enumerate(tqdm(pd.read_csv(
    input_path,
    chunksize=chunk_size,
    dtype={"Consumer disputed?": "str"},
    parse_dates=['Date received', 'Date sent to company']
), total=total_chunks, desc="Cleaning Chunks")):

    # Standardise column names
    chunk.columns = [col.strip().lower().replace(" ", "_") for col in chunk.columns]

    # Clean all object (text) columns
    for col in chunk.select_dtypes(include='object').columns:
        chunk[col] = (
            chunk[col]
            .astype(str)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

    # Lowercase specific columns if they exist
    for col in cols_to_lower:
        col_lower = col.strip().lower().replace(" ", "_")
        if col_lower in chunk.columns:
            chunk[col_lower] = chunk[col_lower].str.lower()

    # Fill missing tags (optional)
    if 'tags' in chunk.columns:
        chunk['tags'] = chunk['tags'].fillna("no_tag")

    # Drop duplicate rows
    chunk.drop_duplicates(inplace=True)

    # Save cleaned chunk
    mode = 'w' if i == 0 else 'a'
    header = i == 0
    chunk.to_csv(output_path, mode=mode, index=False, header=header)

print(f" Finished cleaning and saved to: {output_path}")
