import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from pandasgui import show
import warnings

# Ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Load a sample
file_path = 'complaints.csv'
df_sample = pd.read_csv(file_path, nrows=20, parse_dates=['Date received', 'Date sent to company'])
df_sample = df_sample.sort_values(by='Date received', ascending=False)
print("\n First 20 Rows of the Dataset (formatted clearly):")
show(df_sample)

chunk_size = 500000
print("\n Initialising variables for dataset insights...")

column_data_types = None
total_nulls = None
total_filled = None
distinct_values_per_column = None
value_counts_per_column = None
row_count = 0
duplicates_count = 0
date_min, date_max = None, None
company_counts = Counter()
issue_counts = Counter()
year_counts = Counter()

dtype_mapping = {"Consumer disputed?": "str"}

# Estimate number of chunks
print("\n Estimating total number of chunks (may take a second)...")
total_chunks = sum(1 for _ in pd.read_csv(file_path, chunksize=chunk_size,
                                          dtype=dtype_mapping,
                                          parse_dates=['Date received', 'Date sent to company']))

print("\n Processing dataset in chunks:")
for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size,
                              dtype=dtype_mapping,
                              parse_dates=['Date received', 'Date sent to company']),
                  total=total_chunks, desc="Processing Chunks"):

    if distinct_values_per_column is None:
        distinct_values_per_column = {col: set() for col in chunk.columns}
        value_counts_per_column = {col: Counter() for col in chunk.columns if chunk[col].nunique() <= 100}

    for col in chunk.columns:
        distinct_values_per_column[col].update(chunk[col].dropna().unique())
        if col in value_counts_per_column:
            value_counts_per_column[col].update(chunk[col].fillna("NaN").astype(str).value_counts().to_dict())

    if column_data_types is None:
        column_data_types = chunk.dtypes

    total_nulls = chunk.isnull().sum() if total_nulls is None else total_nulls + chunk.isnull().sum()
    total_filled = chunk.notnull().sum() if total_filled is None else total_filled + chunk.notnull().sum()
    duplicates_count += chunk.duplicated().sum()
    company_counts.update(chunk['Company'].value_counts().to_dict())
    issue_counts.update(chunk['Issue'].value_counts().to_dict())
    year_counts.update(chunk['Date received'].dt.year.value_counts().to_dict())

    chunk_min = chunk['Date received'].min()
    chunk_max = chunk['Date received'].max()
    date_min = chunk_min if date_min is None or chunk_min < date_min else date_min
    date_max = chunk_max if date_max is None or chunk_max > date_max else date_max

    row_count += len(chunk)

# Output results
print("\n Dataset processing completed!")
print(f"\n Date range of dataset: {date_min} to {date_max}")
distinct_counts = {col: len(values) for col, values in distinct_values_per_column.items()}
missing_percentage = (total_nulls / row_count) * 100
low_variance_cols = [col for col, count in distinct_counts.items() if count < 5]
top_companies = company_counts.most_common(10)
top_issues = issue_counts.most_common(10)

print(f"\n Dataset has approximately {row_count:,} rows.")
print(f" Duplicate rows: {duplicates_count:,}\n")
print("\n Column Data Types:")
print(column_data_types)
print("\n Total Filled Values per Column:")
print(total_filled)
print("\n Total Null Values per Column:")
print(total_nulls)
print("\n Percentage of Null Values per Column:")
print(missing_percentage)
print("\n Distinct Values per Column:")
print(pd.Series(distinct_counts))
print("\n Columns with Very Few Unique Values:")
print(low_variance_cols)
print("\n Top 10 Most Complained-About Companies:")
print(top_companies)
print("\n Top 10 Most Frequent Issues:")
print(top_issues)

print("\n🔍 Value Counts for Columns with ≤ 100 Unique Values:")
for col, counter in value_counts_per_column.items():
    print(f"\n🔹 {col} (unique values: {len(counter)})")
    print(pd.Series(counter).sort_values(ascending=False).to_frame(name="Count"))

# Plot missing data
missing_df = pd.DataFrame({'Column': total_nulls.index, 'Missing Percentage': missing_percentage}).sort_values(by='Missing Percentage', ascending=False)
plt.figure(figsize=(12, 5))
sns.barplot(x='Missing Percentage', y='Column', data=missing_df, palette='coolwarm')
plt.xlabel('Missing Data (%)')
plt.ylabel('Columns')
plt.title('Missing Data Distribution')
plt.show(block=False)

# Complaints per year
year_counts_df = pd.DataFrame(list(year_counts.items()), columns=['Year', 'Count']).sort_values(by='Year')
plt.figure(figsize=(10, 5))
sns.lineplot(x='Year', y='Count', data=year_counts_df, marker='o')
plt.xlabel('Year')
plt.ylabel('Number of Complaints')
plt.title('Complaints Over Time')
plt.xticks(rotation=45)
plt.show(block=False)

# Top companies bar chart
top_companies_df = pd.DataFrame(top_companies, columns=['Company', 'Complaints'])
plt.figure(figsize=(10, 6))
sns.barplot(data=top_companies_df, x='Complaints', y='Company', palette='viridis')
plt.title('Top 10 Most Complained-About Companies')
plt.xlabel('Number of Complaints')
plt.ylabel('Company')
plt.tight_layout()
plt.show(block=False)

# Top issues bar chart
top_issues_df = pd.DataFrame(top_issues, columns=['Issue', 'Count'])
top_issues_df['Issue'] = top_issues_df['Issue'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
plt.figure(figsize=(10, 6))
sns.barplot(data=top_issues_df, x='Count', y='Issue', palette='magma')
plt.title('Top 10 Most Frequent Issues')
plt.xlabel('Number of Complaints')
plt.ylabel('Issue')
plt.subplots_adjust(left=0.3)
plt.show(block=False)

input("\n🔵 Press ENTER to close the script...")
