import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from pathlib import Path
import re

# Suppress warnings for cleaner output
warnings.simplefilter(action='ignore', category=FutureWarning)

# === CONFIGURATION ===
file_path = 'data/complaints.csv'
chunk_size = 500_000
dtype_mapping = {"Consumer disputed?": "str"}

# Folder setup
figures_path = Path("figures/EDA_figures")
summary_path = Path("summary")
figures_path.mkdir(parents=True, exist_ok=True)
summary_path.mkdir(parents=True, exist_ok=True)

# === INITIALISE TRACKERS ===
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

# === CHUNKED PROCESSING ===
print("\n🔍 Processing dataset in chunks...")
for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size, dtype=dtype_mapping,
                              parse_dates=['Date received', 'Date sent to company'],
                              low_memory=False), desc="Processing Chunks"):

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

# === COMPUTE SUMMARY STATS ===
distinct_counts = {col: len(vals) for col, vals in distinct_values_per_column.items()}
missing_percentage = (total_nulls / row_count) * 100
low_variance_cols = [col for col, count in distinct_counts.items() if count < 5]
top_companies = company_counts.most_common(10)
top_issues = issue_counts.most_common(10)

# === SAVE SUMMARY TEXT FILE ===
with open(summary_path / "eda_summary.txt", "w") as f:
    f.write(f"Date Range: {date_min} to {date_max}\n")
    f.write(f"Total Rows: {row_count:,}\n")
    f.write(f"Duplicate Rows: {duplicates_count:,}\n\n")
    f.write("Column Data Types:\n")
    f.write(str(column_data_types) + "\n\n")
    f.write("Total Null Values:\n")
    f.write(str(total_nulls) + "\n\n")
    f.write("Missing % per Column:\n")
    f.write(str(missing_percentage) + "\n\n")
    f.write("Distinct Value Counts:\n")
    f.write(str(distinct_counts) + "\n\n")
    f.write("Low Variance Columns (<5):\n")
    f.write(str(low_variance_cols) + "\n\n")
    f.write("Top 10 Companies:\n")
    for company, count in top_companies:
        f.write(f"{company}: {count}\n")
    f.write("\nTop 10 Issues:\n")
    for issue, count in top_issues:
        f.write(f"{issue}: {count}\n")

# === SAVE VALUE COUNTS TO CSV (Safe Filenames) ===
def sanitise_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", name).replace(" ", "_").lower()

for col, counter in value_counts_per_column.items():
    safe_name = sanitise_filename(col)
    df = pd.Series(counter).sort_values(ascending=False).to_frame(name="Count")
    df.to_csv(summary_path / f"value_counts_{safe_name}.csv")

# === PLOT SAVING FUNCTION ===
def save_plot(fig, filename):
    fig.savefig(figures_path / filename, dpi=300)
    plt.close(fig)

# === PLOTS ===
missing_df = pd.DataFrame({'Column': total_nulls.index, 'Missing Percentage': missing_percentage})
missing_df = missing_df.sort_values(by='Missing Percentage', ascending=False)
fig = plt.figure(figsize=(12, 6))
sns.barplot(x='Missing Percentage', y='Column', data=missing_df, palette='coolwarm')
plt.title('Missing Data Distribution')
plt.tight_layout()
save_plot(fig, "missing_data_distribution.png")

year_df = pd.DataFrame(list(year_counts.items()), columns=['Year', 'Count']).sort_values(by='Year')
fig = plt.figure(figsize=(10, 5))
sns.lineplot(data=year_df, x='Year', y='Count', marker='o')
plt.title("Complaints Over Time")
plt.tight_layout()
save_plot(fig, "complaints_over_time.png")

top_companies_df = pd.DataFrame(top_companies, columns=['Company', 'Complaints'])
fig = plt.figure(figsize=(10, 6))
sns.barplot(data=top_companies_df, x='Complaints', y='Company', palette='viridis')
plt.title("Top 10 Most Complained-About Companies")
plt.tight_layout()
save_plot(fig, "top_10_companies.png")

top_issues_df = pd.DataFrame(top_issues, columns=['Issue', 'Count'])
top_issues_df['Issue'] = top_issues_df['Issue'].apply(lambda x: x[:50] + '...' if len(x) > 50 else x)
fig = plt.figure(figsize=(10, 6))
sns.barplot(data=top_issues_df, x='Count', y='Issue', palette='magma')
plt.title("Top 10 Most Frequent Issues")
plt.tight_layout()
save_plot(fig, "top_10_issues.png")

print("\n✅ EDA script complete. Outputs saved to 'figures/' and 'summary/' folders.")
input("Press ENTER to close the script...")
