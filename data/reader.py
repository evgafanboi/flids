import pandas as pd
import pickle
import glob
import os
import argparse
import re

# Argument parser for directory
parser = argparse.ArgumentParser(description="Read client PKLs and output label distribution table.")
parser.add_argument('--dir', type=str, required=False, default='iid/', help='Directory containing client_*.pkl files')
args = parser.parse_args()
data_dir = args.dir

def extract_client_num(filename):
    match = re.search(r'client_(\d+)\.pkl', filename)
    return int(match.group(1)) if match else -1

# Find all client PKL files
file_list = sorted(
    glob.glob(os.path.join(data_dir, 'client_*.pkl')),
    key=extract_client_num
)

# Collect all unique labels across all files
all_labels = set()
for file in file_list:
    with open(file, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, pd.DataFrame):
        all_labels.update(data['label'].unique())
    elif isinstance(data, (tuple, list)) and len(data) == 2:
        _, y = data
        if hasattr(y, 'unique'):
            all_labels.update(y.unique())
        else:
            all_labels.update(set(y))
    else:
        raise ValueError("PKL file should contain DataFrame or (X, y) tuple")

all_labels = sorted(all_labels)

# Count the number of files
num_client = len(file_list)

# Build the table
table = []
label_totals = [0] * len(all_labels)
grand_total = 0
for file in file_list:
    with open(file, 'rb') as f:
        data = pickle.load(f)
    
    if isinstance(data, pd.DataFrame):
        label_counts = data['label'].value_counts().to_dict()
    elif isinstance(data, (tuple, list)) and len(data) == 2:
        _, y = data
        if hasattr(y, 'value_counts'):
            label_counts = y.value_counts().to_dict()
        else:
            # Convert to pandas Series for value_counts
            label_counts = pd.Series(y).value_counts().to_dict()
    else:
        raise ValueError("PKL file should contain DataFrame or (X, y) tuple")
    
    row = [os.path.basename(file)]
    total = 0
    for i, label in enumerate(all_labels):
        count = label_counts.get(label, 0)
        row.append(count)
        label_totals[i] += count
        total += count
    row.append(total)
    grand_total += total
    table.append(row)

# Add total row
total_row = ['Total']
total_row.extend(label_totals)
total_row.append(grand_total)
table.append(total_row)

# --- Add this block to load CIC23_meta.md and extract label totals ---
meta_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'CIC23_meta.md')
meta_label_totals = {}
meta_total_samples = 0

if os.path.exists(meta_path):
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('|') and not line.startswith('| Label'):
                parts = [x.strip() for x in line.strip().strip('|').split('|')]
                if len(parts) == 2 and parts[0] and parts[0] != '**Total label: 34**':
                    try:
                        # Store meta labels as lowercase for case-insensitive matching
                        meta_label_totals[parts[0].lower()] = int(parts[1].replace(',', ''))
                    except ValueError:
                        pass
                elif parts[0].startswith('**Total label'):
                    try:
                        meta_total_samples = int(parts[1].split(':')[-1].replace('**', '').replace(',', '').strip())
                    except Exception:
                        pass

# --- Prepare the meta row for comparison ---
meta_row = ['CIC23_meta']
meta_sum = 0
for label in all_labels:
    # Lookup using lowercase for case-insensitive match
    count = meta_label_totals.get(str(label).lower(), 0)
    meta_row.append(count)
    meta_sum += count
meta_row.append(meta_sum)
table.append(meta_row)

# Output as Markdown
header = ['Client'] + [str(label) for label in all_labels] + ['Total']
md = '| ' + ' | '.join(header) + ' |\n'
md += '| ' + ' | '.join(['---'] * len(header)) + ' |\n'
for row in table:
    md += '| ' + ' | '.join(map(str, row)) + ' |\n'

# Save to a file
with open(f'{data_dir.rstrip(os.sep)}_{num_client}_client_distribution.md', 'w') as f:
    f.write(md)
    
# Convert table to DataFrame
df_out = pd.DataFrame(table, columns=header)

# Save to Excel
excel_path = f'{data_dir.rstrip(os.sep)}_{num_client}_client_distribution.xlsx'
df_out.to_excel(excel_path, index=False)