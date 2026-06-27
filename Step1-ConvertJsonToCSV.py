import gzip
import json
import pandas as pd
import chardet
import string

# File paths
input_path = r'runs.json.gz'
output_path = r'runs_fixed_uniform.csv'

def is_valid_text(text):
    """Check if text contains only valid printable characters"""
    if not text:
        return True
    try:
        encoded = text.encode('utf-8')
        result = chardet.detect(encoded)
        if result['encoding'] not in ['utf-8', 'ascii'] or result['confidence'] < 0.9:
            return False
        printable = set(string.printable)
        return all(char in printable for char in text)
    except:
        return False

def stream_jsonlines_to_csv_uniform_columns(input_file, output_file, batch_size=10000):
    rows = []
    total_written = 0
    writer_initialized = False
    skipped_lines = 0
    all_columns = set()

    # Step 1: Scan all lines to collect all unique columns
    print("Scanning for all unique columns...")
    with gzip.open(input_file, 'rt', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            try:
                item = json.loads(line)
                flat = pd.json_normalize(item, sep='.')
                all_columns.update(flat.columns)
            except:
                continue

    all_columns = sorted(list(all_columns))
    print(f"Total unique columns: {len(all_columns)}")

    # Step 2: Process and write normalized data in batches
    print("Processing and writing to CSV...")
    with gzip.open(input_file, 'rt', encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            try:
                line_result = chardet.detect(line.encode('utf-8'))
                if line_result['encoding'] not in ['utf-8', 'ascii'] or line_result['confidence'] < 0.9:
                    skipped_lines += 1
                    continue

                if not is_valid_text(line):
                    skipped_lines += 1
                    continue

                item = json.loads(line)
                flat = pd.json_normalize(item, sep='.')
                rows.append(flat)

            except:
                skipped_lines += 1
                continue

            if len(rows) >= batch_size:
                df = pd.concat(rows, ignore_index=True)
                df = df.reindex(columns=all_columns)
                df.to_csv(output_file, mode='a', index=False, header=not writer_initialized)
                writer_initialized = True
                total_written += len(df)
                rows = []

        # Write remaining rows
        if rows:
            df = pd.concat(rows, ignore_index=True)
            df = df.reindex(columns=all_columns)
            df.to_csv(output_file, mode='a', index=False, header=not writer_initialized)
            total_written += len(df)

    print(f"Finished. Total written rows: {total_written}")
    print(f"Skipped lines due to encoding or invalid characters: {skipped_lines}")
    print(f"Output saved to: {output_file}")

# Run
stream_jsonlines_to_csv_uniform_columns(input_path, output_path, batch_size=10000)
