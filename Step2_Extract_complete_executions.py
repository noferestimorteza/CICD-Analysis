import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

input_path = r"runs_fixed_uniform.csv"
output_dir = Path(r"")
output_dir.mkdir(parents=True, exist_ok=True)
output_file = output_dir / "complete_executions.csv"

CHUNKSIZE = 100000   
TARGET_COMPLETED = "completed"

# base columns
base_cols = [
    "metadata.event",
    "run_attempt",                    
    "metadata.head_branch",
    "metadata.repository.fork",
    "metadata.head_repository.fork",
    "metadata.repository.private",
    "metadata.head_repository.private",
    "metadata.actor.type",
    "metadata.triggering_actor.type",
    "metadata.head_repository.owner.type",
    "metadata.repository.owner.type",
    "time_to_import",
    "metadata.run_number",
    "metadata.run_started_at",        
    "metadata.updated_at",            
    "metadata.head_commit.timestamp", 
    "metadata.head_commit.message",
    "metadata.pull_requests",
    "metadata.referenced_workflows",
    "repository_name",                
    "workflow_path",                  
    "metadata.actor.login",               
    "metadata.triggering_actor.login",     
    "metadata.name",                                        
    "metadata.head_commit.author.email",   
    "metadata.head_repository.full_name", 
    "metadata.run_attempt",          
]
filtered_chunks = []
total_rows = 0

print("Starting to read the main file...")
for chunk_idx, chunk in enumerate(pd.read_csv(input_path, chunksize=CHUNKSIZE, low_memory=False)):
    print(f"Chunk {chunk_idx + 1} – number of records: {len(chunk)}")

    completed_mask = chunk["metadata.status"].str.lower() == TARGET_COMPLETED
    chunk_completed = chunk[completed_mask].copy()

    if len(chunk_completed) == 0:
        continue

    needed_cols = [c for c in base_cols if c in chunk_completed.columns]

    chunk_filtered = chunk_completed[needed_cols].copy()

    if "metadata.updated_at" in chunk_filtered.columns and "metadata.run_started_at" in chunk_filtered.columns:
        chunk_filtered["metadata.updated_at"] = pd.to_datetime(chunk_filtered["metadata.updated_at"], errors="coerce")
        chunk_filtered["metadata.run_started_at"] = pd.to_datetime(chunk_filtered["metadata.run_started_at"], errors="coerce")
        chunk_filtered["duration_seconds"] = (
            (chunk_filtered["metadata.updated_at"] - chunk_filtered["metadata.run_started_at"])
            .dt.total_seconds()
        )
        # remove records with invalid duration
        chunk_filtered = chunk_filtered[
            chunk_filtered["duration_seconds"].notna() & (chunk_filtered["duration_seconds"] > 0)
        ]
    else:
        print("  -> time columns for building target are missing, skipping this chunk.")
        continue

    if len(chunk_filtered) == 0:
        continue

    filtered_chunks.append(chunk_filtered)
    total_rows += len(chunk_filtered)
    print(f"  -> records kept: {len(chunk_filtered)} | total so far: {total_rows}")

# Merge chunks
df = pd.concat(filtered_chunks, ignore_index=True)

print(f"\nTotal final records: {len(df)}")

#Time feature engineering
if "metadata.run_started_at" in df.columns:
    df["run_start_dt"] = pd.to_datetime(df["metadata.run_started_at"])
    df["start_hour"] = df["run_start_dt"].dt.hour
    df["start_dayofweek"] = df["run_start_dt"].dt.dayofweek
    df["start_month"] = df["run_start_dt"].dt.month

if "metadata.head_commit.timestamp" in df.columns:
    df["commit_timestamp"] = pd.to_datetime(df["metadata.head_commit.timestamp"], errors="coerce")


if "metadata.run_started_at" in df.columns and "metadata.head_commit.timestamp" in df.columns:
    run_start = pd.to_datetime(df["metadata.run_started_at"])
    commit_ts = pd.to_datetime(df["metadata.head_commit.timestamp"], errors="coerce")
    df["commit_to_run_seconds"] = (run_start - commit_ts).dt.total_seconds()
    #df.drop(columns=["metadata.run_started_at", "run_start_dt","commit_timestamp", "metadata.head_commit.timestamp"], inplace=True, errors="ignore")
df.drop(columns=["metadata.updated_at"], inplace=True, errors="ignore")

# Save
df.to_csv(output_file, index=False)
print(f"\nFinal dataset with {len(df)} records and {len(df.columns)} columns saved at:\n{output_file}")
