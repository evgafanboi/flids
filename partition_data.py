#!/usr/bin/env python3
"""
Data Partitioning Script for FL-IDS

This script creates custom non-IID data partitions for federated learning
in intrusion detection systems. It provides different partitioning strategies
to simulate realistic federated scenarios.
"""

import os
import glob
import numpy as np
import pandas as pd
import pickle
import argparse
from sklearn.preprocessing import LabelEncoder

def parse_arguments():
    parser = argparse.ArgumentParser(description="Partitioning for FL-IDS")
    parser.add_argument("--class-pkl-dir", type=str, default="data/CIC23", help="Directory containing class PKL files")
    parser.add_argument("--output-dir", type=str, default="data/partitions", help="Output directory for client partitions")
    parser.add_argument("--num-clients", type=int, default=10, help="Number of clients")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--partition-type", type=str, choices=["iid", "label_skew"], default="iid", help="Partitioning strategy")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha for label_skew. Lower = more skewed.")
    parser.add_argument("--chunksize", type=int, default=100_000, help="Chunk size for reading large PKL files")
    return parser.parse_args()

def sample_dirichlet_counts(n_samples, n_clients, alpha, rng):
    proportions = rng.dirichlet([alpha] * n_clients)
    proportions = proportions / proportions.sum()  # Normalize proportions to sum to 1
    counts = np.floor(proportions * n_samples).astype(int)
    # Adjust counts to ensure sum == n_samples
    diff = n_samples - np.sum(counts)
    # Distribute the remaining samples (positive or negative) to clients with largest remainder
    if diff != 0:
        remainders = proportions * n_samples - counts
        indices = np.argsort(remainders)[::-1]  # descending
        for i in range(abs(diff)):
            idx = indices[i % n_clients]
            counts[idx] += 1 if diff > 0 else -1
    assert np.sum(counts) == n_samples
    
    # Debug
    print(f"Dirichlet Proportions: {proportions}")
    print(f"Counts per Client: {counts}, Total Distributed: {np.sum(counts)}")
    return counts

def main():
    args = parse_arguments()
    n_clients = args.num_clients
    class_pkls = sorted(glob.glob(os.path.join(args.class_pkl_dir, "CIC23_*.pkl")))
    partition_type = args.partition_type
    "create/use output directory as: output + num_clients + partition_type"
    output_dir = os.path.join(args.output_dir, f"{n_clients}_client", partition_type)
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    # Fit LabelEncoder on all labels from all class PKL files
    print("Fitting label encoder on all class labels...")
    all_labels = []
    for class_pkl in class_pkls:
        with open(class_pkl, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, pd.DataFrame):
            all_labels.extend(data['label'].tolist())
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            _, y = data
            all_labels.extend(y.tolist() if hasattr(y, 'tolist') else list(y))
    
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    np.save(os.path.join(output_dir, "label_classes.npy"), label_encoder.classes_)
    num_classes = len(label_encoder.classes_)

    # Remove old client PKL files if exist
    for i in range(n_clients):
        client_pkl = os.path.join(output_dir, f"client_{i}.pkl")
        if os.path.exists(client_pkl):
            os.remove(client_pkl)

    # Initialize client data containers
    client_data = {i: [] for i in range(n_clients)}

    # Partition data in a memory-efficient way
    for class_pkl in class_pkls:
        print(f"Processing {class_pkl} ...")
        
        # Load PKL data
        with open(class_pkl, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            X, y = data
            # Convert to DataFrame for consistent processing
            if hasattr(X, 'shape') and len(X.shape) == 2:
                feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
                df = pd.DataFrame(X, columns=feature_cols)
                df['label'] = y
            else:
                raise ValueError("Unexpected data format in PKL file")
        else:
            raise ValueError("PKL file should contain DataFrame or (X, y) tuple")
        
        n_samples = len(df)
        
        if partition_type == "iid":
            # Shuffle and split equally
            shuffled_df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
            splits = np.array_split(np.arange(n_samples), n_clients)
            
            for i in range(n_clients):
                client_chunk = shuffled_df.iloc[splits[i]]
                client_data[i].append(client_chunk)
                
        elif partition_type == "label_skew":
            # Sample Dirichlet counts for this class
            counts = sample_dirichlet_counts(n_samples, n_clients, args.alpha, rng)
            
            # Shuffle data
            shuffled_df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
            
            # Distribute according to Dirichlet counts
            start_idx = 0
            for i in range(n_clients):
                end_idx = start_idx + counts[i]
                if counts[i] > 0:
                    client_chunk = shuffled_df.iloc[start_idx:end_idx]
                    client_data[i].append(client_chunk)
                    print(f"  Client {i}: +{len(client_chunk)} rows from {class_pkl}")
                start_idx = end_idx
        else:
            raise ValueError("Unknown partition type")

    # Save client data as PKL files and convert to npy
    for i in range(n_clients):
        if not client_data[i]:
            print(f"Warning: Client {i} has no data.")
            continue
            
        # Combine all chunks for this client
        client_df = pd.concat(client_data[i], ignore_index=True)
        
        # Save as PKL
        client_pkl = os.path.join(output_dir, f"client_{i}.pkl")
        with open(client_pkl, 'wb') as f:
            pickle.dump(client_df, f)
        
        # Convert to numpy arrays
        X = client_df.drop('label', axis=1).values.astype(np.float32)
        y = label_encoder.transform(client_df['label'])
        
        # Save as npy
        np.save(os.path.join(output_dir, f"client_{i}_X_train.npy"), X)
        np.save(os.path.join(output_dir, f"client_{i}_y_train.npy"), y)
        print(f"Client {i}: train {X.shape}")
        
        del X, y, client_df  # Unload memory

    # Save meta.txt
    with open(os.path.join(output_dir, "meta.txt"), "w") as f:
        f.write(str(num_classes))

    print(f"\nPartitioning complete! Training sets saved as pkl and npy in: {output_dir}")

if __name__ == "__main__":
    main()