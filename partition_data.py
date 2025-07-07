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
    parser.add_argument("--partition-type", type=str, choices=["iid", "label_skew", "label_extreme", "iid_poisoning"], 
                       default="iid", help="Partitioning strategy")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha for label_skew. Lower = more skewed.")
    parser.add_argument("--chunksize", type=int, default=100_000, help="Chunk size for reading large PKL files")
    parser.add_argument("--poison-ratio", type=float, default=0.2, help="Ratio of clients to poison (default: 0.2)")
    parser.add_argument("--poison-intensity", type=float, default=0.5, help="Fraction of samples to poison per client (0.1-0.9)")
    parser.add_argument("--missing-classes", type=int, default=3, help="Number of classes to exclude for extreme clients")
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

def create_extreme_label_skew(all_data, n_clients, num_classes, missing_classes, rng):
    """
    Create extreme label skew where n/5 clients have 0 samples of 2-3 classes
    """
    print(f"\n=== Creating Extreme Label Skew ===")
    
    # Determine number of extreme clients
    n_extreme = max(1, n_clients // 5)
    n_normal = n_clients - n_extreme
    
    print(f"Extreme clients: {n_extreme}, Normal clients: {n_normal}")
    print(f"Each extreme client will miss {missing_classes} classes")
    
    # Group data by class
    class_data = {}
    for label in range(num_classes):
        class_mask = all_data['label_encoded'] == label
        class_data[label] = all_data[class_mask].copy()
        print(f"Class {label}: {len(class_data[label])} samples")
    
    # Initialize client data
    client_data = {i: [] for i in range(n_clients)}
    
    # For extreme clients: assign missing classes
    extreme_client_missing = {}
    for i in range(n_extreme):
        # Randomly select classes to exclude
        missing = rng.choice(num_classes, size=missing_classes, replace=False)
        extreme_client_missing[i] = set(missing)
        print(f"Extreme client {i} missing classes: {missing}")
    
    # Distribute data class by class
    for label in range(num_classes):
        class_df = class_data[label]
        n_samples = len(class_df)
        
        if n_samples == 0:
            continue
            
        # Determine which clients can receive this class
        eligible_clients = []
        
        # Normal clients can receive any class
        for i in range(n_extreme, n_clients):
            eligible_clients.append(i)
        
        # Extreme clients can only receive classes they're not missing
        for i in range(n_extreme):
            if label not in extreme_client_missing[i]:
                eligible_clients.append(i)
        
        if not eligible_clients:
            print(f"Warning: No eligible clients for class {label}")
            continue
        
        print(f"Class {label}: distributing to clients {eligible_clients}")
        
        # Distribute samples among eligible clients
        shuffled_df = class_df.sample(frac=1, random_state=42).reset_index(drop=True)
        splits = np.array_split(np.arange(n_samples), len(eligible_clients))
        
        
        for idx, client_id in enumerate(eligible_clients):
            if len(splits[idx]) > 0:
                client_chunk = shuffled_df.iloc[splits[idx]]
                client_data[client_id].append(client_chunk)
                print(f"  Client {client_id}: +{len(client_chunk)} samples of class {label}")
    
    return client_data, extreme_client_missing

def create_iid_poisoning(all_data, n_clients, poison_ratio, poison_intensity, rng):
    """
    Create IID distribution with targeted label poisoning for some clients.
    
    Uses realistic poisoning strategies:
    1. Targeted label flipping (e.g., Normal -> Attack, specific attack types)
    2. Partial poisoning (only poison a fraction of samples per client)
    3. Consistent poisoning patterns for better model confusion
    """
    print(f"\n=== Creating IID with Targeted Poisoning ===")
    
    # Determine number of poisoned clients
    n_poisoned = max(1, int(n_clients * poison_ratio))
    poisoned_clients = rng.choice(n_clients, size=n_poisoned, replace=False)
    
    print(f"Poisoned clients: {poisoned_clients} ({n_poisoned}/{n_clients})")
    
    # Shuffle all data
    shuffled_data = all_data.sample(frac=1, random_state=42).reset_index(drop=True)
    n_samples = len(shuffled_data)
    
    # Split equally among clients (IID)
    splits = np.array_split(np.arange(n_samples), n_clients)
    client_data = {i: [] for i in range(n_clients)}
    
    unique_labels = sorted(all_data['label_encoded'].unique())
    num_classes = len(unique_labels)
    
    # Create targeted poisoning strategy
    # Strategy 1: Create specific label flip mappings (more realistic)
    poison_mappings = {}
    
    # For intrusion detection: flip normal to attack, and mix up attack types
    # Assuming label 0 is often "Normal" or "Benign"
    if 0 in unique_labels and num_classes > 1:
        # Map normal traffic to a random attack class
        target_attack = rng.choice([l for l in unique_labels if l != 0])
        poison_mappings[0] = target_attack
        print(f"Poison mapping: Normal (0) -> Attack ({target_attack})")
    
    # Create confusion between similar attack types
    # Map some attack classes to other attack classes
    if num_classes > 2:
        attack_labels = [l for l in unique_labels if l != 0]
        if len(attack_labels) >= 2:
            # Pick pairs of attack types to confuse
            num_pairs = min(3, len(attack_labels) // 2)
            attack_sample = rng.choice(attack_labels, size=min(len(attack_labels), 6), replace=False)
            
            for i in range(0, len(attack_sample) - 1, 2):
                if i + 1 < len(attack_sample):
                    poison_mappings[attack_sample[i]] = attack_sample[i + 1]
                    print(f"Poison mapping: Attack ({attack_sample[i]}) -> Attack ({attack_sample[i + 1]})")
    
    # If no mappings created, fall back to simple strategy
    if not poison_mappings:
        # Create simple round-robin flipping
        for i, label in enumerate(unique_labels):
            target_label = unique_labels[(i + 1) % num_classes]
            poison_mappings[label] = target_label
            print(f"Poison mapping: Class {label} -> Class {target_label}")
    
    poisoning_stats = {}
    
    for i in range(n_clients):
        if len(splits[i]) > 0:
            client_chunk = shuffled_data.iloc[splits[i]].copy()
            
            # Poison labels for selected clients
            if i in poisoned_clients:
                print(f"Poisoning client {i} labels...")
                original_labels = client_chunk['label_encoded'].copy()
                
                # Apply targeted poisoning to only a fraction of samples (more realistic)
                # Use the poison_intensity parameter with some randomness
                base_intensity = max(0.1, min(0.9, poison_intensity))  # Clamp between 10-90%
                intensity_variance = 0.2  # ±20% variance
                poison_fraction = base_intensity + (rng.random() - 0.5) * intensity_variance
                poison_fraction = max(0.1, min(0.9, poison_fraction))  # Clamp final value
                n_to_poison = int(len(client_chunk) * poison_fraction)
                poison_indices = rng.choice(len(client_chunk), size=n_to_poison, replace=False)
                
                poisoned_count = 0
                for idx in poison_indices:
                    original_label = client_chunk.iloc[idx]['label_encoded']
                    if original_label in poison_mappings:
                        client_chunk.iloc[idx, client_chunk.columns.get_loc('label_encoded')] = poison_mappings[original_label]
                        poisoned_count += 1
                
                print(f"  Client {i}: Poisoned {poisoned_count}/{len(client_chunk)} samples ({poisoned_count/len(client_chunk)*100:.1f}%)")
                print(f"  Client {i}: Target poison fraction was {poison_fraction:.1%}")
                
                # Track poisoning statistics
                final_labels = client_chunk['label_encoded']
                flipped_count = (original_labels != final_labels).sum()
                poisoning_stats[i] = {
                    'total_samples': len(client_chunk),
                    'poisoned_samples': poisoned_count,
                    'flipped_samples': flipped_count,
                    'poison_fraction': poison_fraction
                }
            
            client_data[i].append(client_chunk)
            print(f"Client {i}: {len(client_chunk)} samples ({'POISONED' if i in poisoned_clients else 'CLEAN'})")
    
    return client_data, (poisoned_clients, poison_mappings, poisoning_stats)

def main():
    args = parse_arguments()
    n_clients = args.num_clients
    class_pkls = sorted(glob.glob(os.path.join(args.class_pkl_dir, "CIC23_*.pkl")))
    partition_type = args.partition_type
    
    # Create output directory
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
    print(f"Total classes: {num_classes}")

    # Remove old client PKL files if exist
    for i in range(n_clients):
        client_pkl = os.path.join(output_dir, f"client_{i}.pkl")
        if os.path.exists(client_pkl):
            os.remove(client_pkl)

    # UNIFIED APPROACH: Load all data first for ALL partition types
    print("Loading all data for partitioning...")
    all_dataframes = []
    
    for class_pkl in class_pkls:
        print(f"Loading {class_pkl}...")
        with open(class_pkl, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, (tuple, list)) and len(data) == 2:
            X, y = data
            if hasattr(X, 'shape') and len(X.shape) == 2:
                feature_cols = [f'feature_{i}' for i in range(X.shape[1])]
                df = pd.DataFrame(X, columns=feature_cols)
                df['label'] = y
            else:
                raise ValueError("Unexpected data format in PKL file")
        else:
            raise ValueError("PKL file should contain DataFrame or (X, y) tuple")
        
        all_dataframes.append(df)
    
    # Combine all data
    print("Combining all data...")
    all_data = pd.concat(all_dataframes, ignore_index=True)
    print(f"Total samples: {len(all_data)}")
    
    # Encode labels
    all_data['label_encoded'] = label_encoder.transform(all_data['label'])
    
    # Apply partitioning strategy
    if partition_type == "iid":
        print("\n=== Creating IID Distribution ===")
        # Global shuffle with consistent seed
        shuffled_data = all_data.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        n_samples = len(shuffled_data)
        
        # Split equally among clients
        splits = np.array_split(np.arange(n_samples), n_clients)
        client_data = {i: [] for i in range(n_clients)}
        
        for i in range(n_clients):
            if len(splits[i]) > 0:
                client_chunk = shuffled_data.iloc[splits[i]]
                client_data[i].append(client_chunk)
                print(f"Client {i}: {len(client_chunk)} samples")
    
    elif partition_type == "label_skew":
        print("\n=== Creating Label Skew Distribution ===")
        # Global shuffle with consistent seed
        shuffled_data = all_data.sample(frac=1, random_state=args.seed).reset_index(drop=True)
        
        # Group data by class
        class_data = {}
        for label in range(num_classes):
            class_mask = shuffled_data['label_encoded'] == label
            class_data[label] = shuffled_data[class_mask].copy()
            print(f"Class {label}: {len(class_data[label])} samples")
        
        client_data = {i: [] for i in range(n_clients)}
        
        # Apply Dirichlet distribution to each class
        for label in range(num_classes):
            class_df = class_data[label]
            n_samples = len(class_df)
            
            if n_samples == 0:
                continue
            
            # Sample Dirichlet counts for this class
            counts = sample_dirichlet_counts(n_samples, n_clients, args.alpha, rng)
            
            # Already shuffled globally, just split according to counts
            start_idx = 0
            for i in range(n_clients):
                end_idx = start_idx + counts[i]
                if counts[i] > 0:
                    client_chunk = class_df.iloc[start_idx:end_idx]
                    client_data[i].append(client_chunk)
                    print(f"  Client {i}: +{counts[i]} samples of class {label}")
                start_idx = end_idx
    
    elif partition_type == "label_extreme":
        client_data, extreme_info = create_extreme_label_skew(
            all_data, n_clients, num_classes, args.missing_classes, rng
        )
        # Save extreme client info
        with open(os.path.join(output_dir, "extreme_clients.txt"), "w") as f:
            f.write("Extreme Client Missing Classes:\n")
            for client_id, missing_classes in extreme_info.items():
                f.write(f"Client {client_id}: {sorted(missing_classes)}\n")
    
    elif partition_type == "iid_poisoning":
        client_data, poisoning_info = create_iid_poisoning(
            all_data, n_clients, args.poison_ratio, args.poison_intensity, rng
        )
        # Unpack poisoning information
        poisoned_clients, poison_mappings, poisoning_stats = poisoning_info
        
        # Save detailed poisoned client info
        with open(os.path.join(output_dir, "poisoned_clients.txt"), "w") as f:
            f.write("Poisoned Clients Information:\n")
            f.write("=" * 40 + "\n")
            f.write(f"Poisoned clients: {list(poisoned_clients)}\n")
            f.write(f"Total poisoned: {len(poisoned_clients)}/{n_clients}\n\n")
            
            f.write("Label Poisoning Mappings:\n")
            for orig, target in poison_mappings.items():
                f.write(f"  Class {orig} -> Class {target}\n")
            
            f.write("\nPer-Client Poisoning Statistics:\n")
            for client_id, stats in poisoning_stats.items():
                f.write(f"  Client {client_id}:\n")
                f.write(f"    Total samples: {stats['total_samples']}\n")
                f.write(f"    Poisoned samples: {stats['poisoned_samples']}\n") 
                f.write(f"    Poison fraction: {stats['poison_fraction']:.1%}\n")
                f.write(f"    Actually flipped: {stats['flipped_samples']}\n\n")
    
    del all_data, all_dataframes  # Free memory

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
        X = client_df.drop(['label', 'label_encoded'], axis=1).values.astype(np.float32)
        y = client_df['label_encoded'].values  # Use encoded labels
        
        # Save as npy
        np.save(os.path.join(output_dir, f"client_{i}_X_train.npy"), X)
        np.save(os.path.join(output_dir, f"client_{i}_y_train.npy"), y)
        
        # Log class distribution for this client
        class_counts = np.bincount(y, minlength=num_classes)
        print(f"Client {i}: train {X.shape}, class distribution: {class_counts}")
        
        del X, y, client_df  # Unload memory

    # Save meta.txt
    with open(os.path.join(output_dir, "meta.txt"), "w") as f:
        f.write(str(num_classes))

    print(f"\nPartitioning complete! Training sets saved in: {output_dir}")

if __name__ == "__main__":
    main()