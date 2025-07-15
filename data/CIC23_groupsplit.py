import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

attack_to_class = {
    "Backdoor_Malware": "Web-based",
    "BrowserHijacking": "Web-based",
    "CommandInjection": "Web-based",
    "XSS": "Web-based",
    "Uploading_Attack": "Web-based",
    "XSS": "Web-based",
    "XSS": "Web-based",
    "SqlInjection": "Web-based",
    "BenignTraffic": "BenignTraffic",
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-RSTFINFlood": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS",
    "MITM-ArpSpoofing": "Spoofing",
    "DNS_Spoofing": "Spoofing",
    "DictionaryBruteForce": "BruteForce",
    "DoS-HTTP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS",
    "DoS-TCP_Flood": "DoS",
    "DoS-UDP_Flood": "DoS",
    "Mirai-greeth_flood": "Mirai",
    "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
    "Recon-HostDiscovery": "Recon",
    "Recon-OSScan": "Recon",
    "Recon-PingSweep": "Recon",
    "Recon-PortScan": "Recon",
    "VulnerabilityScan": "Recon"
}

def create_output_directory():
    """Create the CIC23 directory if it doesn't exist"""
    output_dir = Path('./CIC23')
    output_dir.mkdir(exist_ok=True)
    return output_dir

def process_test_data():
    """Load test data and save X_test and y_test as both pkl and npy"""
    print("Processing test data...")
    
    # Load test data from PKL file
    with open('CIC23_test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    # Assuming the data is either a DataFrame or tuple/list with features and labels
    if isinstance(test_data, pd.DataFrame):
        # Assuming last column is the label
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]
    elif isinstance(test_data, (tuple, list)) and len(test_data) == 2:
        X_test, y_test = test_data
    else:
        raise ValueError("Unexpected test data format")
    
    # Save X_test and y_test as PKL files
    with open('X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    
    with open('y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    
    # Convert to numpy arrays and save as npy files
    X_test_np = X_test.values if hasattr(X_test, 'values') else np.array(X_test)
    y_test_np = y_test.values if hasattr(y_test, 'values') else np.array(y_test)
    
    # Ensure proper data types
    X_test_np = X_test_np.astype(np.float32)
    
    # Save as npy files
    np.save('X_test.npy', X_test_np)
    np.save('y_test.npy', y_test_np)
    
    print(f"Test data processed:")
    print(f"  X_test shape: {X_test_np.shape}")
    print(f"  y_test shape: {y_test_np.shape}")
    print(f"  Saved as both pkl and npy formats")
    
    return X_test, y_test

def process_train_data_in_chunks(chunk_size=10000):
    """Process training data in chunks and split by labels"""
    print("Processing training data in chunks...")
    
    output_dir = create_output_directory()
    label_data = {}  # Dictionary to store data for each label
    
    # Load entire PKL file and process in memory chunks
    print("Loading training PKL file...")
    
    with open('CIC23_train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    print(f"Loaded training data. Type: {type(train_data)}")
    
    if isinstance(train_data, pd.DataFrame):
        print(f"DataFrame shape: {train_data.shape}")
        total_rows = len(train_data)
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = train_data.iloc[start_idx:end_idx]
            
            print(f"Processing rows {start_idx} to {end_idx}...")
            
            features = chunk.iloc[:, :-1]
            labels = chunk.iloc[:, -1]
            
            unique_labels = labels.unique()
            
            for label in unique_labels:
                label_mask = labels == label
                label_features = features[label_mask]
                label_labels = labels[label_mask]
                
                label_chunk = pd.concat([label_features, label_labels], axis=1)
                
                if label not in label_data:
                    label_data[label] = []
                
                label_data[label].append(label_chunk)
    
    elif isinstance(train_data, (tuple, list)) and len(train_data) == 2:
        X_train, y_train = train_data
        print(f"Tuple format - X shape: {X_train.shape if hasattr(X_train, 'shape') else len(X_train)}")
        print(f"Tuple format - y shape: {y_train.shape if hasattr(y_train, 'shape') else len(y_train)}")
        
        if hasattr(X_train, 'shape'):
            total_rows = X_train.shape[0]
        else:
            total_rows = len(X_train)
        
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            
            print(f"Processing rows {start_idx} to {end_idx}...")
            
            X_chunk = X_train[start_idx:end_idx]
            y_chunk = y_train[start_idx:end_idx]
            
            unique_labels = np.unique(y_chunk)
            
            for label in unique_labels:
                label_mask = y_chunk == label
                label_features = X_chunk[label_mask]
                label_labels = y_chunk[label_mask]
                
                if label not in label_data:
                    label_data[label] = []
                
                label_data[label].append((label_features, label_labels))
    
    elif isinstance(train_data, np.ndarray):
        print(f"NumPy array shape: {train_data.shape}")
        # Handle case where it's just a numpy array
        # Assuming last column is the label
        total_rows = train_data.shape[0]
        
        for start_idx in range(0, total_rows, chunk_size):
            end_idx = min(start_idx + chunk_size, total_rows)
            chunk = train_data[start_idx:end_idx]
            
            print(f"Processing rows {start_idx} to {end_idx}...")
            
            X_chunk = chunk[:, :-1]
            y_chunk = chunk[:, -1]
            
            unique_labels = np.unique(y_chunk)
            
            for label in unique_labels:
                label_mask = y_chunk == label
                label_features = X_chunk[label_mask]
                label_labels = y_chunk[label_mask]
                
                if label not in label_data:
                    label_data[label] = []
                
                label_data[label].append((label_features, label_labels))
    
    else:
        raise ValueError(f"Unsupported training data format: {type(train_data)}")
    
    # Save each label's data to separate PKL files
    print("Saving label-specific PKL files...")
    for label, chunks in label_data.items():
        print(f"Saving data for label {label}...")
        
        if chunks:
            if isinstance(chunks[0], pd.DataFrame):
                # Concatenate all chunks for this label
                combined_data = pd.concat(chunks, ignore_index=True)
            else:
                # Handle tuple/numpy format
                all_features = []
                all_labels = []
                for features, labels in chunks:
                    all_features.append(features)
                    all_labels.append(labels)
                
                if len(all_features) > 1:
                    combined_features = np.vstack(all_features)
                    combined_labels = np.concatenate(all_labels)
                else:
                    combined_features = all_features[0]
                    combined_labels = all_labels[0]
                
                combined_data = (combined_features, combined_labels)
            
            # Save to PKL file
            output_file = output_dir / f'CIC23_label{label}.pkl'
            with open(output_file, 'wb') as f:
                pickle.dump(combined_data, f)
            
            if isinstance(combined_data, pd.DataFrame):
                print(f"Saved {len(combined_data)} samples for label {label}")
            else:
                print(f"Saved {len(combined_data[1])} samples for label {label}")
    
    print(f"All label PKL files saved in {output_dir}")
    return label_data

def main():
    """Main function to process both test and training PKL data"""
    print("Starting CIC23 PKL data processing...")
    
    # Process test PKL data
    try:
        X_test, y_test = process_test_data()
        print("✓ Test PKL data processing completed (saved as both pkl and npy)")
    except Exception as e:
        print(f"✗ Error processing test PKL data: {e}")
        import traceback
        traceback.print_exc()
    
    # Process training PKL data in chunks
    try:
        label_data = process_train_data_in_chunks(chunk_size=10000)
        print("✓ Training PKL data processing completed")
        print(f"✓ Created {len(label_data)} label-specific PKL files")
    except Exception as e:
        print(f"✗ Error processing training PKL data: {e}")
        import traceback
        traceback.print_exc()
    
    print("PKL data processing finished!")

if __name__ == "__main__":
    main()