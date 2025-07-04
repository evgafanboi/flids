import os
import argparse
import numpy as np
import tensorflow as tf
import gc
import logging
import pandas as pd
import time
import pickle
from strategy.FedAvg import FedAvg
from strategy.FedAGRU import FedAGRU
from models import dense, dense_sim, gru

# Control GPU memory usage
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        
        # Memory defragmentation settings (TF 2.x)
        gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=0.9,  # Use 90% of available memory
            allow_growth=True,
            polling_active_delay=10,
            allocator_type='BFC'  # Best-fit with coalescing - better handles fragmentation
        )
        
        # Create a config that optimizes for memory usage
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
        config.gpu_options.allow_growth = True
        
        # Force TensorFlow to use this configuration
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        
    except Exception as e:
        print(f"GPU configuration error: {e}")

### Eye candy
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    PURPLE = '\033[95m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def setup_logger(n_clients, partition_type, strategy="FedUp"):
    """Set up logging configuration"""
    log_filename = f"{strategy}_{n_clients}client_{partition_type}.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format='%(levelname)s: %(message)s',
        filemode='w'
    )
    print(bcolors.OKCYAN + f"Logging to {log_filename}" + bcolors.ENDC)
    return logging.getLogger(), log_filename

def parse_partition_type(partition_type):
    """Parse partition type like 'iid-10' into type and client count"""
    if '-' in partition_type:
        parts = partition_type.split('-')
        if len(parts) == 2:
            try:
                return parts[0], int(parts[1])
            except ValueError:
                pass
    raise ValueError(f"Invalid partition type '{partition_type}'. Use format 'iid-10'")

def setup_paths(client_id, partition_type, client_count):
    """Set up data paths for a specific client"""
    partitions_root = os.path.join("data", "partitions")
    client_folder = f"{client_count}_client"
    partition_dir = os.path.join(partitions_root, client_folder, partition_type)
    
    if not os.path.isdir(partition_dir):
        raise FileNotFoundError(f"Partition directory not found: {partition_dir}")
    
    return {
        'train_X': os.path.join(partition_dir, f"client_{client_id}_X_train.npy"),
        'train_y': os.path.join(partition_dir, f"client_{client_id}_y_train.npy"),
        'test_X': os.path.join("data", "X_test.npy"),
        'test_y': os.path.join("data", "y_test.npy")
    }

# Update create_model function
def create_model(input_dim, num_classes, model_type, batch_size=1024):
    """Create model with batch size parameter"""
    if model_type.lower() == "dense":
        return dense.create_dense_model(input_dim, num_classes)
    elif model_type.lower() in ["gru"]:
        # Use the simplified GRU model
        return gru.create_enhanced_gru_model((1, input_dim), num_classes, batch_size)
    elif model_type.lower() == "dense_sim":
        return dense_sim.create_dense_model(input_dim, num_classes, batch_size)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def create_client_dataset(X_path, y_path, input_dim, num_classes, batch_size, is_gru=False):
    """Create dataset for client training"""
    def generator():
        X_mmap = np.load(X_path, mmap_mode='r')
        y_mmap = np.load(y_path, mmap_mode='r')
        total_samples = X_mmap.shape[0]
        chunk_size = 10000
        
        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            X_chunk = np.array(X_mmap[start_idx:end_idx])
            
            if is_gru:
                X_chunk = X_chunk.reshape(-1, 1, input_dim)
            
            y_chunk = np.array(y_mmap[start_idx:end_idx])
            if num_classes and (len(y_chunk.shape) == 1 or y_chunk.shape[1] == 1):
                y_chunk = tf.keras.utils.to_categorical(y_chunk, num_classes=num_classes)
            
            yield X_chunk, y_chunk
            del X_chunk, y_chunk
    
    if is_gru:
        output_signature = (
            tf.TensorSpec(shape=(None, 1, input_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        )
    else:
        output_signature = (
            tf.TensorSpec(shape=(None, input_dim), dtype=tf.float32),
            tf.TensorSpec(shape=(None, num_classes), dtype=tf.float32)
        )
    
    dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)
    return dataset.unbatch().batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Add this function for more aggressive memory cleanup
def aggressive_memory_cleanup():
    """More aggressive memory cleanup for OOM prevention"""
    tf.keras.backend.clear_session()
    gc.collect()
    
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()
    
    # Clear any cached operations
    if hasattr(tf, 'config') and hasattr(tf.config, 'experimental'):
        try:
            # Reset memory stats if available
            tf.config.experimental.reset_memory_stats('GPU:0')
        except:
            pass
    
    time.sleep(1)  # Allow GPU memory to actually be released

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Simulation")
    parser.add_argument("--n_clients", type=int, default=5, help="Number of clients to simulate")
    parser.add_argument("--partition_type", type=str, default="iid-10", help="Partition type (e.g., 'iid-10')")
    parser.add_argument("--model_type", type=str, default="gru", help="Model type: 'dense' or 'gru'")
    parser.add_argument("--rounds", type=int, default=10, help="Number of federated learning rounds")
    parser.add_argument("--strategy", type=str, default="FedAvg", help="Strategy: 'FedAvg' or 'FedAGRU'")
    parser.add_argument("--batch_size", type=int, default=8192, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs per round")
    parser.add_argument("--weights_cache_dir", type=str, default="temp_weights", help="Directory to cache weights")
    parser.add_argument("--patience_T", type=int, default=3, help="Patience rounds for FedAGRU")
    parser.add_argument("--selection", type=bool, default=True, help="Enable client selection in FedAGRU")
    parser.add_argument("--disable_client_selection", action='store_true', help="Disable client selection based on importance")
    
    args = parser.parse_args()
    
    # Setup
    os.makedirs(args.weights_cache_dir, exist_ok=True)
    
    if args.strategy == "FedAGRU":
        # Calculate fixed threshold based on number of clients
        threshold_v = 1.0 / args.n_clients
        strategy = FedAGRU()
        print(f"{bcolors.PURPLE}Using FedAGRU with fixed threshold_v={threshold_v:.4f}, patience_T={args.patience_T}{bcolors.ENDC}")
    else:
        strategy = FedAvg()
        print(f"{bcolors.PURPLE}Using FedAvg aggregation strategy{bcolors.ENDC}")

    partition_type, client_count = parse_partition_type(args.partition_type)
    n_clients = min(args.n_clients, client_count)
    
    logger, log_filename = setup_logger(n_clients, partition_type, strategy=args.strategy)
    excel_filename = log_filename.replace('.log', '.xlsx')
    results_df = pd.DataFrame(columns=['Round', 'Loss', 'Accuracy'])
    
    # Get metadata
    with open(os.path.join("data", "partitions", "meta.txt"), "r") as f:
        num_classes = int(f.read().strip())
    
    sample_X = np.load(os.path.join("data", "X_test.npy"), mmap_mode='r')
    input_dim = sample_X.shape[1]
    del sample_X
    
    # Setup paths for all clients
    paths_list = []
    for i in range(n_clients):
        paths = setup_paths(str(i), partition_type, client_count)
        paths_list.append(paths)
    
    # Initialize client status for FedAGRU
    client_status = {}
    for i in range(n_clients):
        client_status[i] = {"importance": 1.0, "patience": 0, "active": True}
    
    latest_weights = None
    fit_metrics_history = {'loss': []}
    eval_metrics_history = {'accuracy': []}
    
    # Main federated learning loop
    for round_num in range(1, args.rounds + 1):
        logger.info(f"Round {round_num}/{args.rounds}")
        print(f"\n{bcolors.HEADER}Round {round_num}/{args.rounds}{bcolors.ENDC}")
        
        # Determine active clients
        active_clients = [i for i in range(n_clients) if client_status[i]["active"]]
        
        if args.strategy == "FedAGRU":
            print(f"{bcolors.OKCYAN}Active clients: {active_clients}{bcolors.ENDC}")
        
        if not active_clients:
            print(f"{bcolors.FAIL}No active clients remaining. Stopping training.{bcolors.ENDC}")
            break
        
        weights_files = []
        sample_sizes = []
        client_losses = []
        participating_clients = []
        
        # Train each active client
        for i in active_clients:
            tf.keras.backend.clear_session()
            gc.collect()
            time.sleep(1)
            
            client_id = str(i)
            print(f"\n{bcolors.BOLD}Processing Client {client_id}{bcolors.ENDC}")
            
            is_gru = args.model_type.lower() in ["gru"]
            model = create_model(input_dim, num_classes, args.model_type, args.batch_size)
            
            if latest_weights is not None:
                model.set_weights(latest_weights)
            
            paths = paths_list[i]
            train_dataset = create_client_dataset(
                paths['train_X'], paths['train_y'], 
                input_dim, num_classes, args.batch_size, is_gru
            )
            
            # Get dataset size
            X_train_mmap = np.load(paths['train_X'], mmap_mode='r')
            X_train_size = X_train_mmap.shape[0]
            sample_sizes.append(X_train_size)
            del X_train_mmap
            
            # Train client model
            print(f"Training model for client {client_id} for {args.epochs} epochs")
            history = model.fit(
                train_dataset,
                epochs=args.epochs,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='loss', patience=2, restore_best_weights=True
                    )
                ]
            )
            
            # Get training loss
            loss = float(history.history["loss"][-1])
            client_losses.append(loss)
            
            # Local evaluation (FedAGRU only)
            local_accuracy = None
            local_loss = None
            if args.strategy == "FedAGRU":
                print(f"Evaluating client {client_id} on test set")
                test_dataset = create_client_dataset(
                    paths['test_X'], paths['test_y'],
                    input_dim, num_classes, args.batch_size, is_gru
                )
                test_results = model.evaluate(test_dataset, verbose=1)
                if isinstance(test_results, list):
                    local_loss = test_results[0]    # Get loss
                    local_accuracy = test_results[1]  # Get accuracy
                else:
                    local_loss = test_results
                    local_accuracy = 0.0
                del test_dataset
    
            # Save weights
            weights_file = os.path.join(args.weights_cache_dir, f"client_{client_id}_round_{round_num}_weights.pkl")
            
            if args.strategy == "FedAGRU":
                weights_data = {
                    'weights': model.get_weights(),
                    'local_accuracy': local_accuracy,
                    'local_loss': local_loss
                }
                with open(weights_file, 'wb') as f:
                    pickle.dump(weights_data, f)
            else:
                # Save only weights for other strategies
                with open(weights_file, 'wb') as f:
                    pickle.dump(model.get_weights(), f)
            
            weights_files.append(weights_file)
            participating_clients.append(i)
            
            # Log training completion
            logger.info(f"Client {client_id}: Training Loss {loss:.4f}, Local Test Loss {local_loss if local_loss else 'N/A'}, Local Accuracy {local_accuracy if local_accuracy else 'N/A'}")
            print(f"{bcolors.WARNING}Client {client_id}: Training completed - Loss {loss:.4f}, Test Loss {local_loss if local_loss else 'N/A'}, Accuracy {local_accuracy if local_accuracy else 'N/A'}{bcolors.ENDC}")
            
            del model, train_dataset, history
            aggressive_memory_cleanup()
        
        if not participating_clients:
            print(f"{bcolors.FAIL}No clients participated in this round. Stopping training.{bcolors.ENDC}")
            break
        
        # Load weights from disk for aggregation
        weights_list = []
        local_accuracies_list = []
        local_losses_list = []

        for weights_file in weights_files:
            with open(weights_file, 'rb') as f:
                saved_data = pickle.load(f)
                
                # Handle different save formats
                if isinstance(saved_data, dict):
                    # FedAGRU format
                    weights_list.append(saved_data['weights'])
                    if 'local_accuracy' in saved_data and saved_data['local_accuracy'] is not None:
                        local_accuracies_list.append(saved_data['local_accuracy'])
                    if 'local_loss' in saved_data and saved_data['local_loss'] is not None:
                        local_losses_list.append(saved_data['local_loss'])
                else:
                    # Regular format (just weights)
                    weights_list.append(saved_data)

        # Prepare data for aggregation
        local_accuracies_for_aggregation = local_accuracies_list if local_accuracies_list and len(local_accuracies_list) == len(weights_list) else None
        local_losses_for_aggregation = local_losses_list if local_losses_list and len(local_losses_list) == len(weights_list) else None

        print(f"\n{bcolors.OKBLUE}Aggregating weights with {args.strategy}{bcolors.ENDC}")
        
        # Aggregate weights
        if args.strategy == "FedAGRU":
            # Calculate average training loss for this round
            round_loss = sum(client_losses) / len(client_losses) if client_losses else None
            
            aggregated_weights, importance_vector = strategy.aggregate(
                weights_list, 
                sample_sizes, 
                local_accuracies=local_accuracies_for_aggregation, 
                local_losses=local_losses_for_aggregation,
                round_loss=round_loss
            )
            
            if args.disable_client_selection:
                # Use FedAGRU for importance-weighted aggregation only, no client dropping
                print(f"{bcolors.CYAN}Client selection DISABLED - using all clients with importance weighting{bcolors.ENDC}")
                
                # Just update importance scores for logging, but don't disable any clients
                for idx, client_id in enumerate(participating_clients):
                    client_status[client_id]["importance"] = importance_vector[idx]
                    # Keep all clients active
                    client_status[client_id]["patience"] = 0
                    client_status[client_id]["active"] = True
                    # Log client importance
                    logger.info(f"Round {round_num} - Client {client_id}: Importance = {importance_vector[idx]:.4f}")
            else:
                # Original client selection logic
                threshold_v = 1.0 / args.n_clients
                
                # Update client status using actual importance scores
                for idx, client_id in enumerate(participating_clients):
                    old_importance = client_status[client_id]["importance"]
                    new_importance = importance_vector[idx]
                    client_status[client_id]["importance"] = new_importance
                    
                    # Log client importance
                    logger.info(f"Round {round_num} - Client {client_id}: Importance = {new_importance:.4f}")
                    
                    if new_importance >= threshold_v:
                        client_status[client_id]["patience"] = 0
                        print(f"Client {client_id}: importance={new_importance:.4f} (ABOVE threshold={threshold_v:.4f}) - patience reset")
                    else:
                        client_status[client_id]["patience"] += 1
                        print(f"Client {client_id}: importance={new_importance:.4f} (BELOW threshold={threshold_v:.4f}) - patience={client_status[client_id]['patience']}")
                        
                        if client_status[client_id]["patience"] > args.patience_T:
                            client_status[client_id]["active"] = False
                            print(f"{bcolors.FAIL}Client {client_id} disabled (patience exceeded){bcolors.ENDC}")
        
        else:
            aggregated_weights = strategy.aggregate(weights_list, sample_sizes)
        
        latest_weights = aggregated_weights
        del weights_list
        gc.collect()
        
        # Evaluate aggregated model
        eval_model = create_model(input_dim, num_classes, args.model_type, args.batch_size)
        eval_model.set_weights(aggregated_weights)
        
        test_dataset = create_client_dataset(
            paths_list[0]['test_X'], paths_list[0]['test_y'],
            input_dim, num_classes, args.batch_size, 
            args.model_type.lower() in ["gru"]
        )
        
        print(f"Evaluating aggregated model")
        test_results = eval_model.evaluate(test_dataset, verbose=1)
        
        if isinstance(test_results, list):
            test_loss, accuracy = test_results[0], test_results[1]
        else:
            test_loss, accuracy = test_results, 0
        
        # Store results
        eval_metrics_history['accuracy'].append((round_num, accuracy))
        fit_metrics_history['loss'].append((round_num, test_loss))
        
        new_row = pd.DataFrame({
            'Round': [round_num],
            'Loss': [test_loss],
            'Accuracy': [accuracy]
        })
        results_df = pd.concat([results_df, new_row], ignore_index=True)
        results_df.to_excel(excel_filename, index=False)
        
        round_avg_loss = sum(client_losses) / len(client_losses)
        logger.info(f"Round {round_num} completed - Avg Loss: {round_avg_loss:.4f}, Avg Accuracy: {accuracy*100:.2f}%")
        print(f"{bcolors.OKGREEN}Round {round_num} completed - Avg Loss: {round_avg_loss:.4f}, Avg Accuracy: {accuracy*100:.2f}%{bcolors.ENDC}")
        
        # Clean up weights files after each round
        for weights_file in weights_files:
            if round_num != args.rounds:  # Only delete if not the last round (save final weights)
                try:
                    os.remove(weights_file)
                except OSError:
                    pass
        
        # Delete eval model and cleanup
        del eval_model, test_dataset, aggregated_weights
        aggressive_memory_cleanup()
        time.sleep(1)
    
    # Final cleanup
    try:
        os.rmdir(args.weights_cache_dir)
    except OSError:
        pass
        
    print(f"{bcolors.OKCYAN}Results saved to {excel_filename}{bcolors.ENDC}")

if __name__ == "__main__":
    main()