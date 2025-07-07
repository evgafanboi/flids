import numpy as np
import json
import os

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

class FedAGRU:
    """
    Implementation of Federated Averaging with Attention Mechanism (FedAGRU).
    """

    def __init__(self, log_path="./agru.log", learning_rate=0.001):
        """
        Initialize the FedAGRU strategy.
        
        Args:
            log_path: Path for logging AGRU states
            learning_rate: Learning rate for updating W_m
        """
        self.name = "FedAGRU"
        self.log_path = log_path
        self.learning_rate = learning_rate
        
        # Learnable parameters (initialized when first weights are received)
        self.W_m = None
        self.param_dim = None
        self.num_clients = None
        self.initialized = False
        self.previous_global_loss = None

    def _initialize_parameters(self, model_weights_list):
        """Initialize learnable parameter matrix W_m based on model dimensions"""
        self.num_clients = len(model_weights_list)
        
        # Flatten and concatenate all model parameters to get param_dim
        flattened_params = self._flatten_weights(model_weights_list[0])
        self.param_dim = len(flattened_params)
        
        # Initialize learnable parameter matrix W_m
        self.W_m = np.random.normal(0, 0.1, (self.param_dim, self.num_clients))
        
        self.initialized = True
        print(f"{bcolors.OKGREEN}Initialized W_m matrix: ({self.param_dim}, {self.num_clients}){bcolors.ENDC}")

    def _flatten_weights(self, weights):
        """Flatten model weights into a single vector"""
        flattened = []
        for layer_weights in weights:
            flattened.extend(layer_weights.flatten())
        return np.array(flattened)

    def _log_agru_state(self, importance_vector):
        """Log the current importance vector"""
        log_entry = {
            "importance_vector": np.asarray(importance_vector).tolist()
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def _update_W_m(self, model_weights_list, importance_vector, round_loss):
        if round_loss is None:
            return
            
        num_clients = len(model_weights_list)
        
        for client_idx in range(num_clients):
            client_params = self._flatten_weights(model_weights_list[client_idx])
            M_i = np.tanh(client_params)
            
            gradient = importance_vector[client_idx] * M_i * (-round_loss)
            
            # Update W_m column for this client
            self.W_m[:, client_idx] += self.learning_rate * gradient
        
        print(f"{bcolors.OKCYAN}Updated W_m parameters based on round loss: {round_loss:.4f}{bcolors.ENDC}")

    def aggregate(self, model_weights_list, sample_sizes=None, local_accuracies=None, local_losses=None, round_loss=None):
        """
        Aggregate model weights using accuracy and loss-based importance weighting
        """
        if not model_weights_list:
            raise ValueError("Cannot aggregate empty weights list")
        
        if sample_sizes is None:
            sample_sizes = [1] * len(model_weights_list)
        
        if local_accuracies is None or len(local_accuracies) != len(model_weights_list):
            raise ValueError("Local accuracies are required for FedAGRU and must match number of clients")
        
        if local_losses is None or len(local_losses) != len(model_weights_list):
            raise ValueError("Local losses are required for FedAGRU and must match number of clients")
        
        importance_vector = self.calculate_importance(model_weights_list, local_accuracies, local_losses)
        
        print(f"Importance vector: {importance_vector}")
        
        # Initialize aggregated weights
        weights_shape = [w.shape for w in model_weights_list[0]]
        aggregated_weights = [np.zeros(shape) for shape in weights_shape]
        
        # Normalize sample sizes
        total_samples = sum(sample_sizes)
        
        total_combined_weight = 0
        for i in range(len(model_weights_list)):
            weights = model_weights_list[i]
            sample_weight = sample_sizes[i] / total_samples
            importance = importance_vector[i]
            
            # Combined weight: sample size 30% * importance 70%
            combined_weight = 0.3*sample_weight + 0.7*importance
            total_combined_weight += combined_weight
            
            for j, layer_weights in enumerate(weights):
                aggregated_weights[j] += combined_weight * layer_weights
            
            print(f"{bcolors.BOLD}Participant {i}: sample_weight={sample_weight:.4f}, importance={importance:.4f}, combined={combined_weight:.4f}{bcolors.ENDC}")
        
        # Normalize aggregated weights
        if total_combined_weight > 0:
            for j in range(len(aggregated_weights)):
                aggregated_weights[j] /= total_combined_weight
        
        # Update W_m based on aggregation performance
        if round_loss is not None:
            self._update_W_m(model_weights_list, importance_vector, round_loss)
        
        # Log the state
        self._log_agru_state(importance_vector)
        
        return aggregated_weights, importance_vector

    def calculate_importance(self, model_weights_list, local_accuracies, local_losses):
        """
        Calculate importance using local accuracies + inverse losses + learned parameters
        """
        num_clients = len(model_weights_list)
        
        # Initialize W_m if needed
        if not self.initialized:
            self._initialize_parameters(model_weights_list)
        
        # Ensure W_m has correct dimensions
        if self.W_m.shape[1] != num_clients:
            print(f"{bcolors.WARNING}Adjusting W_m for {num_clients} clients{bcolors.ENDC}")
            if num_clients > self.W_m.shape[1]:
                new_cols = np.random.normal(0, 0.1, (self.param_dim, num_clients - self.W_m.shape[1]))
                self.W_m = np.hstack([self.W_m, new_cols])
            else:
                self.W_m = self.W_m[:, :num_clients]
        
        print(f"{bcolors.PURPLE}Calculating importance using accuracy + inverse loss + learned parameters{bcolors.ENDC}")
        
        # Combine accuracy, inverse loss, and learned attention
        importance_scores = []
        for idx in range(num_clients):
            client_params = self._flatten_weights(model_weights_list[idx])
            M_i = np.tanh(client_params)
            
            # Learned attention score
            alpha_learned = np.clip(np.dot(self.W_m[:, idx], M_i), -2.0, 2.0)
            
            # Accuracy-based score
            alpha_accuracy = local_accuracies[idx]
            
            # Inverse loss score
            alpha_loss_inverse = 1.0 / (1.0 + local_losses[idx])
            
            combined_alpha = 0.4 * alpha_accuracy + 0.4 * alpha_loss_inverse + 0.2 * alpha_learned
            importance_scores.append(combined_alpha)
            
            print(f"Participant {idx}: accuracy={alpha_accuracy:.4f}, loss_inv={alpha_loss_inverse:.4f}, learned={alpha_learned:.6f}, combined={combined_alpha:.4f}")
        
        # Apply softmax normalization with temperature scaling 
        importance_scores = np.array(importance_scores)
        
        # Temperature scaling to control the sharpness of the distribution
        temperature = 2.0  # Higher temperature = more uniform distribution
        scaled_scores = importance_scores / temperature
        
        # Softmax with numerical stability
        exp_scores = np.exp(scaled_scores - np.max(scaled_scores))
        importance_vector = exp_scores / np.sum(exp_scores)
        
        # Ensure no single client gets more than 60% of total importance
        max_importance = 0.6
        if np.max(importance_vector) > max_importance:
            clipped_vector = np.clip(importance_vector, 0, max_importance)
            importance_vector = clipped_vector / np.sum(clipped_vector)
        
        return importance_vector

    def update_from_global_evaluation(self, model_weights_list, importance_vector, global_loss):
        """Update W_m based on global model performance (better learning signal)"""
        if self.previous_global_loss is not None:
            # Use performance delta: positive = model improved, negative = model got worse
            performance_delta = self.previous_global_loss - global_loss
            
            print(f"{bcolors.OKCYAN}Updating W_m based on global performance delta: {performance_delta:.6f}{bcolors.ENDC}")
            
            num_clients = len(model_weights_list)
            for client_idx in range(num_clients):
                client_params = self._flatten_weights(model_weights_list[client_idx])
                M_i = np.tanh(client_params)
                
                # Update based on global performance improvement and client importance
                gradient = importance_vector[client_idx] * M_i * performance_delta
                
                # Update W_m column for this client
                self.W_m[:, client_idx] += self.learning_rate * gradient
        
        # Store current global loss for next round
        self.previous_global_loss = global_loss
