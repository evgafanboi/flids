import numpy as np
import logging

class FedProxAdaptive:
    def __init__(self, base_mu=0, negative_mu_factor=0.5):
        self.name = "FedProxAdaptive"
        self.base_mu = base_mu
        self.negative_mu_factor = negative_mu_factor
        self.client_mu_values = {}
    
    def compute_adaptive_mu(self, client_losses, client_ids):
        """Compute adaptive mu values based on client training losses"""
        logger = logging.getLogger()
        
        # Create list of (loss, client_id) pairs and sort by loss
        loss_client_pairs = list(zip(client_losses, client_ids))
        sorted_pairs = sorted(loss_client_pairs, key=lambda x: x[0])  # Sort by loss (ascending)
        
        num_clients = len(client_losses)
        
        # Top 10% get negative mu (encourage drift)
        top_10_percent = max(1, num_clients // 10)
        
        # Bottom 20% get high mu (force close to global)
        bottom_20_percent = max(1, num_clients // 5)
        
        # Create mu_values list in the same order as original client_ids
        mu_dict = {}
        
        for rank, (loss, client_id) in enumerate(sorted_pairs):
            if rank < top_10_percent:
                # Top performers get negative mu to encourage specialization
                mu = -0.05
                logger.info(f"FedProxAdaptive - Client {client_id}: TOP PERFORMER (rank {rank+1}/{num_clients}) - Loss={loss:.4f}, μ={mu:.4f} (ENCOURAGE DRIFT)")
            elif rank >= num_clients - bottom_20_percent:
                # Bottom performers get high mu to stay close to global
                mu = 0.5
                logger.info(f"FedProxAdaptive - Client {client_id}: STRAGGLER (rank {rank+1}/{num_clients}) - Loss={loss:.4f}, μ={mu:.4f} (FORCE CLOSE)")
            else:
                # Middle performers get FedAvg
                mu = 0
                logger.info(f"FedProxAdaptive - Client {client_id}: STANDARD (rank {rank+1}/{num_clients}) - Loss={loss:.4f}, μ={mu:.4f}")
            
            mu_dict[client_id] = mu
        
        # Return mu values in original client_ids order
        mu_values = [mu_dict[client_id] for client_id in client_ids]
        self.client_mu_values.update(mu_dict)
        
        return mu_values
    
    def get_client_mu(self, client_id):
        """Get the current mu value for a specific client"""
        return self.client_mu_values.get(client_id, self.base_mu)
    
    def aggregate(self, weights_list, sample_sizes, client_losses=None, client_ids=None):
        """Standard FedAvg aggregation - mu values are handled in the model wrapper"""
        if client_losses is not None and client_ids is not None:
            # Compute adaptive mu values for next round
            self.compute_adaptive_mu(client_losses, client_ids)
        
        # Standard weighted average aggregation
        total_samples = sum(sample_sizes)
        normalized_weights = [s / total_samples for s in sample_sizes]
        
        if not weights_list:
            return None
        
        num_layers = len(weights_list[0])
        averaged_weights = []
        
        for layer_idx in range(num_layers):
            layer_sum = np.zeros_like(weights_list[0][layer_idx])
            for client_idx, client_weights in enumerate(weights_list):
                layer_sum += normalized_weights[client_idx] * client_weights[layer_idx]
            averaged_weights.append(layer_sum)
        
        return averaged_weights
