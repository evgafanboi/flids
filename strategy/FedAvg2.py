import numpy as np
import logging

class FedAvg2:
    def __init__(self):
        self.name = "FedAvg2"
    
    def aggregate(self, weights_list, sample_sizes, client_losses=None):
        logger = logging.getLogger()
        
        if client_losses is None:
            # Log uniform weighting when no losses provided
            uniform_weights = [s / sum(sample_sizes) for s in sample_sizes]
            for i, weight in enumerate(uniform_weights):
                logger.info(f"FedAvg2 - Client {i}: Importance = {weight:.4f} (sample-size only)")
            return self._weighted_average(weights_list, sample_sizes)
        
        # Apply softmax to negative losses for stable weights
        neg_losses = [-loss for loss in client_losses]
        exp_losses = np.exp(neg_losses - np.max(neg_losses))
        loss_weights = exp_losses / np.sum(exp_losses)
        
        combined_weights = [sample_size * loss_weight for sample_size, loss_weight in zip(sample_sizes, loss_weights)]
        
        # Log each client's importance
        total_combined = sum(combined_weights)
        for i, (loss, loss_weight, sample_size, combined_weight) in enumerate(zip(client_losses, loss_weights, sample_sizes, combined_weights)):
            importance = combined_weight / total_combined
            logger.info(f"FedAvg2 - Client {i}: Loss={loss:.4f}, LossWeight={loss_weight:.4f}, SampleSize={sample_size}, Importance={importance:.4f}")
        
        return self._weighted_average(weights_list, combined_weights)
    
    def _weighted_average(self, weights_list, weights):
        """Weighted average for nested weight structures"""
        if not weights_list:
            return None
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Initialize result with zeros like the first weight structure
        num_layers = len(weights_list[0])
        averaged_weights = []
        
        for layer_idx in range(num_layers):
            layer_sum = np.zeros_like(weights_list[0][layer_idx])
            for client_idx, client_weights in enumerate(weights_list):
                layer_sum += normalized_weights[client_idx] * client_weights[layer_idx]
            averaged_weights.append(layer_sum)
        
        return averaged_weights
