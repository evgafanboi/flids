import numpy as np

class FedAvg:
    """
    Implementation of Federated Averaging (FedAvg) aggregation strategy.
    
    FedAvg aggregates model weights based on the number of samples each client has.
    """
    
    def __init__(self):
        """Initialize the FedAvg strategy."""
        self.name = "FedAvg"
    
    def aggregate(self, model_weights_list, sample_sizes=None):
        """
        Aggregate model weights using FedAvg strategy with proper weighting.
        
        Args:
            model_weights_list: List of model weights from different clients
            sample_sizes: List of dataset sizes for each client (number of samples)
            
        Returns:
            Aggregated model weights
        """
        if not model_weights_list:
            raise ValueError("Cannot aggregate empty weights list")
        
        # If sample sizes not provided, use equal weighting
        if sample_sizes is None:
            sample_sizes = [1] * len(model_weights_list)
        
        # Get the shape of weights
        weights_shape = [w.shape for w in model_weights_list[0]]
        
        # Initialize aggregated weights with zeros
        aggregated_weights = [np.zeros(shape) for shape in weights_shape]
        
        # Sum weighted weights
        total_samples = sum(sample_sizes)
        for i, weights in enumerate(model_weights_list):
            # Get weight based on client dataset size
            client_weight = sample_sizes[i] / total_samples if total_samples > 0 else 1.0 / len(model_weights_list)
            
            for j, layer_weights in enumerate(weights):
                aggregated_weights[j] += client_weight * layer_weights
        
        return aggregated_weights