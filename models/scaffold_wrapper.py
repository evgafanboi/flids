import tensorflow as tf
import numpy as np

class SCAFFOLDModelWrapper:
    def __init__(self, base_model, scaffold_strategy, client_id):
        self.base_model = base_model
        self.scaffold = scaffold_strategy
        self.client_id = client_id
        self.model = base_model.model if hasattr(base_model, 'model') else base_model
        
        # SCAFFOLD control variates
        self.global_control = None
        self.client_control = None
        self.old_client_control = None
        self.initial_weights = None
        self.control_correction = None
        
    def set_weights(self, weights):
        self.base_model.set_weights(weights)
        self.initial_weights = [w.copy() for w in weights]
        
        # Get control variates from strategy
        self.global_control = self.scaffold.get_global_control()
        self.client_control = self.scaffold.get_client_control(self.client_id)
        self.old_client_control = [c.copy() for c in self.client_control] if self.client_control else None
        
        # Compute control correction: c_i - c (client control - global control)
        if self.global_control and self.client_control:
            self.control_correction = [c_i - c for c_i, c in zip(self.client_control, self.global_control)]
        else:
            self.control_correction = None
    
    def fit(self, *args, **kwargs):
        # Apply SCAFFOLD control variate correction by modifying weights before training
        if self.control_correction:
            # Get learning rate
            lr = self._get_learning_rate()
            
            # Apply correction to initial weights: w' = w - η * (c_i - c)
            current_weights = self.get_weights()
            corrected_weights = []
            for i, (w, correction) in enumerate(zip(current_weights, self.control_correction)):
                corrected_w = w - lr * correction
                corrected_weights.append(corrected_w)
            
            # Set corrected weights
            self.base_model.set_weights(corrected_weights)
        
        # Standard training
        history = self.base_model.fit(*args, **kwargs)
        
        # Compute SCAFFOLD updates after training
        final_weights = self.get_weights()
        
        if self.initial_weights and self.global_control and self.client_control:
            # Compute weight difference (from initial, not corrected weights)
            weight_diff = [fw - iw for fw, iw in zip(final_weights, self.initial_weights)]
            
            # Extract learning rate from optimizer
            lr = self._get_learning_rate()
            
            # Update client control: c_i^{t+1} = c_i^t - c + (1/η) * (x^{t+1} - x^t)
            new_client_control = []
            for i in range(len(self.client_control)):
                control_update = (self.client_control[i] - self.global_control[i] + 
                                (1.0 / lr) * weight_diff[i])
                new_client_control.append(control_update)
            
            # Store updated control
            self.scaffold.update_client_control(self.client_id, new_client_control)
            
            # Compute control update for server
            if self.old_client_control:
                control_update = [new_c - old_c for new_c, old_c in 
                                zip(new_client_control, self.old_client_control)]
            else:
                control_update = new_client_control
            
            # Store for aggregation
            self._control_update = control_update
        
        return history
    
    def _get_learning_rate(self):
        """Extract current learning rate from optimizer"""
        try:
            optimizer = self.model.optimizer
            if hasattr(optimizer, 'learning_rate'):
                lr = optimizer.learning_rate
                if hasattr(lr, 'numpy'):
                    return float(lr.numpy())
                else:
                    return float(lr)
            else:
                return 0.001  # Fallback
        except:
            return 0.001  # Fallback
    
    def get_scaffold_update(self):
        """Get SCAFFOLD update data for server aggregation"""
        if hasattr(self, '_control_update'):
            return {
                'weights': self.get_weights(),
                'control_update': self._control_update,
                'client_id': self.client_id
            }
        else:
            return {'weights': self.get_weights(), 'control_update': None, 'client_id': self.client_id}
    
    def predict(self, *args, **kwargs):
        return self.base_model.predict(*args, **kwargs)
    
    def evaluate(self, *args, **kwargs):
        return self.base_model.evaluate(*args, **kwargs)
    
    def get_weights(self):
        return self.base_model.get_weights()

def create_scaffold_dense_model(input_dim, num_classes, batch_size, scaffold_strategy, client_id):
    from .dense import create_enhanced_dense_model
    base_model = create_enhanced_dense_model(input_dim, num_classes, batch_size)
    return SCAFFOLDModelWrapper(base_model, scaffold_strategy, client_id)

def create_scaffold_gru_model(input_shape, num_classes, batch_size, scaffold_strategy, client_id):
    from .gru import create_enhanced_gru_model
    base_model = create_enhanced_gru_model(input_shape, num_classes, batch_size)
    return SCAFFOLDModelWrapper(base_model, scaffold_strategy, client_id)
