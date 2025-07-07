import tensorflow as tf
import numpy as np

class FedProxAdaptiveModelWrapper:
    def __init__(self, base_model, strategy, client_id):
        self.base_model = base_model
        self.strategy = strategy
        self.client_id = client_id
        self.model = base_model.model if hasattr(base_model, 'model') else base_model
        self.global_weights = None
        
    def set_weights(self, weights):
        self.base_model.set_weights(weights)
        self.global_weights = [w.copy() for w in weights]
    
    def fit(self, *args, **kwargs):
        # Get current mu for this client
        current_mu = self.strategy.get_client_mu(self.client_id)
        
        # Create custom training step with adaptive proximal term
        if current_mu != 0 and self.global_weights is not None:
            original_train_step = self.model.train_step
            
            def proximal_train_step(data):
                with tf.GradientTape() as tape:
                    x, y = data
                    y_pred = self.model(x, training=True)
                    loss = self.model.compiled_loss(y, y_pred, regularization_losses=self.model.losses)
                    
                    # Add proximal term: μ/2 * ||w - w_global||^2
                    if current_mu != 0:
                        current_weights = self.model.trainable_variables
                        proximal_loss = 0.0
                        
                        for i, (current_w, global_w) in enumerate(zip(current_weights, self.global_weights)):
                            if i < len(self.global_weights):
                                diff = current_w - tf.convert_to_tensor(global_w, dtype=current_w.dtype)
                                proximal_loss += tf.reduce_sum(tf.square(diff))
                        
                        # Note: negative mu encourages drift, positive mu discourages it
                        loss += (current_mu / 2.0) * proximal_loss
                
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.model.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
                
                self.model.compiled_metrics.update_state(y, y_pred)
                return {m.name: m.result() for m in self.model.metrics}
            
            # Temporarily replace training step
            self.model.train_step = proximal_train_step
            
            try:
                history = self.base_model.fit(*args, **kwargs)
            finally:
                # Restore original training step
                self.model.train_step = original_train_step
        else:
            # Standard training without proximal term
            history = self.base_model.fit(*args, **kwargs)
        
        return history
    
    def predict(self, *args, **kwargs):
        return self.base_model.predict(*args, **kwargs)
    
    def evaluate(self, *args, **kwargs):
        return self.base_model.evaluate(*args, **kwargs)
    
    def get_weights(self):
        return self.base_model.get_weights()

def create_fedprox_adaptive_dense_model(input_dim, num_classes, batch_size, strategy, client_id):
    from .dense import create_enhanced_dense_model
    base_model = create_enhanced_dense_model(input_dim, num_classes, batch_size)
    return FedProxAdaptiveModelWrapper(base_model, strategy, client_id)

def create_fedprox_adaptive_gru_model(input_shape, num_classes, batch_size, strategy, client_id):
    from .gru import create_enhanced_gru_model
    base_model = create_enhanced_gru_model(input_shape, num_classes, batch_size)
    return FedProxAdaptiveModelWrapper(base_model, strategy, client_id)
