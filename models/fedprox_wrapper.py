import tensorflow as tf
import numpy as np

class FedProxModelWrapper:
    def __init__(self, base_model, fedprox_strategy):
        self.base_model = base_model
        self.fedprox = fedprox_strategy
        self.model = base_model.model if hasattr(base_model, 'model') else base_model
        self._global_weights_at_start = None
        self._compiled_with_proximal = False
    
    def _setup_proximal_loss(self):
        if self._global_weights_at_start is None:
            return
        
        # Convert global weights to TensorFlow constants
        global_weights_tf = [tf.constant(w, dtype=tf.float32) for w in self._global_weights_at_start]
        mu = self.fedprox.mu
        model_weights = self.model.weights
        
        # Get original loss function
        original_loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05)
        
        def proximal_loss(y_true, y_pred):
            ce_loss = original_loss(y_true, y_pred)
            
            proximal_term = tf.constant(0.0, dtype=tf.float32)
            
            # Calculate proximal term against the ORIGINAL global weights
            for current_w, global_w in zip(model_weights, global_weights_tf):
                if current_w.shape == global_w.shape:
                    diff = tf.cast(current_w, tf.float32) - global_w
                    proximal_term += tf.reduce_sum(tf.square(diff))
            
            return ce_loss + (mu / 2.0) * proximal_term
        
        # Recompile with proximal loss
        self.model.compile(
            optimizer=self.model.optimizer,
            loss=proximal_loss,
            metrics=['accuracy']
        )
        self._compiled_with_proximal = True
    
    def set_weights(self, weights):
        # Store the global weights BEFORE setting them
        if self.fedprox.global_weights is not None:
            self._global_weights_at_start = [w.copy() for w in self.fedprox.global_weights]
        
        # Set the weights to the model
        self.base_model.set_weights(weights)
        
        # Setup proximal loss AFTER setting weights
        if self._global_weights_at_start is not None and not self._compiled_with_proximal:
            self._setup_proximal_loss()
    
    def fit(self, *args, **kwargs):
        return self.base_model.fit(*args, **kwargs)
    
    def predict(self, *args, **kwargs):
        return self.base_model.predict(*args, **kwargs)
    
    def evaluate(self, *args, **kwargs):
        return self.base_model.evaluate(*args, **kwargs)
    
    def get_weights(self):
        return self.base_model.get_weights()

def create_fedprox_dense_model(input_dim, num_classes, batch_size, fedprox_strategy):
    from .dense import create_enhanced_dense_model
    base_model = create_enhanced_dense_model(input_dim, num_classes, batch_size)
    return FedProxModelWrapper(base_model, fedprox_strategy)

def create_fedprox_gru_model(input_shape, num_classes, batch_size, fedprox_strategy):
    from .gru import create_enhanced_gru_model
    base_model = create_enhanced_gru_model(input_shape, num_classes, batch_size)
    return FedProxModelWrapper(base_model, fedprox_strategy)
