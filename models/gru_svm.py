import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

class GRUModel:
    
    def __init__(self, input_shape, num_classes, batch_size=1024):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size
        
        # Create the GRU model
        self.model = self._create_gru_model()
    
    def _create_gru_model(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        
        # Input normalization
        x = tf.keras.layers.BatchNormalization()(inputs)
        
        x = tf.keras.layers.GRU(
            128,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.1,
            name='gru_layer_1'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Second GRU layer
        x = tf.keras.layers.GRU(
            64,
            return_sequences=False,
            dropout=0.2,
            recurrent_dropout=0.1,
            name='gru_layer_2'
        )(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.3)(x)
        
        residual = x
        x = tf.keras.layers.Dense(128, activation='relu', name='dense_1')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        if residual.shape[-1] != 128:
            residual = tf.keras.layers.Dense(128, activation=None)(residual)
        x = tf.keras.layers.add([x, residual])
        
        # Final dense layer
        x = tf.keras.layers.Dense(64, activation='relu', name='dense_2')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        
        # Output layer
        outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax', name='output')(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001,
                clipnorm=1.0
            ),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy', precision_m, recall_m, f1_m]
            # metrics=['accuracy']
        )
        
        return model
    
    def fit(self, dataset, epochs=5, **kwargs):
        # Remove verbose from kwargs to avoid conflict
        kwargs.pop('verbose', None)
        
        return self.model.fit(
            dataset,
            epochs=epochs,
            verbose=1,
            **kwargs
        )
    
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X, batch_size=self.batch_size, verbose=0)
    
    def predict_proba(self, X):
        """Predict probabilities (same as predict for this model)"""
        return self.predict(X)
    
    def evaluate(self, dataset, verbose=0):
        """Evaluate the model"""
        return self.model.evaluate(dataset, verbose=verbose)
    
    def get_weights(self):
        """Get model weights for federated learning"""
        return self.model.get_weights()
    
    def set_weights(self, weights):
        """Set model weights for federated learning"""
        self.model.set_weights(weights)
    
    def get_hidden_states(self, X):
        """Extract GRU hidden states for FedAGRU - using the second GRU layer"""
        try:
            # Get the second GRU layer output (final hidden state)
            hidden_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer('gru_layer_2').output
            )
            hidden_states = hidden_model.predict(X, verbose=0, batch_size=self.batch_size)
            print(f"Extracted GRU hidden states: shape {hidden_states.shape}")
            return hidden_states
        except Exception as e:
            print(f"Error extracting hidden states: {e}")
            # Fallback to first GRU layer if second fails
            try:
                hidden_model = tf.keras.Model(
                    inputs=self.model.input,
                    outputs=self.model.get_layer('gru_layer_1').output
                )
                # Get last timestep of sequence output
                hidden_states = hidden_model.predict(X, verbose=0, batch_size=self.batch_size)
                hidden_states = hidden_states[:, -1, :]  # Take last timestep
                print(f"Extracted GRU hidden states (fallback): shape {hidden_states.shape}")
                return hidden_states
            except Exception as e2:
                print(f"Error extracting hidden states (fallback): {e2}")
                return None

    def _warmup_schedule(self, epoch):
        """Learning rate warmup schedule for large batch training"""
        base_lr = 0.001
        batch_size_factor = self.batch_size / 1024
        scaled_lr = base_lr * np.sqrt(batch_size_factor)
        
        # Warmup for first 3 epochs
        if epoch < 3:
            return scaled_lr * (epoch + 1) / 3
        else:
            return scaled_lr

def create_enhanced_gru_model(input_shape, num_classes, batch_size=1024):
    """Create enhanced GRU model"""
    return GRUModel(input_shape, num_classes, batch_size)

# Keep old function name for compatibility
def create_enhanced_gru_svm_model(input_shape, num_classes, batch_size=1024):
    """Create GRU model (legacy name for compatibility)"""
    return GRUModel(input_shape, num_classes, batch_size)