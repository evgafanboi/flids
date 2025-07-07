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

class DenseModel:
    """Enhanced Dense Model matching GRU's architectural principles"""
    
    def __init__(self, input_dim, num_classes, batch_size=4096, learning_rate=None):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        
        # Auto-scale learning rate based on batch size (same as GRU)
        if learning_rate is None:
            base_lr = 0.001
            self.learning_rate = base_lr * np.sqrt(batch_size / 1024)
        else:
            self.learning_rate = learning_rate
            
        print(f"Dense Model - Using learning rate: {self.learning_rate:.6f} for batch size: {batch_size}")
        
        # Create the Dense model
        self.model = self._create_dense_model()
    
    def _create_dense_model(self):
        """Create dense model with architecture matching GRU principles"""
        inputs = tf.keras.layers.Input(shape=(self.input_dim,))
        
        # Input normalization - use LayerNorm for consistency with GRU
        x = tf.keras.layers.LayerNormalization()(inputs)
        
        # First dense block - equivalent to GRU hidden size
        x = tf.keras.layers.Dense(
            128,  # Match GRU's effective capacity
            activation='swish',  # Same activation as GRU
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name='dense_1'
        )(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.15)(x)  # Match GRU dropout
        
        # Second dense block with residual connection
        residual = x
        x = tf.keras.layers.Dense(
            128,
            activation='swish',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name='dense_2'
        )(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Add residual connection
        x = tf.keras.layers.add([x, residual])
        
        # Third dense block - match GRU's final dense layer
        x = tf.keras.layers.Dense(
            64, 
            activation='swish',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name='dense_3'
        )(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.15)(x)
        
        # Output layer - no regularization (same as GRU)
        outputs = tf.keras.layers.Dense(
            self.num_classes, 
            activation='softmax', 
            name='output'
        )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Use same optimizer strategy as GRU
        if self.batch_size >= 2048:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-6,
                clipnorm=1.0
            )
        else:
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                clipnorm=1.0
            )
        
        model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(
                label_smoothing=0.05  # Same label smoothing as GRU
            ),
            metrics=['accuracy', precision_m, recall_m, f1_m]
        )
        
        return model
    
    def get_callbacks(self, validation_data=None):
        """Get training callbacks matching GRU's schedule"""
        callbacks = []
        
        # Same learning rate schedule as GRU
        def lr_schedule(epoch, lr):
            if epoch < 5:
                return self.learning_rate * (epoch + 1) / 5
            else:
                decay_epochs = max(1, epoch - 5)
                return self.learning_rate * 0.5 * (1 + np.cos(np.pi * decay_epochs / 50))
        
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1))
        
        if validation_data is not None:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ))
            
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ))
        
        return callbacks
    
    def fit(self, dataset, epochs=50, validation_data=None, **kwargs):
        kwargs.pop('verbose', None)
        kwargs.pop('callbacks', None)
        
        callbacks = self.get_callbacks(validation_data)
        
        return self.model.fit(
            dataset,
            epochs=epochs,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1,
            **kwargs
        )
    
    def predict(self, X):
        """Make predictions with larger batch size for efficiency"""
        return self.model.predict(X, batch_size=min(self.batch_size, 8192), verbose=0)
    
    def predict_proba(self, X):
        """Predict probabilities"""
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

# Factory function to maintain compatibility
def create_enhanced_dense_model(input_dim, num_classes, batch_size=4096, learning_rate=None):
    """Create enhanced dense model that matches GRU performance"""
    return DenseModel(input_dim, num_classes, batch_size, learning_rate)

# Legacy function for backward compatibility
def create_dense_model(input_dim, num_classes):
    """Legacy function - creates enhanced model with default batch size"""
    return create_enhanced_dense_model(input_dim, num_classes, batch_size=1024)
