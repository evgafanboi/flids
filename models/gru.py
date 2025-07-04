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
    
    def __init__(self, input_shape, num_classes, batch_size=4096, learning_rate=None):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.batch_size = batch_size
        
        # Auto-scale learning rate based on batch size (linear scaling rule)
        if learning_rate is None:
            base_lr = 0.001
            self.learning_rate = base_lr * np.sqrt(batch_size / 1024)
        else:
            self.learning_rate = learning_rate
            
        print(f"Using learning rate: {self.learning_rate:.6f} for batch size: {batch_size}")
        
        # Create the GRU model
        self.model = self._create_gru_model()
    
    def _create_gru_model(self):
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        
        # Input normalization - use LayerNorm for better stability
        x = tf.keras.layers.LayerNormalization()(inputs)
        
        # Simplified architecture with fewer parameters
        # Single GRU layer with moderate size
        x = tf.keras.layers.GRU(
            96,  # Reduced from 128
            return_sequences=True,
            dropout=0.15,  # Reduced dropout
            recurrent_dropout=0.05,  # Reduced recurrent dropout
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            recurrent_regularizer=tf.keras.regularizers.l2(1e-4),
            name='gru_layer_1'
        )(x)
        
        # Global average pooling instead of taking last timestep
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        x = tf.keras.layers.LayerNormalization()(x)
        
        # Simplified dense layers
        x = tf.keras.layers.Dense(
            128, 
            activation='swish',  # Swish often works better than ReLU
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name='dense_1'
        )(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.2)(x)
        
        # Additional dense layer for complex classification
        x = tf.keras.layers.Dense(
            64, 
            activation='swish',
            kernel_regularizer=tf.keras.regularizers.l2(1e-4),
            name='dense_2'
        )(x)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dropout(0.15)(x)
        
        # Output layer with no regularization
        outputs = tf.keras.layers.Dense(
            self.num_classes, 
            activation='softmax', 
            name='output'
        )(x)
        
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Use different based on batch size
        if self.batch_size >= 2048:
            # LAMB optimizer for large batch training
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
                label_smoothing=0.05  # Reduced label smoothing
            ),
            metrics=['accuracy', precision_m, recall_m, f1_m]
        )
        
        return model
    
    def get_callbacks(self, validation_data=None):
        """Get training callbacks for better convergence"""
        callbacks = []
        
        # Learning rate scheduling
        def lr_schedule(epoch, lr):
            # Warmup for first 5 epochs, then cosine decay
            if epoch < 5:
                return self.learning_rate * (epoch + 1) / 5
            else:
                # Cosine decay
                decay_epochs = max(1, epoch - 5)
                return self.learning_rate * 0.5 * (1 + np.cos(np.pi * decay_epochs / 50))
        
        callbacks.append(tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1))
        
        # Early stopping if validation data is provided
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
    
    def get_hidden_states(self, X):
        """Extract GRU hidden states for FedAGRU"""
        try:
            # Get the GRU layer output after global average pooling
            hidden_model = tf.keras.Model(
                inputs=self.model.input,
                outputs=self.model.get_layer('gru_layer_1').output
            )
            hidden_states = hidden_model.predict(X, verbose=0, batch_size=self.batch_size)
            
            # Apply global average pooling to get fixed-size representation
            hidden_states = np.mean(hidden_states, axis=1)
            
            print(f"Extracted GRU hidden states: shape {hidden_states.shape}")
            return hidden_states
        except Exception as e:
            print(f"Error extracting hidden states: {e}")
            return None

def create_enhanced_gru_model(input_shape, num_classes, batch_size=4096, learning_rate=None):
    return GRUModel(input_shape, num_classes, batch_size, learning_rate)