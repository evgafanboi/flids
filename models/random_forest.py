import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import time

class KerasCompatibleRandomForest:
    """
    A wrapper for sklearn's RandomForestClassifier that provides a Keras-like interface
    for compatibility with the federated learning system.
    """
    
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, 
                 min_samples_leaf=1, max_features='sqrt', n_jobs=-1, random_state=42):
        """
        Initialize the random forest model with configurable hyperparameters.
        """
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=0
        )
        self.classes_ = None
        self.history = {'loss': [], 'accuracy': [], 'precision_m': [], 'recall_m': [], 'f1_m': []}
        self.weights_file = f"rf_weights_{int(time.time())}.joblib"
    
    def fit(self, x, y, epochs=1, batch_size=None, verbose=0, callbacks=None, validation_split=0.0, **kwargs):
        """
        Train the random forest model.
        
        Args:
            x: Training features
            y: Training labels (one-hot encoded)
            epochs: Ignored (included for API compatibility)
            batch_size: Ignored (included for API compatibility)
            verbose: Controls output verbosity
            callbacks: Ignored (included for API compatibility)
            validation_split: Ignored (included for API compatibility)
            
        Returns:
            Self for history tracking compatibility
        """
        if verbose:
            print("Training Random Forest model...")
        
        # Convert from one-hot encoding to class indices if needed
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_indices = np.argmax(y, axis=1)
        else:
            y_indices = y
            
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
            
        # Fit the model
        self.model.fit(x, y_indices)
        self.classes_ = self.model.classes_
        
        # Calculate metrics on training data
        train_preds = self.predict(x)
        train_acc = np.mean(np.argmax(train_preds, axis=1) == y_indices)
        
        # Add metrics to history
        self.history['accuracy'].append(train_acc)
        self.history['loss'].append(1.0 - train_acc)  # Use 1-accuracy as a proxy for loss
        
        # Track precision, recall, f1 (simplified calculation)
        precision, recall, f1 = self._calculate_metrics(y_indices, np.argmax(train_preds, axis=1))
        self.history['precision_m'].append(precision)
        self.history['recall_m'].append(recall)
        self.history['f1_m'].append(f1)
        
        if verbose:
            print(f"Training completed - Accuracy: {train_acc:.4f}")
            
        return self
    
    def predict(self, x, batch_size=None, verbose=0):
        """
        Generate predictions.
        
        Args:
            x: Input features
            batch_size: Ignored (included for API compatibility)
            verbose: Controls output verbosity
            
        Returns:
            Probability predictions in one-hot encoded format
        """
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
            
        # Get probability predictions
        y_pred_proba = self.model.predict_proba(x)
        
        return y_pred_proba
    
    def evaluate(self, x, y, batch_size=None, verbose=0):
        """
        Evaluate the model on test data.
        
        Args:
            x: Test features
            y: Test labels (one-hot encoded)
            batch_size: Ignored (included for API compatibility)
            verbose: Controls output verbosity
            
        Returns:
            List containing [loss, accuracy, precision, recall, f1]
        """
        if verbose:
            print("Evaluating Random Forest model...")
            
        # Flatten input if needed
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
            
        # Convert from one-hot encoding to class indices if needed
        if len(y.shape) > 1 and y.shape[1] > 1:
            y_indices = np.argmax(y, axis=1)
        else:
            y_indices = y
            
        # Get predictions
        y_pred = self.predict(x)
        y_pred_indices = np.argmax(y_pred, axis=1)
        
        # Calculate accuracy
        accuracy = np.mean(y_pred_indices == y_indices)
        
        # Calculate other metrics
        precision, recall, f1 = self._calculate_metrics(y_indices, y_pred_indices)
        
        # Use 1-accuracy as a proxy for loss
        loss = 1.0 - accuracy
        
        if verbose:
            print(f"Evaluation results - Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return [loss, accuracy, precision, recall, f1]
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculate precision, recall and F1 score for multi-class classification.
        Uses macro-averaging (calculate metrics for each class and average).
        
        Args:
            y_true: True class indices
            y_pred: Predicted class indices
            
        Returns:
            Tuple of (precision, recall, f1)
        """
        classes = np.unique(np.concatenate([y_true, y_pred]))
        
        precision_sum = 0
        recall_sum = 0
        f1_sum = 0
        classes_count = 0
        
        for cls in classes:
            true_positives = np.sum((y_true == cls) & (y_pred == cls))
            false_positives = np.sum((y_true != cls) & (y_pred == cls))
            false_negatives = np.sum((y_true == cls) & (y_pred != cls))
            
            # Avoid division by zero
            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives)
                precision_sum += precision
                classes_count += 1
            
            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives)
                recall_sum += recall
                
                if true_positives + false_positives > 0:
                    precision = true_positives / (true_positives + false_positives)
                    if precision + recall > 0:
                        f1 = 2 * precision * recall / (precision + recall)
                        f1_sum += f1
        
        # Calculate macro-average
        if classes_count > 0:
            precision = precision_sum / classes_count
            recall = recall_sum / classes_count
            f1 = f1_sum / classes_count
            return precision, recall, f1
        else:
            return 0.0, 0.0, 0.0
    
    def get_weights(self):
        """
        Get model weights - for compatibility with Keras API.
        For Random Forest, we serialize the model to disk and return a reference.
        
        Returns:
            List with a single item - the path to the saved model
        """
        joblib.dump(self.model, self.weights_file)
        return [self.weights_file]
    
    def set_weights(self, weights):
        """
        Set model weights - for compatibility with Keras API.
        For Random Forest, we load the serialized model from disk.
        
        Args:
            weights: List with a single item - the path to the saved model
        """
        if isinstance(weights, list) and len(weights) > 0:
            weights_file = weights[0]
            if isinstance(weights_file, str) and os.path.exists(weights_file):
                self.model = joblib.load(weights_file)
                self.classes_ = self.model.classes_
            else:
                print(f"Warning: Could not load weights from {weights_file}")


def create_random_forest_model(input_dim, num_classes, **kwargs):
    """
    Create a random forest model with Keras-compatible interface.
    
    Args:
        input_dim: Number of features (ignored but kept for API compatibility)
        num_classes: Number of output classes (ignored but kept for API compatibility)
        
    Returns:
        KerasCompatibleRandomForest instance
    """
    # You can adjust these hyperparameters based on your needs
    return KerasCompatibleRandomForest(
        n_estimators=100,         # Number of trees
        max_depth=20,             # Maximum tree depth
        min_samples_split=5,      # Min samples to split an internal node
        min_samples_leaf=2,       # Min samples at a leaf node
        max_features='sqrt',      # Number of features to consider for best split
        n_jobs=-1,                # Use all available cores
        random_state=42           # For reproducibility
    )