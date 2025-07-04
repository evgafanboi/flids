import tensorflow as tf
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

# def create_dense_model(input_dim, num_classes):
#     model = tf.keras.Sequential([
#         tf.keras.layers.InputLayer(input_shape=(input_dim,)),
#         tf.keras.layers.Dense(256, activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(64, activation='relu'),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(num_classes, activation='softmax')
#     ])
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
#         loss=tf.keras.losses.CategoricalCrossentropy(),
#         metrics=['accuracy']
#     )
#     return model

# accuracy-focused dense model with regularization and learning rate scheduling (alt2)
# def create_dense_model(input_dim, num_classes):
#     model = tf.keras.Sequential([
#         tf.keras.layers.InputLayer(input_shape=(input_dim,)),
#         tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(num_classes, activation='softmax')
#     ])
    
#     # Learning rate scheduler for more stable convergence
#     lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
#         initial_learning_rate=0.0003,  # Lower initial rate
#         decay_steps=1000,
#         decay_rate=0.9,
#         staircase=True
#     )
    
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
#         loss=tf.keras.losses.CategoricalCrossentropy(),
#         metrics=['loss']
#     )
#     return model

# alt3
# def create_dense_model(input_dim, num_classes, batch_size=4096):
#     inputs = tf.keras.layers.Input(shape=(input_dim,))
    
#     # Normalization layer
#     x = tf.keras.layers.BatchNormalization()(inputs)
    
#     x = tf.keras.layers.Dense(128, activation='relu')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Dropout(0.2)(x)
    
#     # Second block with residual connection
#     residual = x
#     x = tf.keras.layers.Dense(64, activation='relu')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Dropout(0.15)(x)
    
#     # Project residual (128 -> 64)
#     residual = tf.keras.layers.Dense(64, activation=None)(residual)
#     x = tf.keras.layers.add([x, residual])
    
#     # Third block
#     x = tf.keras.layers.Dense(64, activation='relu')(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Dropout(0.1)(x)
    
#     # Output layer
#     outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
#     base_lr = 0.001
#     # Linear scaling for large batches
#     scaled_lr = base_lr * (batch_size / 1024)  # Scale from baseline 1024
    
#     lr_schedule = tf.keras.optimizers.schedules.CosineDecayRestarts(
#         scaled_lr,
#         first_decay_steps=500,
#         t_mul=1.5,  # Less aggressive restart
#         m_mul=0.95,  # Slower decay
#         alpha=0.001  # Higher minimum LR
#     )
    
#     model.compile(
#         optimizer=tf.keras.optimizers.Adam(
#             learning_rate=lr_schedule,
#             clipnorm=1.0,
#             beta_1=0.9,
#             beta_2=0.999,
#             epsilon=1e-7
#         ),
#         loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
#         metrics=['accuracy', precision_m, recall_m, f1_m]
#     )
#     return model

def create_dense_model(input_dim, num_classes, batch_size=4096):
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy', precision_m, recall_m, f1_m]
    )
    return model