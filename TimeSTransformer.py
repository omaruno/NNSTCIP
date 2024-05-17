import matplotlib.pyplot as plt
import numpy as np
import shap
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import layers, models

X = np.array([[(1,2,3,3,1),(3,2,1,3,2),(3,2,2,3,3),(2,2,1,1,2),(2,1,1,1,1)],
              [(4,5,6,4,4),(5,6,4,3,2),(5,5,6,1,3),(3,3,3,2,2),(2,3,3,2,1)],
              [(7,8,9,4,7),(7,7,6,7,8),(5,8,7,8,8),(6,7,6,7,8),(5,7,6,6,6)],
              [(7,8,9,8,6),(6,6,7,8,6),(8,7,8,8,8),(8,6,7,8,7),(8,6,7,8,8)],
              [(4,5,6,5,5),(5,5,5,6,4),(6,5,5,5,6),(4,4,3,3,3),(5,5,4,4,5)],
              [(4,5,6,5,5),(5,5,5,6,4),(6,5,5,5,6),(4,4,3,3,3),(5,5,4,4,5)],
              [(1,2,3,3,1),(3,2,1,3,2),(3,2,2,3,3),(2,2,1,1,2),(2,1,1,1,1)]])
y = np.array([0, 1, 2, 2, 1, 1, 0])

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res

def build_model(input_shape, head_size,num_heads,ff_dim,num_transformer_blocks,mlp_units,dropout=0,mlp_dropout=0,):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(3, activation="softmax")(x)
    return keras.Model(inputs, outputs)
input_shape = X.shape[1:]
print("Sono qua",input_shape)
model = build_model(input_shape,head_size=256,num_heads=4,ff_dim=4,num_transformer_blocks=1,mlp_units=[128],mlp_dropout=0.4,dropout=0.25,)

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    metrics=["sparse_categorical_accuracy"],
)

callbacks = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

model.fit(X,y,validation_split=0.2,epochs=10, batch_size=64,callbacks=callbacks,)

model.evaluate(X, y, verbose=1)

predictions = model.predict(X)
print(predictions)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

predicted_labels = predictions.argmax(axis=1)
accuracy = accuracy_score(y, predicted_labels)
precision = precision_score(y, predicted_labels, average='weighted')
recall = recall_score(y, predicted_labels, average='weighted')
f1 = f1_score(y, predicted_labels, average='weighted')
conf_matrix = confusion_matrix(y, predicted_labels)

# Printing the metrics
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix:\n{conf_matrix}')
