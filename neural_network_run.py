import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt

def run_neural_network(X_train, y_train, X_test, y_test, 
                       hidden_layers=[64, 32], activation='relu', dropout_rates=[0.3, 0.2], 
                       batch_size=32, learning_rate=0.001, epochs=50):
    """
    Trains and evaluates a feedforward neural network (MLP) for binary classification.
    
    Parameters:
    - X_train, y_train: Training data (features, labels)
    - X_test, y_test: Test data (features, labels)
    - hidden_layers: List defining number of neurons per hidden layer (default: [64, 32])
    - activation: Activation function (default: 'relu')
    - dropout_rates: List defining dropout rate per hidden layer (default: [0.3, 0.2])
    - batch_size: Training batch size (default: 32)
    - learning_rate: Optimizer learning rate (default: 0.001)
    - epochs: Number of training epochs (default: 50)
    
    Returns:
    - Trained model
    - Accuracy & ROC-AUC score
    """
    
    model = Sequential()
    model.add(Dense(hidden_layers[0], activation=activation, input_shape=(X_train.shape[1],)))
    model.add(Dropout(dropout_rates[0]))

    for i in range(1, len(hidden_layers)):
        model.add(Dense(hidden_layers[i], activation=activation))
        model.add(Dropout(dropout_rates[i]))

    model.add(Dense(1, activation='sigmoid')) 

    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test),
                        epochs=epochs, 
                        batch_size=batch_size,
                        verbose=1)

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_prob)

    print(f'Model Evaluation:')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'ROC AUC Score: {roc_auc:.4f}')
    print("Classification Report:\n", classification_report(y_test, y_pred))

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Model Loss')
    
    plt.show()
    
    return model, accuracy, roc_auc
