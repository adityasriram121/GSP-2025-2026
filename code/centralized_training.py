"""
Federated Urban Heat Island (FedUHI) - Centralized Training Module

This module implements traditional centralized machine learning training
for urban temperature prediction using all zone data combined.

The model predicts temperature based on:
- Hour of day (cyclical encoding)
- Humidity level
- Zone identifier (one-hot encoded)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import os
import pickle
import matplotlib.pyplot as plt


class CentralizedTemperaturePredictor:
    """Centralized ML model for temperature prediction across urban zones."""
    
    def __init__(self, seed=42):
        """Initialize the centralized predictor with random seed."""
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        self.model = None
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.training_history = None
        self.training_time = 0
        self.feature_names = []
        
    def prepare_features(self, data):
        """
        Prepare features for the neural network.
        
        Args:
            data (pd.DataFrame): Input data with timestamp, temperature, humidity, zone columns
            
        Returns:
            tuple: (X, y) where X is features and y is target temperature
        """
        df = data.copy()
        
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        df['zone_encoded'] = self.label_encoder.fit_transform(df['zone'])
        
        zone_dummies = pd.get_dummies(df['zone'], prefix='zone')
        df = pd.concat([df, zone_dummies], axis=1)
        
        feature_columns = ['humidity', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        feature_columns.extend(zone_dummies.columns.tolist())
        
        self.feature_names = feature_columns
        
        X = df[feature_columns].values
        y = df['temperature'].values
        
        return X, y
    
    def build_model(self, input_dim):
        """
        Build a lightweight neural network model.
        
        Args:
            input_dim (int): Number of input features
            
        Returns:
            tf.keras.Model: Compiled neural network model
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(input_dim,)),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(0.1),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None, epochs=50, batch_size=32):
        """
        Train the centralized model.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training targets
            X_val (np.array): Validation features (optional)
            y_val (np.array): Validation targets (optional)
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            dict: Training history
        """
        print("Starting centralized training...")
        start_time = time.time()
        
        X_train_scaled = self.scaler_features.fit_transform(X_train)
        y_train_scaled = self.scaler_target.fit_transform(y_train.reshape(-1, 1)).flatten()
        
        self.model = self.build_model(X_train_scaled.shape[1])
        
        validation_data = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler_features.transform(X_val)
            y_val_scaled = self.scaler_target.transform(y_val.reshape(-1, 1)).flatten()
            validation_data = (X_val_scaled, y_val_scaled)
        
        history = self.model.fit(
            X_train_scaled, y_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            verbose=0,
            callbacks=[]
        )
        
        self.training_time = time.time() - start_time
        self.training_history = history.history
        
        print(f"Centralized training completed in {self.training_time:.2f} seconds")
        
        return history.history
    
    def predict(self, X):
        """
        Make temperature predictions.
        
        Args:
            X (np.array): Input features
            
        Returns:
            np.array: Predicted temperatures
        """
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        X_scaled = self.scaler_features.transform(X)
        y_pred_scaled = self.model.predict(X_scaled, verbose=0)
        y_pred = self.scaler_target.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        return y_pred
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model performance.
        
        Args:
            X_test (np.array): Test features
            y_test (np.array): Test targets
            
        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        return metrics, y_pred
    
    def save_model(self, filepath):
        """
        Save the trained model and preprocessors.
        
        Args:
            filepath (str): Base filepath for saving (without extension)
        """
        if self.model is None:
            raise ValueError("No model to save")
        
        self.model.save(f"{filepath}_model")
        
        with open(f"{filepath}_scaler_features.pkl", 'wb') as f:
            pickle.dump(self.scaler_features, f)
        
        with open(f"{filepath}_scaler_target.pkl", 'wb') as f:
            pickle.dump(self.scaler_target, f)
        
        with open(f"{filepath}_label_encoder.pkl", 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        with open(f"{filepath}_feature_names.pkl", 'wb') as f:
            pickle.dump(self.feature_names, f)
        
        print(f"Model saved to {filepath}_*")
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history (loss and metrics over epochs).
        
        Args:
            save_path (str): Optional path to save the plot
        """
        if self.training_history is None:
            raise ValueError("No training history available")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.training_history['loss'], label='Training Loss', linewidth=2)
        if 'val_loss' in self.training_history:
            ax1.plot(self.training_history['val_loss'], label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (MSE)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(self.training_history['mae'], label='Training MAE', linewidth=2)
        if 'val_mae' in self.training_history:
            ax2.plot(self.training_history['val_mae'], label='Validation MAE', linewidth=2)
        ax2.set_title('Mean Absolute Error Over Epochs', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE (°C)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training history plot saved to {save_path}")


def train_centralized_model(data_path, test_split=0.2, results_dir='results', models_dir='models'):
    """
    Train centralized model on combined zone data.
    
    Args:
        data_path (str): Path to combined dataset CSV
        test_split (float): Fraction of data to use for testing
        results_dir (str): Directory to save results
        models_dir (str): Directory to save models
        
    Returns:
        tuple: (model, metrics, training_time)
    """
    print("Loading combined zone data...")
    data = pd.read_csv(data_path)
    
    predictor = CentralizedTemperaturePredictor(seed=42)
    
    print("Preparing features...")
    X, y = predictor.prepare_features(data)
    
    n_samples = len(X)
    n_train = int(n_samples * (1 - test_split))
    
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    
    val_split = 0.2
    n_val = int(len(X_train) * val_split)
    val_indices = np.random.choice(len(X_train), n_val, replace=False)
    train_indices = np.setdiff1d(np.arange(len(X_train)), val_indices)
    
    X_val = X_train[val_indices]
    y_val = y_train[val_indices]
    X_train = X_train[train_indices]
    y_train = y_train[train_indices]
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    
    predictor.train(X_train, y_train, X_val, y_val, epochs=50, batch_size=32)
    
    print("\nEvaluating model...")
    metrics, predictions = predictor.evaluate(X_test, y_test)
    
    from code.model_manager import ModelManager
    model_manager = ModelManager(models_dir=models_dir)
    model_manager.save_centralized_model(
        model=predictor.model,
        metrics=metrics,
        training_time=predictor.training_time,
        config={'test_split': 0.2, 'epochs': 50}
    )
    
    predictor.plot_training_history(os.path.join(results_dir, 'centralized_training_history.png'))
    
    print("\n" + "="*50)
    print("CENTRALIZED MODEL RESULTS")
    print("="*50)
    print(f"Training time: {predictor.training_time:.2f} seconds")
    print(f"Test RMSE: {metrics['rmse']:.3f}°C")
    print(f"Test MAE: {metrics['mae']:.3f}°C")
    print(f"Test R²: {metrics['r2']:.3f}")
    
    return predictor, metrics, predictor.training_time








