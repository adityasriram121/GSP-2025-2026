"""
Federated Urban Heat Island (FedUHI) - Federated Training Module

This module implements federated learning using Flower (flwr) framework
for urban temperature prediction. Each zone acts as a separate client
with its own local dataset, and a central server coordinates the training.

The federated setup preserves data privacy by keeping each zone's data
local while still enabling collaborative model training.
"""

import os
import sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

import time
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import flwr as fl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from typing import Dict, List, Tuple, Optional

class FederatedTemperatureClient(fl.client.NumPyClient):
    """Flower client for federated temperature prediction."""
    
    def __init__(self, zone_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray, cid: int, 
                 scaler_features: Optional[StandardScaler] = None,
                 scaler_target: Optional[StandardScaler] = None,
                 save_dir: Optional[str] = None):
        """
        Initialize federated client for a specific zone.
        
        Args:
            zone_name (str): Name of the urban zone
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            cid (int): Client ID
            scaler_features (StandardScaler, optional): Pre-fitted feature scaler
            scaler_target (StandardScaler, optional): Pre-fitted target scaler
            save_dir (str, optional): Directory to save scalers
        """
        self.zone_name = zone_name
        self.cid = cid
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.save_dir = save_dir
        
        if scaler_features is not None and scaler_target is not None:
            self.scaler_features = scaler_features
            self.scaler_target = scaler_target
            self.X_train_scaled = self.scaler_features.transform(X_train)
            self.y_train_scaled = self.scaler_target.transform(y_train.reshape(-1, 1)).flatten()
        else:
            self.scaler_features = StandardScaler()
            self.scaler_target = StandardScaler()
            self.X_train_scaled = self.scaler_features.fit_transform(X_train)
            self.y_train_scaled = self.scaler_target.fit_transform(y_train.reshape(-1, 1)).flatten()
            if save_dir:
                self._save_scalers()
        
        self.X_test_scaled = self.scaler_features.transform(X_test)
        self.y_test_scaled = self.scaler_target.transform(y_test.reshape(-1, 1)).flatten()
        
        self.model = self._build_model()
        
        self.local_metrics = []
        self.training_time = 0
        
    def _save_scalers(self):
        """Save scalers to disk for future use."""
        if not self.save_dir:
            return
        
        os.makedirs(self.save_dir, exist_ok=True)
        scaler_dir = os.path.join(self.save_dir, f'client_{self.cid}_scalers')
        os.makedirs(scaler_dir, exist_ok=True)
        
        feature_scaler_path = os.path.join(scaler_dir, 'feature_scaler.pkl')
        target_scaler_path = os.path.join(scaler_dir, 'target_scaler.pkl')
        
        with open(feature_scaler_path, 'wb') as f:
            pickle.dump(self.scaler_features, f)
        with open(target_scaler_path, 'wb') as f:
            pickle.dump(self.scaler_target, f)
    
    def _build_model(self) -> tf.keras.Model:
        """
        Build the neural network model using saved or default architecture.
        
        Returns:
            tf.keras.Model: Compiled model ready for training
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.X_train_scaled.shape[1],)),
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
    
    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """Return current model parameters."""
        return self.model.get_weights()
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None: 
        """Set model parameters from server."""
        self.model.set_weights(parameters)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """
        Train the model on local data.
        
        Args:
            parameters: Model parameters from server
            config: Training configuration
            
        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Extract training configuration
        epochs = int(config.get("epochs", 5))
        batch_size = int(config.get("batch_size", 32))
        
        # Train model
        start_time = time.time()
        history = self.model.fit(
            self.X_train_scaled, 
            self.y_train_scaled,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        self.training_time += time.time() - start_time
        
        # Calculate metrics
        train_loss = history.history['loss'][-1]
        train_mae = history.history['mae'][-1]
        
        # Store local metrics
        self.local_metrics.append({
            'loss': train_loss,
            'mae': train_mae,
            'samples': len(self.X_train_scaled)
        })
        
        print(f"Client {self.cid} ({self.zone_name}): Loss={train_loss:.4f}, MAE={train_mae:.4f}, Samples={len(self.X_train_scaled)}")
        
        return (
            self.get_parameters({}),
            len(self.X_train_scaled),
            {"loss": train_loss, "mae": train_mae}
        )
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict[str, float]]:
        """
        Evaluate the model on local test data.
        
        Args:
            parameters: Model parameters from server
            config: Evaluation configuration
            
        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        # Set parameters from server
        self.set_parameters(parameters)
        
        # Evaluate model
        loss, mae, mse = self.model.evaluate(
            self.X_test_scaled, 
            self.y_test_scaled, 
            verbose=0
        )
        
        # Calculate R score
        y_pred = self.model.predict(self.X_test_scaled)
        y_true = self.y_test_scaled
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            "loss": float(loss),
            "mae": float(mae),
            "mse": float(mse),
            "r2": float(r2)
        }
        
        return loss, len(self.X_test_scaled), metrics





def prepare_federated_data(data_dir: str) -> Tuple[Dict, Dict]:
    """
    Prepare data for federated learning by splitting into zone-specific datasets.
    
    Args:
        data_dir (str): Directory containing zone data files
        
    Returns:
        Tuple of (training_data, test_data) dictionaries
    """
    zone_files = {f"Zone_{zone.upper()}_{location}": f"zone_{zone}_{location}_data.csv" for zone, location in zip(['a', 'b', 'c', 'd'], ['rooftop', 'street', 'park', 'parking'])}
    
    training_data = {}
    test_data = {}
    
    for zone_name, filename in zone_files.items():
        filepath = os.path.join(data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Zone data file not found: {filepath}")
        
        # Load zone data
        data = pd.read_csv(filepath)
        
        # Prepare features (same as centralized)
        df = data.copy()
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # For federated learning, we don't use zone encoding since each client 
        # only has data from one zone
        feature_columns = ['humidity', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        X = df[feature_columns].values
        y = df['temperature'].values
        
        # Split data (80% train, 20% test)
        n_samples = len(X)
        n_train = int(n_samples * 0.8)
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        training_data[zone_name] = (X[train_indices], y[train_indices])
        test_data[zone_name] = (X[test_indices], y[test_indices])
        
        print(f"Prepared {zone_name}: {len(train_indices)} train, {len(test_indices)} test samples")
    
    return training_data, test_data


def run_federated_simulation(training_data: Dict, test_data: Dict, rounds: int = 10, models_dir='models') -> Dict:
    """
    Run federated learning simulation.
    
    Args:
        training_data (Dict): Training data for each zone
        test_data (Dict): Test data for each zone  
        rounds (int): Number of federated rounds
        models_dir (str): Directory to save models
        
    Returns:
        Dict containing federated training results
    """
    # Store global data for client creation
    global _global_training_data, _global_test_data
    _global_training_data = training_data
    _global_test_data = test_data
    
    def create_client_fn(cid: str) -> FederatedTemperatureClient:
        """Create a federated client."""
        zone_names = list(training_data.keys())
        zone_idx = int(cid) % len(zone_names)
        zone_name = zone_names[zone_idx]
        
        X_train, y_train = _global_training_data[zone_name]
        X_test, y_test = _global_test_data[zone_name]
        
        return FederatedTemperatureClient(
            zone_name=zone_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            cid=int(cid)
        )
    
    # Custom strategy to save model weights
    class SaveModelStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        ) -> Tuple[fl.common.Parameters, Dict[str, fl.common.Scalar]]:
            """Aggregate model weights and save them."""
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
            if aggregated_parameters is not None:
                print(f"Saving round {server_round} aggregated_parameters...")
                # parameters_to_ndarrays returns a list of ndarrays with different shapes.
                # np.save expects a single homogeneous ndarray; saving the list triggers a ValueError.
                # Use np.savez to persist each array as a separate entry.
                weights_list = fl.common.parameters_to_ndarrays(aggregated_parameters)
                np.savez(
                    f"round-{server_round}-weights.npz",
                    **{f"arr_{i}": w for i, w in enumerate(weights_list)}
                )
            return aggregated_parameters, aggregated_metrics

    # Configure federated learning strategy
    strategy = SaveModelStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=4,
        evaluate_metrics_aggregation_fn=lambda metrics: {
            "loss": np.average([m[1]["loss"] for m in metrics], weights=[m[0] for m in metrics]),
            "mae": np.average([m[1]["mae"] for m in metrics], weights=[m[0] for m in metrics]),
            "r2": np.average([m[1]["r2"] for m in metrics], weights=[m[0] for m in metrics]),
        }
    )
    
    # Run federated simulation
    print(f"Starting federated simulation with {len(training_data)} clients...")
    start_time = time.time()
    
    history = fl.simulation.start_simulation(
        client_fn=create_client_fn,
        num_clients=len(training_data),
        config=fl.server.ServerConfig(num_rounds=rounds),
        strategy=strategy,
        ray_init_args={
            "runtime_env": {
                "working_dir": REPO_ROOT,
            }
        }
    )

    training_time = time.time() - start_time

    # Extract final metrics from history
    final_metrics = None
    if history is not None:
        # Try to extract from metrics_distributed (Flower's standard format)
        if hasattr(history, 'metrics_distributed') and history.metrics_distributed:
            metrics = history.metrics_distributed
            final_metrics = {
                'loss': metrics['loss'][-1][1] if metrics.get('loss') and isinstance(metrics['loss'], list) else 0,
                'mae': metrics['mae'][-1][1] if metrics.get('mae') and isinstance(metrics['mae'], list) else 0,
                'r2': metrics['r2'][-1][1] if metrics.get('r2') and isinstance(metrics['r2'], list) else 0
            }
        # Fallback to metrics_centralized if available
        elif hasattr(history, 'metrics_centralized') and history.metrics_centralized:
            metrics_centralized = history.metrics_centralized if isinstance(history.metrics_centralized, dict) else {}
            distributed_metrics = metrics_centralized.get('distributed', [])
            if isinstance(distributed_metrics, list) and distributed_metrics:
                final_metrics = distributed_metrics[-1][1]

    results = {
        'training_time': training_time,
        'rounds': rounds,
        'num_clients': len(training_data),
        'final_metrics': final_metrics,
        'history': history
    }
    
    print(f"Federated simulation completed in {training_time:.2f} seconds")
    
    # Save the final federated model
    last_round_weights_file = f"round-{rounds}-weights.npz"
    if os.path.exists(last_round_weights_file):
        with np.load(last_round_weights_file) as loaded:
            # Reconstruct list of arrays in index order
            final_weights = [loaded[key] for key in sorted(loaded.files, key=lambda k: int(k.split('_')[1]))]
        
        from code.model_manager import ModelManager
        model_manager = ModelManager(models_dir=models_dir)
        model_manager.save_federated_model(
            model_weights=final_weights,
            metrics=final_metrics,
            training_time=training_time,
            config={'rounds': rounds}
        )
        
        # Clean up round weight files
        for i in range(1, rounds + 1):
            round_weights_file = f"round-{i}-weights.npz"
            if os.path.exists(round_weights_file):
                os.remove(round_weights_file)

    return results

def estimate_bandwidth_usage(training_data: Dict, rounds: int) -> Dict:
    """
    Estimate the total bandwidth usage for federated training.
    
    Args:
        training_data (Dict): Training data for each zone
        rounds (int): Number of federated rounds
        
    Returns:
        Dict containing bandwidth estimation
    """
    if not training_data:
        return {
            'model_size_mb': 0,
            'bytes_per_round_mb': 0,
            'total_bytes_mb': 0,
        }

    # Create a dummy model to get the size of the weights
    zone_name = list(training_data.keys())[0]
    X_train, _ = training_data[zone_name]
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='linear')
    ])
    
    model_weights = model.get_weights()
    
    # Calculate the size of the model in bytes
    model_size_bytes = sum(w.nbytes for w in model_weights)
    
    # In each round, server sends model to each client, and each client sends updated model back
    num_clients = len(training_data)
    bytes_per_round = 2 * num_clients * model_size_bytes
    total_bytes = bytes_per_round * rounds
    
    return {
        'model_size_mb': model_size_bytes / (1024 * 1024),
        'bytes_per_round_mb': bytes_per_round / (1024 * 1024),
        'total_bytes_mb': total_bytes / (1024 * 1024),
    }
    
    

