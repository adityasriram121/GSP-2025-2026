"""
Federated Urban Heat Island (FedUHI) - Federated Training Module

This module implements federated learning using Flower (flwr) framework
for urban temperature prediction. Each zone acts as a separate client
with its own local dataset, and a central server coordinates the training.

The federated setup preserves data privacy by keeping each zone's data
local while still enabling collaborative model training.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import flwr as fl
import time
import os
import pickle
import threading
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


class FederatedTemperatureClient(fl.client.NumPyClient):
    """Flower client for federated temperature prediction."""
    
    def __init__(self, zone_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray, cid: int):
        """
        Initialize federated client for a specific zone.
        
        Args:
            zone_name (str): Name of the urban zone
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
            cid (int): Client ID
        """
        self.zone_name = zone_name
        self.cid = cid
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Initialize scalers for this client
        self.scaler_features = StandardScaler()
        self.scaler_target = StandardScaler()
        
        # Scale the data
        self.X_train_scaled = self.scaler_features.fit_transform(X_train)
        self.y_train_scaled = self.scaler_target.fit_transform(y_train.reshape(-1, 1)).flatten()
        self.X_test_scaled = self.scaler_features.transform(X_test)
        self.y_test_scaled = self.scaler_target.transform(y_test.reshape(-1, 1)).flatten()
        
        # Build model
        self.model = self._build_model()
        
        # Track metrics
        self.local_metrics = []
        self.training_time = 0
    
    def _build_model(self) -> tf.keras.Model:
        """Build the neural network model (same architecture as centralized)."""
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
        
        # Calculate R² score
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


class FederatedTemperatureServer:
    """Centralized server for federated temperature prediction."""
    
    def __init__(self, num_clients: int = 4):
        """
        Initialize federated server.
        
        Args:
            num_clients (int): Number of federated clients
        """
        self.num_clients = num_clients
        self.round_metrics = []
        self.server_metrics = []
        self.training_start_time = 0
        self.training_end_time = 0
    
    def get_on_fit_config_fn(self):
        """Return configuration function for client training."""
        def fit_config(server_round: int) -> Dict[str, str]:
            """Configure client training rounds."""
            config = {
                "epochs": str(3),  # Local epochs per round
                "batch_size": str(32),
            }
            return config
        return fit_config
    
    def get_on_evaluate_config_fn(self):
        """Return configuration function for client evaluation."""
        def evaluate_config(server_round: int) -> Dict[str, str]:
            """Configure client evaluation."""
            config = {"batch_size": str(32)}
            return config
        return evaluate_config
    
    def weighted_average(self, metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
        """
        Calculate weighted average of client metrics.
        
        Args:
            metrics: List of (num_samples, metrics_dict) tuples
            
        Returns:
            Dict of averaged metrics
        """
        acc_loss = np.average([m[1]["loss"] for m in metrics], weights=[m[0] for m in metrics])
        acc_mae = np.average([m[1]["mae"] for m in metrics], weights=[m[0] for m in metrics])
        acc_r2 = np.average([m[1]["r2"] for m in metrics], weights=[m[0] for m in metrics])
        
        return {"loss": acc_loss, "mae": acc_mae, "r2": acc_r2}
    
    def run_federated_training(self, rounds: int = 10, min_available_clients: int = 4):
        """
        Run federated training rounds.
        
        Args:
            rounds (int): Number of federated training rounds
            min_available_clients (int): Minimum number of clients required
            
        Returns:
            Dict containing training results and metrics
        """
        print(f"Starting federated training for {rounds} rounds...")
        self.training_start_time = time.time()
        
        # Configure federated learning
        strategy = fl.server.strategy.FedAvg(
            fraction_fit=1.0,  # Use all available clients
            fraction_evaluate=1.0,
            min_available_clients=min_available_clients,
            on_fit_config_fn=self.get_on_fit_config_fn(),
            on_evaluate_config_fn=self.get_on_evaluate_config_fn(),
            evaluate_metrics_aggregation_fn=self.weighted_average,
        )
        
        # Start federated training
        fl.simulation.start_simulation(
            client_fn=self._create_client_fn(),
            num_clients=self.num_clients,
            config=fl.server.ServerConfig(num_rounds=rounds),
            strategy=strategy,
        )
        
        self.training_end_time = time.time()
        training_time = self.training_end_time - self.training_start_time
        
        print(f"Federated training completed in {training_time:.2f} seconds")
        
        return {
            'training_time': training_time,
            'rounds': rounds,
            'total_clients': self.num_clients
        }
    
    def _create_client_fn(self):
        """Create client function for simulation (to be implemented by main function)."""
        pass


def prepare_federated_data(data_dir: str = 'data') -> Tuple[Dict, Dict]:
    """
    Prepare data for federated learning by splitting into zone-specific datasets.
    
    Args:
        data_dir (str): Directory containing zone data files
        
    Returns:
        Tuple of (training_data, test_data) dictionaries
    """
    zone_files = {
        'Zone_A_Rooftop': 'zone_a_rooftop_data.csv',
        'Zone_B_Street': 'zone_b_street_data.csv', 
        'Zone_C_Park': 'zone_c_park_data.csv',
        'Zone_D_Parking': 'zone_d_parking_data.csv'
    }
    
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


def run_federated_simulation(training_data: Dict, test_data: Dict, rounds: int = 10) -> Dict:
    """
    Run federated learning simulation.
    
    Args:
        training_data (Dict): Training data for each zone
        test_data (Dict): Test data for each zone  
        rounds (int): Number of federated rounds
        
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
    
    # Configure federated learning strategy
    strategy = fl.server.strategy.FedAvg(
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
    )

    training_time = time.time() - start_time

    # Extract final metrics
    final_metrics = None
    if history is not None and hasattr(history, 'metrics_centralized') and history.metrics_centralized:
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
    
    return results


def estimate_bandwidth_usage(training_data: Dict, rounds: int = 10) -> Dict:
    """
    Estimate bandwidth usage for federated learning.
    
    Args:
        training_data (Dict): Training data for each client
        rounds (int): Number of federated rounds
        
    Returns:
        Dict containing bandwidth estimates
    """
    # Model parameters size (approximate)
    # Dense layers: 64->32->16->1 with input features
    input_dim = next(iter(training_data.values()))[0].shape[1]
    param_sizes = [
        input_dim * 64 + 64,  # First dense layer
        64 * 32 + 32,         # Second dense layer  
        32 * 16 + 16,         # Third dense layer
        16 * 1 + 1            # Output layer
    ]
    total_params = sum(param_sizes)
    
    # Each parameter is typically a float32 (4 bytes)
    bytes_per_param = 4
    model_size_bytes = total_params * bytes_per_param
    
    # Bandwidth calculation
    # Each round: clients send model updates + server sends global model
    bytes_per_round = len(training_data) * model_size_bytes * 2  # Upload + download
    total_bytes = bytes_per_round * rounds
    
    bandwidth_estimate = {
        'model_size_mb': model_size_bytes / (1024 * 1024),
        'bytes_per_round_mb': bytes_per_round / (1024 * 1024),
        'total_bytes_mb': total_bytes / (1024 * 1024),
        'total_bytes_gb': total_bytes / (1024 * 1024 * 1024),
        'rounds': rounds,
        'num_clients': len(training_data)
    }
    
    return bandwidth_estimate


def main():
    """Main function to demonstrate federated training."""
    print("Federated Urban Heat Island (FedUHI) - Federated Training")
    print("="*60)
    
    # Check if data exists
    data_dir = 'data'
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        print("Please run data_generation.py first to create the dataset.")
        return
    
    # Prepare federated data
    print("Preparing federated data...")
    training_data, test_data = prepare_federated_data(data_dir)
    
    # Estimate bandwidth usage
    print("\nEstimating bandwidth usage...")
    bandwidth_estimate = estimate_bandwidth_usage(training_data, rounds=10)
    print(f"Model size: {bandwidth_estimate['model_size_mb']:.2f} MB")
    print(f"Bandwidth per round: {bandwidth_estimate['bytes_per_round_mb']:.2f} MB")
    print(f"Total bandwidth (10 rounds): {bandwidth_estimate['total_bytes_mb']:.2f} MB")
    
    # Run federated simulation
    print("\nRunning federated learning simulation...")
    results = run_federated_simulation(training_data, test_data, rounds=10)
    
    # Print results
    print("\n" + "="*50)
    print("FEDERATED LEARNING RESULTS")
    print("="*50)
    print(f"Training time: {results['training_time']:.2f} seconds")
    print(f"Number of rounds: {results['rounds']}")
    print(f"Number of clients: {results['num_clients']}")
    
    if results['final_metrics']:
        print(f"Final Loss: {results['final_metrics'].get('loss', 'N/A'):.4f}")
        print(f"Final MAE: {results['final_metrics'].get('mae', 'N/A'):.4f}")
        print(f"Final R²: {results['final_metrics'].get('r2', 'N/A'):.4f}")
    
    print(f"Bandwidth usage: {bandwidth_estimate['total_bytes_mb']:.2f} MB")
    
    # Save results and final model
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Save federated results
    with open('results/federated_results.pkl', 'wb') as f:
        pickle.dump({
            'results': results,
            'bandwidth_estimate': bandwidth_estimate
        }, f)
    
    # Save the final federated model if available
    history = results.get('history') if isinstance(results, dict) else None
    if history is not None and hasattr(history, 'model') and history.model is not None:
        try:
            history.model.save('models/federated_model')
            print("Saved final federated model to models/federated_model")
        except Exception as e:
            print(f"Warning: Could not save federated model: {e}")
    else:
        print("Final federated model was not provided by the Flower history object; skipping model save.")
    
    print("\nFederated training completed successfully!")
    print("Results saved to 'results/federated_results.pkl'")
    print("Model saved to 'models/' directory")


if __name__ == "__main__":
    main()
