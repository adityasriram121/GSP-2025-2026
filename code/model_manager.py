"""
FedUHI Model Manager - Comprehensive model saving and loading system

This module handles saving, loading, and managing all trained models
with their metadata, performance metrics, and training configurations.
"""

import os
import pickle
import json
import shutil
from datetime import datetime
from typing import Dict, Any, Optional
import tensorflow as tf
import numpy as np


class ModelManager:
    """Comprehensive model management system for FedUHI project."""
    
    def __init__(self, models_dir: str):
        """
        Initialize model manager.
        
        Args:
            models_dir (str): Directory to store all models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Create subdirectories
        self.centralized_dir = os.path.join(models_dir, 'centralized')
        self.federated_dir = os.path.join(models_dir, 'federated')
        self.reliable_dir = os.path.join(models_dir, 'reliable')
        
        for dir_path in [self.centralized_dir, self.federated_dir, self.reliable_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def save_centralized_model(self, model, metrics, training_time, config=None):
        """
        Save centralized model with complete metadata.
        
        Args:
            model: Trained TensorFlow model
            metrics (dict): Performance metrics
            training_time (float): Training time in seconds
            config (dict): Training configuration
            
        Returns:
            tuple: (model_path, metadata) if successful, (None, None) if failed
            
        Raises:
            ValueError: If model is not a valid TensorFlow model
            TypeError: If metrics is not a dictionary
        """
        # Validate inputs
        if not isinstance(model, tf.keras.Model):
            raise ValueError("Model must be a valid TensorFlow model")
        if not isinstance(metrics, dict):
            raise TypeError("Metrics must be a dictionary")
        if not isinstance(training_time, (int, float)):
            raise TypeError("Training time must be a number")
            
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"centralized_model_{timestamp}"
            model_path = os.path.join(self.centralized_dir, model_name)
            
            # Validate model can be saved
            try:
                model.summary()
            except:
                raise ValueError("Invalid model: cannot get model summary")
            
            # Save TensorFlow model with error handling
            try:
                model.save(model_path + '.keras')
            except Exception as e:
                raise RuntimeError(f"Failed to save model: {str(e)}")
            
            # Save metadata
            metadata = {
                'model_type': 'centralized',
                'timestamp': timestamp,
                'model_path': model_path,
                'metrics': metrics,
                'training_time': training_time,
                'config': config or {},
                'architecture': {
                    'layers': len(model.layers),
                    'total_params': model.count_params(),
                    'input_shape': model.input_shape,
                    'output_shape': model.output_shape
                }
            }
            
            metadata_path = os.path.join(self.centralized_dir, f"{model_name}_metadata.json")
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            except Exception as e:
                # Clean up model file if metadata save fails
                if os.path.exists(model_path + '.keras'):
                    os.remove(model_path + '.keras')
                raise RuntimeError(f"Failed to save metadata: {str(e)}")
            
            # Save as pickle for easy loading
            pickle_path = os.path.join(self.centralized_dir, f"{model_name}.pkl")
            try:
                with open(pickle_path, 'wb') as f:
                    pickle.dump({
                        'model': model,
                        'metadata': metadata
                    }, f)
            except Exception as e:
                # Clean up previous files if pickle save fails
                if os.path.exists(model_path + '.keras'):
                    os.remove(model_path + '.keras')
                if os.path.exists(metadata_path):
                    os.remove(metadata_path)
                raise RuntimeError(f"Failed to save pickle: {str(e)}")
            
            print(f'Centralized model saved to: {model_path}')
            print(f" Metadata saved to: {metadata_path}")
            print(f" Pickle saved to: {pickle_path}")
            
            return model_path, metadata
            
        except Exception as e:
            print(f"[ERROR] Failed to save centralized model: {str(e)}")
            return None, None
    
    def save_federated_model(self, model_weights, metrics, training_time, config=None, bandwidth_info=None):
        """
        Save federated model with complete metadata.
        
        Args:
            model_weights: Final aggregated model weights
            metrics (dict): Performance metrics
            training_time (float): Training time in seconds
            config (dict): Training configuration
            bandwidth_info (dict): Bandwidth usage information
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"federated_model_{timestamp}"
        model_path = os.path.join(self.federated_dir, model_name)
        
        # Save model weights
        weights_path = os.path.join(model_path, "model_weights.pkl")
        os.makedirs(model_path, exist_ok=True)
        
        with open(weights_path, 'wb') as f:
            pickle.dump(model_weights, f)
        
        # Save metadata
        metadata = {
            'model_type': 'federated',
            'timestamp': timestamp,
            'model_path': model_path,
            'metrics': metrics,
            'training_time': training_time,
            'config': config or {},
            'bandwidth_info': bandwidth_info or {},
            'architecture': {
                'weights_shape': [w.shape for w in model_weights] if model_weights else None,
                'total_parameters': sum(w.size for w in model_weights) if model_weights else 0
            }
        }
        
        metadata_path = os.path.join(model_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        print(f" Federated model saved to: {model_path}")
        print(f" Metadata saved to: {metadata_path}")
        
        return model_path, metadata
    
    def save_reliable_models(self, centralized_results, federated_results):
        """
        Save all models from reliable training runs.
        
        Args:
            centralized_results (list): List of centralized training results
            federated_results (list): List of federated training results
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        reliable_path = os.path.join(self.reliable_dir, f"reliable_training_{timestamp}")
        os.makedirs(reliable_path, exist_ok=True)
        
        # Save centralized models
        centralized_path = os.path.join(reliable_path, "centralized")
        os.makedirs(centralized_path, exist_ok=True)
        
        for i, result in enumerate(centralized_results):
            model_name = f"centralized_run_{i+1}"
            model_path = os.path.join(centralized_path, model_name)
            
            if 'model' in result:
                result['model'].save(model_path)
            
            # Save run metadata
            run_metadata = {
                'run_number': i + 1,
                'seed': result.get('seed', 'unknown'),
                'metrics': result.get('metrics', {}),
                'training_time': result.get('training_time', 0)
            }
            
            with open(os.path.join(centralized_path, f"{model_name}_metadata.json"), 'w') as f:
                json.dump(run_metadata, f, indent=2, default=str)
        
        # Save federated models
        federated_path = os.path.join(reliable_path, "federated")
        os.makedirs(federated_path, exist_ok=True)
        
        for i, result in enumerate(federated_results):
            model_name = f"federated_run_{i+1}"
            model_path = os.path.join(federated_path, model_name)
            os.makedirs(model_path, exist_ok=True)
            
            # Save run metadata
            run_metadata = {
                'run_number': i + 1,
                'seed': result.get('seed', 'unknown'),
                'metrics': result.get('final_metrics', {}),
                'training_time': result.get('training_time', 0),
                'bandwidth_estimate': result.get('bandwidth_estimate', {})
            }
            
            with open(os.path.join(federated_path, f"{model_name}_metadata.json"), 'w') as f:
                json.dump(run_metadata, f, indent=2, default=str)
        
        # Save summary statistics
        summary_path = os.path.join(reliable_path, "reliable_training_summary.json")
        summary_data = {
            'timestamp': timestamp,
            'num_centralized_runs': len(centralized_results),
            'num_federated_runs': len(federated_results),
            'centralized_summary': self._calculate_summary_stats(centralized_results),
            'federated_summary': self._calculate_summary_stats(federated_results)
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        print(f" Reliable training models saved to: {reliable_path}")
        return reliable_path
    
    def _calculate_summary_stats(self, results):
        """Calculate summary statistics for multiple runs."""
        if not results:
            return {}
        
        # Extract metrics
        rmse_values = [r.get('metrics', {}).get('rmse', 0) for r in results if r.get('metrics', {}).get('rmse', 0) > 0]
        mae_values = [r.get('metrics', {}).get('mae', 0) for r in results if r.get('metrics', {}).get('mae', 0) > 0]
        r2_values = [r.get('metrics', {}).get('r2', 0) for r in results if r.get('metrics', {}).get('r2', 0) != 0]
        time_values = [r.get('training_time', 0) for r in results]
        
        return {
            'rmse_mean': np.mean(rmse_values) if rmse_values else 0,
            'rmse_std': np.std(rmse_values) if rmse_values else 0,
            'mae_mean': np.mean(mae_values) if mae_values else 0,
            'mae_std': np.std(mae_values) if mae_values else 0,
            'r2_mean': np.mean(r2_values) if r2_values else 0,
            'r2_std': np.std(r2_values) if r2_values else 0,
            'time_mean': np.mean(time_values),
            'time_std': np.std(time_values)
        }
    
    def list_all_models(self):
        """List all saved models with their metadata."""
        print("\n SAVED MODELS INVENTORY:")
        print("=" * 60)
        
        # List centralized models
        if os.path.exists(self.centralized_dir):
            centralized_metadata = [f for f in os.listdir(self.centralized_dir) if f.endswith('_metadata.json')]
            if centralized_metadata:
                print(f"\n Centralized Models ({len(centralized_metadata)}):")
                for metadata_file_name in sorted(centralized_metadata):
                    model_name = metadata_file_name.replace('_metadata.json', '')
                    metadata_file = os.path.join(self.centralized_dir, metadata_file_name)
                    if os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            print(f"    {model_name}: RMSE={metadata.get('metrics', {}).get('rmse', 'N/A'):.4f}C")
                        except:
                            print(f"    {model_name}: (metadata unavailable)")
        
        # List federated models
        if os.path.exists(self.federated_dir):
            federated_models = [d for d in os.listdir(self.federated_dir) if os.path.isdir(os.path.join(self.federated_dir, d))]
            if federated_models:
                print(f"\n Federated Models ({len(federated_models)}):")
                for model in sorted(federated_models):
                    model_path = os.path.join(self.federated_dir, model)
                    metadata_file = os.path.join(model_path, "metadata.json")
                    if os.path.exists(metadata_file):
                        try:
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                            print(f"    {model}: RMSE={metadata.get('metrics', {}).get('loss', 'N/A'):.4f}C")
                        except:
                            print(f"    {model}: (metadata unavailable)")
        
        # List reliable training results
        if os.path.exists(self.reliable_dir):
            reliable_runs = [d for d in os.listdir(self.reliable_dir) if os.path.isdir(os.path.join(self.reliable_dir, d))]
            if reliable_runs:
                print(f"\n Reliable Training Runs ({len(reliable_runs)}):")
                for run in sorted(reliable_runs):
                    summary_file = os.path.join(self.reliable_dir, run, "reliable_training_summary.json")
                    if os.path.exists(summary_file):
                        try:
                            with open(summary_file, 'r') as f:
                                summary = json.load(f)
                            print(f"    {run}: {summary.get('num_centralized_runs', 0)} centralized + {summary.get('num_federated_runs', 0)} federated runs")
                        except:
                            print(f"    {run}: (summary unavailable)")
    
    def load_best_model(self, model_type='centralized'):
        """
        Load the best performing model of specified type.
        
        Args:
            model_type (str): 'centralized', 'federated', or 'reliable'
            
        Returns:
            tuple: (model, metadata) or (None, None) if not found
        """
        if model_type == 'centralized':
            return self._load_best_centralized_model()
        elif model_type == 'federated':
            return self._load_best_federated_model()
        elif model_type == 'reliable':
            return self._load_best_reliable_model()
        else:
            print(f"Unknown model type: {model_type}")
            return None, None
    
    def _load_best_centralized_model(self):
        """Load the best centralized model based on RMSE."""
        if not os.path.exists(self.centralized_dir):
            return None, None
        
        best_model = None
        best_rmse = float('inf')
        best_metadata = None
        
        for item in os.listdir(self.centralized_dir):
            item_path = os.path.join(self.centralized_dir, item)
            # Check if it's a metadata file
            if item.endswith('_metadata.json'):
                model_name = item.replace('_metadata.json', '')
                model_path = os.path.join(self.centralized_dir, model_name + '.keras')
                
                try:
                    with open(item_path, 'r') as f:
                        metadata = json.load(f)
                    
                    rmse = metadata.get('metrics', {}).get('rmse', float('inf'))
                    if rmse < best_rmse and os.path.exists(model_path):
                        best_rmse = rmse
                        best_model = model_path
                        best_metadata = metadata
                except:
                    continue
        
        if best_model:
            try:
                model = tf.keras.models.load_model(best_model)
                return model, best_metadata
            except:
                return None, None
        
        return None, None
    
    def _load_best_federated_model(self):
        """Load the best federated model based on RMSE."""
        if not os.path.exists(self.federated_dir):
            return None, None
        
        best_model = None
        best_rmse = float('inf')
        best_metadata = None
        
        for model_dir in os.listdir(self.federated_dir):
            model_path = os.path.join(self.federated_dir, model_dir)
            metadata_file = os.path.join(model_path, "metadata.json")
            
            if os.path.exists(metadata_file):
                try:
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    rmse = metadata.get('metrics', {}).get('loss', float('inf'))
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model_path
                        best_metadata = metadata
                except:
                    continue
        
        if best_model:
            try:
                weights_file = os.path.join(best_model, "model_weights.pkl")
                with open(weights_file, 'rb') as f:
                    weights = pickle.load(f)
                return weights, best_metadata
            except:
                return None, None
        
        return None, None
    
    def _load_best_reliable_model(self):
        """Load the best reliable training results."""
        if not os.path.exists(self.reliable_dir):
            return None, None
        
        latest_run = None
        latest_time = None
        
        for run_dir in os.listdir(self.reliable_dir):
            run_path = os.path.join(self.reliable_dir, run_dir)
            summary_file = os.path.join(run_path, "reliable_training_summary.json")
            
            if os.path.exists(summary_file):
                try:
                    with open(summary_file, 'r') as f:
                        summary = json.load(f)
                    
                    timestamp = summary.get('timestamp', '')
                    if timestamp > (latest_time or ''):
                        latest_time = timestamp
                        latest_run = run_path
                except:
                    continue
        
        return latest_run, latest_time
    
    def export_model_for_deployment(self, model_type='centralized', export_dir='deployment_models'):
        """
        Export model in deployment-ready format.
        
        Args:
            model_type (str): Type of model to export
            export_dir (str): Directory to export deployment models
        """
        os.makedirs(export_dir, exist_ok=True)
        
        model, metadata = self.load_best_model(model_type)
        
        if model and metadata:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_path = os.path.join(export_dir, f"{model_type}_deployment_{timestamp}")
            
            if model_type == 'centralized':
                # Export TensorFlow model
                model.save(export_path)
            elif model_type == 'federated':
                # Export federated model weights
                os.makedirs(export_path, exist_ok=True)
                weights_file = os.path.join(export_path, "model_weights.pkl")
                with open(weights_file, 'wb') as f:
                    pickle.dump(model, f)
            
            # Export metadata
            metadata_file = os.path.join(export_path, "deployment_metadata.json")
            with open(metadata_file, 'w') as f:
                json.dump({
                    'model_type': model_type,
                    'export_timestamp': timestamp,
                    'performance_metrics': metadata.get('metrics', {}),
                    'deployment_ready': True
                }, f, indent=2, default=str)
            
            print(f" Model exported for deployment to: {export_path}")
            return export_path
        
        print(f" No {model_type} model found to export")
        return None







