"""
FedUHI Reliable Training Module

This module runs multiple training iterations for both centralized and federated
models to provide statistically reliable results with confidence intervals.
"""

import numpy as np
import pandas as pd
import time
import pickle
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

from centralized_training import train_centralized_model
from federated_training import prepare_federated_data, run_federated_simulation, estimate_bandwidth_usage


class ReliableTrainer:
    """Train models multiple times for reliable statistical results."""
    
    def __init__(self, num_iterations=3, seed_base=42):
        """
        Initialize reliable trainer.
        
        Args:
            num_iterations (int): Number of training runs per model
            seed_base (int): Base seed for reproducibility
        """
        self.num_iterations = num_iterations
        self.seed_base = seed_base
        self.results = {
            'centralized': [],
            'federated': []
        }
    
    def run_multiple_centralized_training(self, data_path='data/combined_zone_data.csv'):
        """
        Run centralized training multiple times with different seeds.
        
        Args:
            data_path (str): Path to combined dataset
            
        Returns:
            dict: Statistical summary of results
        """
        print(f"\n🧠 Running {self.num_iterations} centralized training iterations...")
        print("-" * 60)
        
        centralized_results = []
        
        for i in range(self.num_iterations):
            seed = self.seed_base + i
            print(f"\n🔄 Centralized Training Run {i+1}/{self.num_iterations} (Seed: {seed})")
            
            try:
                # Set random seed for this iteration
                np.random.seed(seed)
                
                # Train model
                model, metrics, training_time = train_centralized_model(data_path, test_split=0.2)
                
                # Store results
                result = {
                    'iteration': i + 1,
                    'seed': seed,
                    'metrics': metrics,
                    'training_time': training_time,
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'r2': metrics['r2']
                }
                
                centralized_results.append(result)
                
                print(f"   ✅ RMSE: {metrics['rmse']:.4f}°C, MAE: {metrics['mae']:.4f}°C, R²: {metrics['r2']:.4f}")
                print(f"   ⏱️ Training Time: {training_time:.2f}s")
                
            except Exception as e:
                print(f"   ❌ Error in iteration {i+1}: {e}")
                continue
        
        self.results['centralized'] = centralized_results
        return self._calculate_statistics(centralized_results, 'centralized')
    
    def run_multiple_federated_training(self, data_dir='data'):
        """
        Run federated training multiple times with different seeds.
        
        Args:
            data_dir (str): Directory containing zone data files
            
        Returns:
            dict: Statistical summary of results
        """
        print(f"\n🤝 Running {self.num_iterations} federated training iterations...")
        print("-" * 60)
        
        federated_results = []
        
        for i in range(self.num_iterations):
            seed = self.seed_base + i
            print(f"\n🔄 Federated Training Run {i+1}/{self.num_iterations} (Seed: {seed})")
            
            try:
                # Set random seed for this iteration
                np.random.seed(seed)
                
                # Prepare federated data
                training_data, test_data = prepare_federated_data(data_dir)
                
                # Estimate bandwidth
                bandwidth_estimate = estimate_bandwidth_usage(training_data, rounds=10)
                
                # Run federated simulation
                federated_result = run_federated_simulation(training_data, test_data, rounds=10)
                
                # Store results
                result = {
                    'iteration': i + 1,
                    'seed': seed,
                    'training_time': federated_result['training_time'],
                    'rounds': federated_result['rounds'],
                    'bandwidth_estimate': bandwidth_estimate,
                    'final_metrics': federated_result['final_metrics']
                }
                
                # Extract metrics if available
                if federated_result['final_metrics']:
                    result.update({
                        'rmse': federated_result['final_metrics'].get('loss', 0),
                        'mae': federated_result['final_metrics'].get('mae', 0),
                        'r2': federated_result['final_metrics'].get('r2', 0)
                    })
                else:
                    result.update({'rmse': 0, 'mae': 0, 'r2': 0})
                
                federated_results.append(result)
                
                print(f"   ✅ RMSE: {result['rmse']:.4f}°C, MAE: {result['mae']:.4f}°C, R²: {result['r2']:.4f}")
                print(f"   ⏱️ Training Time: {result['training_time']:.2f}s")
                print(f"   📊 Bandwidth: {bandwidth_estimate['total_bytes_mb']:.2f} MB")
                
            except Exception as e:
                print(f"   ❌ Error in iteration {i+1}: {e}")
                continue
        
        self.results['federated'] = federated_results
        return self._calculate_statistics(federated_results, 'federated')
    
    def _calculate_statistics(self, results: List[Dict], model_type: str) -> Dict:
        """
        Calculate statistical summary for multiple training runs.
        
        Args:
            results (List[Dict]): List of training results
            model_type (str): Type of model ('centralized' or 'federated')
            
        Returns:
            dict: Statistical summary
        """
        if not results:
            return {}
        
        # Extract metrics
        rmse_values = [r['rmse'] for r in results if r['rmse'] > 0]
        mae_values = [r['mae'] for r in results if r['mae'] > 0]
        r2_values = [r['r2'] for r in results if r['r2'] != 0]
        training_times = [r['training_time'] for r in results]
        
        # Calculate statistics
        stats = {
            'model_type': model_type,
            'num_successful_runs': len(results),
            'total_runs': self.num_iterations,
            'success_rate': len(results) / self.num_iterations * 100,
            'rmse': {
                'mean': np.mean(rmse_values) if rmse_values else 0,
                'std': np.std(rmse_values) if rmse_values else 0,
                'min': np.min(rmse_values) if rmse_values else 0,
                'max': np.max(rmse_values) if rmse_values else 0,
                'values': rmse_values
            },
            'mae': {
                'mean': np.mean(mae_values) if mae_values else 0,
                'std': np.std(mae_values) if mae_values else 0,
                'min': np.min(mae_values) if mae_values else 0,
                'max': np.max(mae_values) if mae_values else 0,
                'values': mae_values
            },
            'r2': {
                'mean': np.mean(r2_values) if r2_values else 0,
                'std': np.std(r2_values) if r2_values else 0,
                'min': np.min(r2_values) if r2_values else 0,
                'max': np.max(r2_values) if r2_values else 0,
                'values': r2_values
            },
            'training_time': {
                'mean': np.mean(training_times),
                'std': np.std(training_times),
                'min': np.min(training_times),
                'max': np.max(training_times),
                'values': training_times
            }
        }
        
        # Add federated-specific metrics
        if model_type == 'federated' and results:
            bandwidth_values = [r['bandwidth_estimate']['total_bytes_mb'] for r in results]
            stats['bandwidth'] = {
                'mean': np.mean(bandwidth_values),
                'std': np.std(bandwidth_values),
                'min': np.min(bandwidth_values),
                'max': np.max(bandwidth_values),
                'values': bandwidth_values
            }
        
        return stats
    
    def create_reliability_plots(self, save_path='results/reliability_analysis.png'):
        """
        Create plots showing reliability analysis across multiple runs.
        
        Args:
            save_path (str): Path to save the plot
        """
        if not self.results['centralized'] or not self.results['federated']:
            print("⚠️ Cannot create reliability plots without both model results")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('FedUHI - Reliability Analysis (Multiple Training Runs)', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        central_data = self.results['centralized']
        fed_data = self.results['federated']
        
        # 1. RMSE Distribution
        central_rmse = [r['rmse'] for r in central_data if r['rmse'] > 0]
        fed_rmse = [r['rmse'] for r in fed_data if r['rmse'] > 0]
        
        ax1.hist(central_rmse, alpha=0.7, label='Centralized', color='skyblue', bins=10)
        ax1.hist(fed_rmse, alpha=0.7, label='Federated', color='lightcoral', bins=10)
        ax1.set_title('RMSE Distribution Across Runs', fontweight='bold')
        ax1.set_xlabel('RMSE (°C)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Training Time Distribution
        central_time = [r['training_time'] for r in central_data]
        fed_time = [r['training_time'] for r in fed_data]
        
        ax2.hist(central_time, alpha=0.7, label='Centralized', color='lightgreen', bins=10)
        ax2.hist(fed_time, alpha=0.7, label='Federated', color='orange', bins=10)
        ax2.set_title('Training Time Distribution', fontweight='bold')
        ax2.set_xlabel('Training Time (seconds)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. R² Score Comparison
        central_r2 = [r['r2'] for r in central_data if r['r2'] != 0]
        fed_r2 = [r['r2'] for r in fed_data if r['r2'] != 0]
        
        # Box plot for R² scores
        data_for_box = [central_r2, fed_r2]
        labels = ['Centralized', 'Federated']
        ax3.boxplot(data_for_box, labels=labels, patch_artist=True,
                   boxprops=dict(facecolor='lightblue', alpha=0.7),
                   medianprops=dict(color='red', linewidth=2))
        ax3.set_title('R² Score Distribution', fontweight='bold')
        ax3.set_ylabel('R² Score')
        ax3.grid(True, alpha=0.3)
        
        # 4. Confidence Intervals
        # Calculate 95% confidence intervals
        central_rmse_mean = np.mean(central_rmse)
        central_rmse_std = np.std(central_rmse)
        central_ci = 1.96 * central_rmse_std / np.sqrt(len(central_rmse))
        
        fed_rmse_mean = np.mean(fed_rmse)
        fed_rmse_std = np.std(fed_rmse)
        fed_ci = 1.96 * fed_rmse_std / np.sqrt(len(fed_rmse))
        
        models = ['Centralized', 'Federated']
        means = [central_rmse_mean, fed_rmse_mean]
        errors = [central_ci, fed_ci]
        
        bars = ax4.bar(models, means, yerr=errors, capsize=5, 
                      color=['skyblue', 'lightcoral'], alpha=0.7)
        ax4.set_title('RMSE with 95% Confidence Intervals', fontweight='bold')
        ax4.set_ylabel('RMSE (°C)')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, mean, error in zip(bars, means, errors):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + error + 0.01,
                    f'{mean:.3f}±{error:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Reliability analysis plot saved to {save_path}")
        
        plt.show()
    
    def print_reliability_report(self):
        """Print detailed reliability report."""
        print("\n" + "="*80)
        print("📊 FEDUHI RELIABILITY ANALYSIS REPORT")
        print("="*80)
        
        if self.results['centralized']:
            central_stats = self._calculate_statistics(self.results['centralized'], 'centralized')
            print(f"\n🧠 CENTRALIZED MODEL ({central_stats['num_successful_runs']}/{central_stats['total_runs']} successful runs):")
            print("-" * 60)
            print(f"Success Rate: {central_stats['success_rate']:.1f}%")
            print(f"RMSE: {central_stats['rmse']['mean']:.4f} ± {central_stats['rmse']['std']:.4f}°C")
            print(f"MAE:  {central_stats['mae']['mean']:.4f} ± {central_stats['mae']['std']:.4f}°C")
            print(f"R²:   {central_stats['r2']['mean']:.4f} ± {central_stats['r2']['std']:.4f}")
            print(f"Time: {central_stats['training_time']['mean']:.2f} ± {central_stats['training_time']['std']:.2f}s")
        
        if self.results['federated']:
            fed_stats = self._calculate_statistics(self.results['federated'], 'federated')
            print(f"\n🤝 FEDERATED MODEL ({fed_stats['num_successful_runs']}/{fed_stats['total_runs']} successful runs):")
            print("-" * 60)
            print(f"Success Rate: {fed_stats['success_rate']:.1f}%")
            print(f"RMSE: {fed_stats['rmse']['mean']:.4f} ± {fed_stats['rmse']['std']:.4f}°C")
            print(f"MAE:  {fed_stats['mae']['mean']:.4f} ± {fed_stats['mae']['std']:.4f}°C")
            print(f"R²:   {fed_stats['r2']['mean']:.4f} ± {fed_stats['r2']['std']:.4f}")
            print(f"Time: {fed_stats['training_time']['mean']:.2f} ± {fed_stats['training_time']['std']:.2f}s")
            if 'bandwidth' in fed_stats:
                print(f"Bandwidth: {fed_stats['bandwidth']['mean']:.2f} ± {fed_stats['bandwidth']['std']:.2f} MB")
        
        # Statistical significance test
        if self.results['centralized'] and self.results['federated']:
            print(f"\n⚖️ STATISTICAL COMPARISON:")
            print("-" * 60)
            
            central_rmse = [r['rmse'] for r in self.results['centralized'] if r['rmse'] > 0]
            fed_rmse = [r['rmse'] for r in self.results['federated'] if r['rmse'] > 0]
            
            if central_rmse and fed_rmse:
                central_mean = np.mean(central_rmse)
                fed_mean = np.mean(fed_rmse)
                difference = abs(central_mean - fed_mean)
                
                print(f"Mean RMSE Difference: {difference:.4f}°C")
                
                if central_mean < fed_mean:
                    improvement = ((fed_mean - central_mean) / fed_mean) * 100
                    print(f"Centralized model is {improvement:.1f}% more accurate")
                else:
                    improvement = ((central_mean - fed_mean) / central_mean) * 100
                    print(f"Federated model is {improvement:.1f}% more accurate")
        
        print("\n" + "="*80)
    
    def save_reliability_results(self, output_dir='results'):
        """Save reliability results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save raw results
        with open(f'{output_dir}/reliability_results.pkl', 'wb') as f:
            pickle.dump(self.results, f)
        
        # Save summary statistics
        central_stats = self._calculate_statistics(self.results['centralized'], 'centralized') if self.results['centralized'] else {}
        fed_stats = self._calculate_statistics(self.results['federated'], 'federated') if self.results['federated'] else {}
        
        summary = {
            'centralized': central_stats,
            'federated': fed_stats,
            'num_iterations': self.num_iterations
        }
        
        with open(f'{output_dir}/reliability_summary.pkl', 'wb') as f:
            pickle.dump(summary, f)
        
        print(f"Reliability results saved to {output_dir}/")


def main():
    """Main function to run reliable training."""
    print("🌡️ FedUHI Reliable Training Analysis")
    print("=" * 50)
    
    # Check if data exists
    if not os.path.exists('data/combined_zone_data.csv'):
        print("❌ No data found. Please run data generation first:")
        print("   python data_generation.py")
        return
    
    # Initialize reliable trainer
    trainer = ReliableTrainer(num_iterations=3, seed_base=42)
    
    # Run multiple centralized training
    central_stats = trainer.run_multiple_centralized_training()
    
    # Run multiple federated training
    fed_stats = trainer.run_multiple_federated_training()
    
    # Create reliability plots
    trainer.create_reliability_plots()
    
    # Print reliability report
    trainer.print_reliability_report()
    
    # Save results and models
    trainer.save_reliability_results()
    
    # Save all models using comprehensive model manager
    from model_manager import ModelManager
    model_manager = ModelManager()
    model_manager.save_reliable_models(trainer.results['centralized'], trainer.results['federated'])
    
    print("\n✅ Reliable training analysis completed!")


if __name__ == "__main__":
    main()
