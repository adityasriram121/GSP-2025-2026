"""
Comprehensive Trials System for FedUHI

This module implements the complete trial system that:
1. Runs 10 trials for centralized training
2. Runs 10 trials for federated training
3. Generates metrics for each trial (raw and visualized)
4. Calculates averages for each method
5. Creates head-to-head comparisons for each trial and averages
6. Exports 3 separate data sets as specified
"""

import os
import sys
import json
import time
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from datetime import datetime

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from code.centralized_training import train_centralized_model
from code.federated_training import prepare_federated_data, run_federated_simulation, estimate_bandwidth_usage


class ComprehensiveTrials:
    """Comprehensive trial system for running 10 trials of each method with full metrics."""
    
    def __init__(self, num_trials: int = 10, seed_base: int = 42, results_dir: str = 'results', models_dir: str = 'models'):
        """
        Initialize comprehensive trials system.
        
        Args:
            num_trials (int): Number of trials to run for each method (default: 10)
            seed_base (int): Base seed for reproducibility
            results_dir (str): Directory to save results
            models_dir (str): Directory to save models
        """
        self.num_trials = num_trials
        self.seed_base = seed_base
        self.results_dir = results_dir
        self.models_dir = models_dir
        
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'trials'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'centralized_metrics'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'federated_metrics'), exist_ok=True)
        os.makedirs(os.path.join(results_dir, 'head_to_head'), exist_ok=True)
        
        self.centralized_trials = []
        self.federated_trials = []
        
        plt.style.use('default')
        sns.set_palette("husl")
    
    def run_all_trials(self, data_path: str, data_dir: str):
        """
        Run all trials for both centralized and federated training.
        
        Args:
            data_path (str): Path to combined dataset for centralized training
            data_dir (str): Directory containing zone data for federated training
        """
        print("\n" + "="*80)
        print("COMPREHENSIVE TRIALS SYSTEM - RUNNING ALL TRIALS")
        print("="*80)
        print(f"Number of trials per method: {self.num_trials}")
        print(f"Results directory: {self.results_dir}")
        print("="*80)
        
        print(f"\n[1/2] Running {self.num_trials} Centralized Training Trials...")
        print("-" * 80)
        self._run_centralized_trials(data_path)
        
        print(f"\n[2/2] Running {self.num_trials} Federated Training Trials...")
        print("-" * 80)
        self._run_federated_trials(data_dir)
        
        print(f"\n[PROCESSING] Generating comprehensive metrics and visualizations...")
        print("-" * 80)
        self._generate_all_outputs()
        
        print("\n" + "="*80)
        print("ALL TRIALS COMPLETED SUCCESSFULLY!")
        print("="*80)
    
    def _run_centralized_trials(self, data_path: str):
        """Run multiple centralized training trials."""
        for trial_num in range(1, self.num_trials + 1):
            seed = self.seed_base + trial_num
            print(f"\n[TRIAL {trial_num}/{self.num_trials}] Centralized Training (Seed: {seed})")
            
            try:
                np.random.seed(seed)
                
                predictor, metrics, training_time = train_centralized_model(
                    data_path, 
                    test_split=0.2,
                    results_dir=self.results_dir,
                    models_dir=self.models_dir
                )
                
                trial_result = {
                    'trial_number': trial_num,
                    'seed': seed,
                    'metrics': metrics,
                    'training_time': training_time,
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'r2': metrics['r2'],
                    'mse': metrics.get('mse', 0),
                    'model_saved': True
                }
                
                self.centralized_trials.append(trial_result)
                
                print(f"   [OK] RMSE: {metrics['rmse']:.4f} C | MAE: {metrics['mae']:.4f} C | R2: {metrics['r2']:.4f} | Time: {training_time:.2f}s")
                
            except Exception as e:
                print(f"   [ERR] Error in trial {trial_num}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def _run_federated_trials(self, data_dir: str):
        """Run multiple federated training trials."""
        for trial_num in range(1, self.num_trials + 1):
            seed = self.seed_base + trial_num
            print(f"\n[TRIAL {trial_num}/{self.num_trials}] Federated Training (Seed: {seed})")
            
            try:
                np.random.seed(seed)
                
                training_data, test_data = prepare_federated_data(data_dir)
                
                bandwidth_estimate = estimate_bandwidth_usage(training_data, rounds=10)
                
                federated_result = run_federated_simulation(
                    training_data, 
                    test_data, 
                    rounds=10,
                    models_dir=self.models_dir
                )
                
                rmse = 0
                mae = 0
                r2 = 0
                
                if federated_result.get('final_metrics') and isinstance(federated_result['final_metrics'], dict):
                    final_metrics = federated_result['final_metrics']
                    loss = final_metrics.get('loss', 0)
                    rmse = np.sqrt(loss) if loss > 0 else 0
                    mae = final_metrics.get('mae', 0)
                    r2 = final_metrics.get('r2', 0)
                elif federated_result.get('history'):
                    history = federated_result['history']
                    if hasattr(history, 'metrics_distributed') and history.metrics_distributed:
                        metrics = history.metrics_distributed
                        loss = metrics['loss'][-1][1] if metrics.get('loss') and isinstance(metrics['loss'], list) else 0
                        rmse = np.sqrt(loss) if loss > 0 else 0
                        mae = metrics['mae'][-1][1] if metrics.get('mae') and isinstance(metrics['mae'], list) else 0
                        r2 = metrics['r2'][-1][1] if metrics.get('r2') and isinstance(metrics['r2'], list) else 0
                
                trial_result = {
                    'trial_number': trial_num,
                    'seed': seed,
                    'training_time': federated_result['training_time'],
                    'rounds': federated_result['rounds'],
                    'bandwidth_estimate': bandwidth_estimate,
                    'final_metrics': federated_result.get('final_metrics'),
                    'history': federated_result.get('history'),
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'num_clients': federated_result.get('num_clients', 4)
                }
                
                self.federated_trials.append(trial_result)
                
                print(f"   [OK] RMSE: {rmse:.4f} C | MAE: {mae:.4f} C | R2: {r2:.4f} | Time: {trial_result['training_time']:.2f}s | Bandwidth: {bandwidth_estimate['total_bytes_mb']:.2f} MB")
                
            except Exception as e:
                print(f"   [ERR] Error in trial {trial_num}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    def _calculate_averages(self, trials: List[Dict], method: str) -> Dict:
        """Calculate average metrics for a set of trials."""
        if not trials:
            return {}
        
        rmse_values = [t['rmse'] for t in trials if t['rmse'] > 0]
        mae_values = [t['mae'] for t in trials if t['mae'] > 0]
        r2_values = [t['r2'] for t in trials if t['r2'] != 0]
        time_values = [t['training_time'] for t in trials]
        
        averages = {
            'method': method,
            'num_trials': len(trials),
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
                'mean': np.mean(time_values),
                'std': np.std(time_values),
                'min': np.min(time_values),
                'max': np.max(time_values),
                'values': time_values
            }
        }
        
        if method == 'federated' and trials:
            bandwidth_values = [t['bandwidth_estimate']['total_bytes_mb'] for t in trials if t.get('bandwidth_estimate')]
            if bandwidth_values:
                averages['bandwidth'] = {
                    'mean': np.mean(bandwidth_values),
                    'std': np.std(bandwidth_values),
                    'min': np.min(bandwidth_values),
                    'max': np.max(bandwidth_values),
                    'values': bandwidth_values
                }
        
        return averages
    
    def _generate_all_outputs(self):
        """Generate all three output sets as specified."""
        centralized_avg = self._calculate_averages(self.centralized_trials, 'centralized')
        federated_avg = self._calculate_averages(self.federated_trials, 'federated')
        
        print("\n[SET 1] Generating Centralized Metrics Output...")
        self._generate_centralized_outputs(centralized_avg)
        
        print("\n[SET 2] Generating Federated Metrics Output...")
        self._generate_federated_outputs(federated_avg)
        
        print("\n[SET 3] Generating Head-to-Head Comparisons...")
        self._generate_head_to_head_outputs(centralized_avg, federated_avg)
        
        print("\n[EXPORT] Saving raw data files...")
        self._save_raw_data(centralized_avg, federated_avg)
    
    def _generate_centralized_outputs(self, averages: Dict):
        """Generate Set 1: Centralized metrics (every trial + average)."""
        output_dir = os.path.join(self.results_dir, 'centralized_metrics')
        
        raw_data = []
        for trial in self.centralized_trials:
            raw_data.append({
                'Trial': trial['trial_number'],
                'Seed': trial['seed'],
                'RMSE': trial['rmse'],
                'MAE': trial['mae'],
                'R2': trial['r2'],
                'Training_Time_Seconds': trial['training_time'],
                'MSE': trial.get('mse', 0)
            })
        
        raw_data.append({
            'Trial': 'AVERAGE',
            'Seed': 'N/A',
            'RMSE': averages['rmse']['mean'],
            'MAE': averages['mae']['mean'],
            'R2': averages['r2']['mean'],
            'Training_Time_Seconds': averages['training_time']['mean'],
            'MSE': 0
        })
        
        df = pd.DataFrame(raw_data)
        df.to_csv(os.path.join(output_dir, 'centralized_raw_metrics.csv'), index=False)
        
        self._visualize_centralized_metrics(output_dir, averages)
        
        print(f"   [OK] Centralized metrics saved to: {output_dir}/")
    
    def _generate_federated_outputs(self, averages: Dict):
        """Generate Set 2: Federated metrics (every trial + average)."""
        output_dir = os.path.join(self.results_dir, 'federated_metrics')
        
        raw_data = []
        for trial in self.federated_trials:
            raw_data.append({
                'Trial': trial['trial_number'],
                'Seed': trial['seed'],
                'RMSE': trial['rmse'],
                'MAE': trial['mae'],
                'R2': trial['r2'],
                'Training_Time_Seconds': trial['training_time'],
                'Bandwidth_MB': trial['bandwidth_estimate']['total_bytes_mb'],
                'Rounds': trial['rounds'],
                'Num_Clients': trial['num_clients']
            })
        
        avg_bandwidth = averages.get('bandwidth', {}).get('mean', 0) if 'bandwidth' in averages else 0
        raw_data.append({
            'Trial': 'AVERAGE',
            'Seed': 'N/A',
            'RMSE': averages['rmse']['mean'],
            'MAE': averages['mae']['mean'],
            'R2': averages['r2']['mean'],
            'Training_Time_Seconds': averages['training_time']['mean'],
            'Bandwidth_MB': avg_bandwidth,
            'Rounds': 10,
            'Num_Clients': 4
        })
        
        df = pd.DataFrame(raw_data)
        df.to_csv(os.path.join(output_dir, 'federated_raw_metrics.csv'), index=False)
        
        self._visualize_federated_metrics(output_dir, averages)
        
        print(f"   [OK] Federated metrics saved to: {output_dir}/")
    
    def _generate_head_to_head_outputs(self, centralized_avg: Dict, federated_avg: Dict):
        """Generate Set 3: Head-to-head comparisons (each trial + average)."""
        output_dir = os.path.join(self.results_dir, 'head_to_head')
        
        num_comparisons = min(len(self.centralized_trials), len(self.federated_trials))
        
        raw_data = []
        for i in range(num_comparisons):
            central = self.centralized_trials[i]
            federated = self.federated_trials[i]
            
            raw_data.append({
                'Trial': central['trial_number'],
                'Centralized_RMSE': central['rmse'],
                'Federated_RMSE': federated['rmse'],
                'RMSE_Difference': central['rmse'] - federated['rmse'],
                'Centralized_MAE': central['mae'],
                'Federated_MAE': federated['mae'],
                'MAE_Difference': central['mae'] - federated['mae'],
                'Centralized_R2': central['r2'],
                'Federated_R2': federated['r2'],
                'R2_Difference': central['r2'] - federated['r2'],
                'Centralized_Time': central['training_time'],
                'Federated_Time': federated['training_time'],
                'Time_Difference': central['training_time'] - federated['training_time'],
                'Winner_RMSE': 'Centralized' if central['rmse'] < federated['rmse'] else 'Federated',
                'Winner_MAE': 'Centralized' if central['mae'] < federated['mae'] else 'Federated',
                'Winner_R2': 'Centralized' if central['r2'] > federated['r2'] else 'Federated'
            })
        
        raw_data.append({
            'Trial': 'AVERAGE',
            'Centralized_RMSE': centralized_avg['rmse']['mean'],
            'Federated_RMSE': federated_avg['rmse']['mean'],
            'RMSE_Difference': centralized_avg['rmse']['mean'] - federated_avg['rmse']['mean'],
            'Centralized_MAE': centralized_avg['mae']['mean'],
            'Federated_MAE': federated_avg['mae']['mean'],
            'MAE_Difference': centralized_avg['mae']['mean'] - federated_avg['mae']['mean'],
            'Centralized_R2': centralized_avg['r2']['mean'],
            'Federated_R2': federated_avg['r2']['mean'],
            'R2_Difference': centralized_avg['r2']['mean'] - federated_avg['r2']['mean'],
            'Centralized_Time': centralized_avg['training_time']['mean'],
            'Federated_Time': federated_avg['training_time']['mean'],
            'Time_Difference': centralized_avg['training_time']['mean'] - federated_avg['training_time']['mean'],
            'Winner_RMSE': 'Centralized' if centralized_avg['rmse']['mean'] < federated_avg['rmse']['mean'] else 'Federated',
            'Winner_MAE': 'Centralized' if centralized_avg['mae']['mean'] < federated_avg['mae']['mean'] else 'Federated',
            'Winner_R2': 'Centralized' if centralized_avg['r2']['mean'] > federated_avg['r2']['mean'] else 'Federated'
        })
        
        df = pd.DataFrame(raw_data)
        df.to_csv(os.path.join(output_dir, 'head_to_head_raw_metrics.csv'), index=False)
        
        self._visualize_head_to_head_comparisons(output_dir, centralized_avg, federated_avg)
        
        print(f"   [OK] Head-to-head comparisons saved to: {output_dir}/")
    
    def _visualize_centralized_metrics(self, output_dir: str, averages: Dict):
        """Create visualizations for centralized metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Centralized Training Metrics - All Trials + Average', fontsize=16, fontweight='bold')
        
        trial_nums = [t['trial_number'] for t in self.centralized_trials]
        rmse_vals = [t['rmse'] for t in self.centralized_trials]
        mae_vals = [t['mae'] for t in self.centralized_trials]
        r2_vals = [t['r2'] for t in self.centralized_trials]
        time_vals = [t['training_time'] for t in self.centralized_trials]
        
        ax1.bar(trial_nums, rmse_vals, alpha=0.7, color='skyblue', label='Trials')
        ax1.axhline(y=averages['rmse']['mean'], color='red', linestyle='--', linewidth=2, label=f'Average: {averages["rmse"]["mean"]:.4f}')
        ax1.set_title('RMSE by Trial', fontweight='bold')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('RMSE (°C)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(trial_nums, mae_vals, alpha=0.7, color='lightgreen', label='Trials')
        ax2.axhline(y=averages['mae']['mean'], color='red', linestyle='--', linewidth=2, label=f'Average: {averages["mae"]["mean"]:.4f}')
        ax2.set_title('MAE by Trial', fontweight='bold')
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('MAE (°C)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.bar(trial_nums, r2_vals, alpha=0.7, color='gold', label='Trials')
        ax3.axhline(y=averages['r2']['mean'], color='red', linestyle='--', linewidth=2, label=f'Average: {averages["r2"]["mean"]:.4f}')
        ax3.set_title('R2 Score by Trial', fontweight='bold')
        ax3.set_xlabel('Trial Number')
        ax3.set_ylabel('R2 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.bar(trial_nums, time_vals, alpha=0.7, color='orange', label='Trials')
        ax4.axhline(y=averages['training_time']['mean'], color='red', linestyle='--', linewidth=2, label=f'Average: {averages["training_time"]["mean"]:.2f}s')
        ax4.set_title('Training Time by Trial', fontweight='bold')
        ax4.set_xlabel('Trial Number')
        ax4.set_ylabel('Time (seconds)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'centralized_metrics_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    def _visualize_federated_metrics(self, output_dir: str, averages: Dict):
        """Create visualizations for federated metrics."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Federated Training Metrics - All Trials + Average', fontsize=16, fontweight='bold')
        
        trial_nums = [t['trial_number'] for t in self.federated_trials]
        rmse_vals = [t['rmse'] for t in self.federated_trials]
        mae_vals = [t['mae'] for t in self.federated_trials]
        r2_vals = [t['r2'] for t in self.federated_trials]
        time_vals = [t['training_time'] for t in self.federated_trials]
        
        ax1.bar(trial_nums, rmse_vals, alpha=0.7, color='lightcoral', label='Trials')
        ax1.axhline(y=averages['rmse']['mean'], color='red', linestyle='--', linewidth=2, label=f'Average: {averages["rmse"]["mean"]:.4f}')
        ax1.set_title('RMSE by Trial', fontweight='bold')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('RMSE (°C)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(trial_nums, mae_vals, alpha=0.7, color='lightgreen', label='Trials')
        ax2.axhline(y=averages['mae']['mean'], color='red', linestyle='--', linewidth=2, label=f'Average: {averages["mae"]["mean"]:.4f}')
        ax2.set_title('MAE by Trial', fontweight='bold')
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('MAE (°C)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.bar(trial_nums, r2_vals, alpha=0.7, color='gold', label='Trials')
        ax3.axhline(y=averages['r2']['mean'], color='red', linestyle='--', linewidth=2, label=f'Average: {averages["r2"]["mean"]:.4f}')
        ax3.set_title('R2 Score by Trial', fontweight='bold')
        ax3.set_xlabel('Trial Number')
        ax3.set_ylabel('R2 Score')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.bar(trial_nums, time_vals, alpha=0.7, color='orange', label='Trials')
        ax4.axhline(y=averages['training_time']['mean'], color='red', linestyle='--', linewidth=2, label=f'Average: {averages["training_time"]["mean"]:.2f}s')
        ax4.set_title('Training Time by Trial', fontweight='bold')
        ax4.set_xlabel('Trial Number')
        ax4.set_ylabel('Time (seconds)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'federated_metrics_visualization.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        if 'bandwidth' in averages:
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))
            bandwidth_vals = [t['bandwidth_estimate']['total_bytes_mb'] for t in self.federated_trials]
            ax.bar(trial_nums, bandwidth_vals, alpha=0.7, color='purple', label='Trials')
            ax.axhline(y=averages['bandwidth']['mean'], color='red', linestyle='--', linewidth=2, label=f'Average: {averages["bandwidth"]["mean"]:.2f} MB')
            ax.set_title('Bandwidth Usage by Trial', fontweight='bold')
            ax.set_xlabel('Trial Number')
            ax.set_ylabel('Bandwidth (MB)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'federated_bandwidth_visualization.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _visualize_head_to_head_comparisons(self, output_dir: str, centralized_avg: Dict, federated_avg: Dict):
        """Create visualizations for head-to-head comparisons."""
        num_comparisons = min(len(self.centralized_trials), len(self.federated_trials))
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Head-to-Head Comparison - Centralized vs Federated (All Trials + Average)', fontsize=16, fontweight='bold')
        
        trial_nums = list(range(1, num_comparisons + 1))
        central_rmse = [self.centralized_trials[i]['rmse'] for i in range(num_comparisons)]
        fed_rmse = [self.federated_trials[i]['rmse'] for i in range(num_comparisons)]
        central_mae = [self.centralized_trials[i]['mae'] for i in range(num_comparisons)]
        fed_mae = [self.federated_trials[i]['mae'] for i in range(num_comparisons)]
        central_r2 = [self.centralized_trials[i]['r2'] for i in range(num_comparisons)]
        fed_r2 = [self.federated_trials[i]['r2'] for i in range(num_comparisons)]
        central_time = [self.centralized_trials[i]['training_time'] for i in range(num_comparisons)]
        fed_time = [self.federated_trials[i]['training_time'] for i in range(num_comparisons)]
        
        x = np.arange(len(trial_nums))
        width = 0.35
        
        ax1.bar(x - width/2, central_rmse, width, label='Centralized', alpha=0.7, color='skyblue')
        ax1.bar(x + width/2, fed_rmse, width, label='Federated', alpha=0.7, color='lightcoral')
        ax1.axhline(y=centralized_avg['rmse']['mean'], color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Centralized Avg: {centralized_avg["rmse"]["mean"]:.4f}')
        ax1.axhline(y=federated_avg['rmse']['mean'], color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Federated Avg: {federated_avg["rmse"]["mean"]:.4f}')
        ax1.set_title('RMSE Comparison', fontweight='bold')
        ax1.set_xlabel('Trial Number')
        ax1.set_ylabel('RMSE (°C)')
        ax1.set_xticks(x)
        ax1.set_xticklabels(trial_nums)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(x - width/2, central_mae, width, label='Centralized', alpha=0.7, color='lightgreen')
        ax2.bar(x + width/2, fed_mae, width, label='Federated', alpha=0.7, color='orange')
        ax2.axhline(y=centralized_avg['mae']['mean'], color='green', linestyle='--', linewidth=2, alpha=0.7, label=f'Centralized Avg: {centralized_avg["mae"]["mean"]:.4f}')
        ax2.axhline(y=federated_avg['mae']['mean'], color='orange', linestyle='--', linewidth=2, alpha=0.7, label=f'Federated Avg: {federated_avg["mae"]["mean"]:.4f}')
        ax2.set_title('MAE Comparison', fontweight='bold')
        ax2.set_xlabel('Trial Number')
        ax2.set_ylabel('MAE (°C)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(trial_nums)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        ax3.bar(x - width/2, central_r2, width, label='Centralized', alpha=0.7, color='gold')
        ax3.bar(x + width/2, fed_r2, width, label='Federated', alpha=0.7, color='purple')
        ax3.axhline(y=centralized_avg['r2']['mean'], color='gold', linestyle='--', linewidth=2, alpha=0.7, label=f'Centralized Avg: {centralized_avg["r2"]["mean"]:.4f}')
        ax3.axhline(y=federated_avg['r2']['mean'], color='purple', linestyle='--', linewidth=2, alpha=0.7, label=f'Federated Avg: {federated_avg["r2"]["mean"]:.4f}')
        ax3.set_title('R2 Score Comparison', fontweight='bold')
        ax3.set_xlabel('Trial Number')
        ax3.set_ylabel('R2 Score')
        ax3.set_xticks(x)
        ax3.set_xticklabels(trial_nums)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        ax4.bar(x - width/2, central_time, width, label='Centralized', alpha=0.7, color='lightblue')
        ax4.bar(x + width/2, fed_time, width, label='Federated', alpha=0.7, color='lightpink')
        ax4.axhline(y=centralized_avg['training_time']['mean'], color='blue', linestyle='--', linewidth=2, alpha=0.7, label=f'Centralized Avg: {centralized_avg["training_time"]["mean"]:.2f}s')
        ax4.axhline(y=federated_avg['training_time']['mean'], color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Federated Avg: {federated_avg["training_time"]["mean"]:.2f}s')
        ax4.set_title('Training Time Comparison', fontweight='bold')
        ax4.set_xlabel('Trial Number')
        ax4.set_ylabel('Time (seconds)')
        ax4.set_xticks(x)
        ax4.set_xticklabels(trial_nums)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'head_to_head_comparison_all_trials.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle('Average Comparison - Centralized vs Federated', fontsize=16, fontweight='bold')
        
        methods = ['Centralized', 'Federated']
        
        ax1.bar(methods, [centralized_avg['rmse']['mean'], federated_avg['rmse']['mean']], 
                color=['skyblue', 'lightcoral'], alpha=0.7)
        ax1.errorbar(methods, [centralized_avg['rmse']['mean'], federated_avg['rmse']['mean']],
                     yerr=[centralized_avg['rmse']['std'], federated_avg['rmse']['std']],
                     fmt='none', color='black', capsize=5)
        ax1.set_title('Average RMSE', fontweight='bold')
        ax1.set_ylabel('RMSE (°C)')
        ax1.grid(True, alpha=0.3)
        
        ax2.bar(methods, [centralized_avg['mae']['mean'], federated_avg['mae']['mean']],
                color=['lightgreen', 'orange'], alpha=0.7)
        ax2.errorbar(methods, [centralized_avg['mae']['mean'], federated_avg['mae']['mean']],
                     yerr=[centralized_avg['mae']['std'], federated_avg['mae']['std']],
                     fmt='none', color='black', capsize=5)
        ax2.set_title('Average MAE', fontweight='bold')
        ax2.set_ylabel('MAE (°C)')
        ax2.grid(True, alpha=0.3)
        
        ax3.bar(methods, [centralized_avg['r2']['mean'], federated_avg['r2']['mean']],
                color=['gold', 'purple'], alpha=0.7)
        ax3.errorbar(methods, [centralized_avg['r2']['mean'], federated_avg['r2']['mean']],
                     yerr=[centralized_avg['r2']['std'], federated_avg['r2']['std']],
                     fmt='none', color='black', capsize=5)
        ax3.set_title('Average R2 Score', fontweight='bold')
        ax3.set_ylabel('R2 Score')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'head_to_head_average_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        for i in range(num_comparisons):
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            fig.suptitle(f'Trial {i+1} Head-to-Head Comparison', fontsize=16, fontweight='bold')
            
            central = self.centralized_trials[i]
            federated = self.federated_trials[i]
            
            metrics_names = ['RMSE', 'MAE', 'R2']
            central_vals = [central['rmse'], central['mae'], central['r2']]
            fed_vals = [federated['rmse'], federated['mae'], federated['r2']]
            
            x = np.arange(len(metrics_names))
            width = 0.35
            
            ax1.bar(x - width/2, central_vals, width, label='Centralized', alpha=0.7, color='skyblue')
            ax1.bar(x + width/2, fed_vals, width, label='Federated', alpha=0.7, color='lightcoral')
            ax1.set_title(f'Trial {i+1} - Metrics Comparison', fontweight='bold')
            ax1.set_ylabel('Metric Value')
            ax1.set_xticks(x)
            ax1.set_xticklabels(metrics_names)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            ax2.bar(['Centralized', 'Federated'], [central['training_time'], federated['training_time']],
                    color=['lightblue', 'lightpink'], alpha=0.7)
            ax2.set_title(f'Trial {i+1} - Training Time', fontweight='bold')
            ax2.set_ylabel('Time (seconds)')
            ax2.grid(True, alpha=0.3)
            
            winners = {
                'RMSE': 'Centralized' if central['rmse'] < federated['rmse'] else 'Federated',
                'MAE': 'Centralized' if central['mae'] < federated['mae'] else 'Federated',
                'R2': 'Centralized' if central['r2'] > federated['r2'] else 'Federated'
            }
            ax3.axis('off')
            winner_text = f"Trial {i+1} Winners:\n\n"
            for metric, winner in winners.items():
                winner_text += f"{metric}: {winner}\n"
            winner_text += f"\nTraining Time:\nCentralized: {central['training_time']:.2f}s\nFederated: {federated['training_time']:.2f}s"
            ax3.text(0.5, 0.5, winner_text, ha='center', va='center', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.5),
                    transform=ax3.transAxes)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'trial_{i+1}_comparison.png'), dpi=300, bbox_inches='tight')
            plt.close()
    
    def _save_raw_data(self, centralized_avg: Dict, federated_avg: Dict):
        """Save all raw data to JSON and CSV files."""
        all_data = {
            'timestamp': datetime.now().isoformat(),
            'num_trials': self.num_trials,
            'seed_base': self.seed_base,
            'centralized_trials': [
                {
                    'trial_number': t['trial_number'],
                    'seed': t['seed'],
                    'rmse': t['rmse'],
                    'mae': t['mae'],
                    'r2': t['r2'],
                    'training_time': t['training_time']
                } for t in self.centralized_trials
            ],
            'federated_trials': [
                {
                    'trial_number': t['trial_number'],
                    'seed': t['seed'],
                    'rmse': t['rmse'],
                    'mae': t['mae'],
                    'r2': t['r2'],
                    'training_time': t['training_time'],
                    'bandwidth_mb': t['bandwidth_estimate']['total_bytes_mb']
                } for t in self.federated_trials
            ],
            'centralized_averages': centralized_avg,
            'federated_averages': federated_avg
        }
        
        json_path = os.path.join(self.results_dir, 'all_trials_raw_data.json')
        with open(json_path, 'w') as f:
            json.dump(all_data, f, indent=2, default=str)
        
        pickle_path = os.path.join(self.results_dir, 'all_trials_data.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'centralized_trials': self.centralized_trials,
                'federated_trials': self.federated_trials,
                'centralized_averages': centralized_avg,
                'federated_averages': federated_avg
            }, f)
        
        print(f"   [OK] Raw data saved to: {json_path}")
        print(f"   [OK] Pickle data saved to: {pickle_path}")


def main():
    """Main function to run comprehensive trials."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Trials System for FedUHI')
    parser.add_argument('--trials', type=int, default=10,
                       help='Number of trials to run per method (default: 10)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Base random seed (default: 42)')
    
    args = parser.parse_args()
    
    data_path = 'data/combined_zone_data.csv'
    data_dir = 'data'
    
    if not os.path.exists(data_path):
        print(f"ERROR: Combined data not found at {data_path}")
        print("Please run data generation first: python main_pipeline.py --step 1")
        return
    
    if not os.path.exists(data_dir):
        print(f"ERROR: Data directory not found: {data_dir}")
        print("Please run data generation first: python main_pipeline.py --step 1")
        return
    
    trials_system = ComprehensiveTrials(
        num_trials=args.trials,
        seed_base=args.seed,
        results_dir='results',
        models_dir='models'
    )
    
    trials_system.run_all_trials(data_path, data_dir)
    
    print("\n[OK] Comprehensive trials completed!")
    print(f"[OK] Results saved to: results/")
    print(f"  - Centralized metrics: results/centralized_metrics/")
    print(f"  - Federated metrics: results/federated_metrics/")
    print(f"  - Head-to-head comparisons: results/head_to_head/")


if __name__ == "__main__":
    main()

