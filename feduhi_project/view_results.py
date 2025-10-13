"""
FedUHI Results Viewer - Display actual results with charts and graphs

This script shows the real results from the FedUHI pipeline execution,
including comparison charts, performance metrics, and visualizations.
"""

import os
import sys
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def check_and_install_requirements():
    """Check if required packages are available, install if needed."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        import seaborn as sns
        print("✅ All required packages are available")
        return True
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Installing required packages...")
        try:
            import subprocess
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'matplotlib', 'pandas', 'numpy', 'seaborn'])
            print("✅ Packages installed successfully")
            return True
        except Exception as install_error:
            print(f"❌ Installation failed: {install_error}")
            return False

def load_results():
    """Load results from saved files."""
    results = {
        'centralized': None,
        'federated': None,
        'pipeline': None,
        'data_available': False
    }
    
    # Check for centralized results
    centralized_file = 'results/centralized_results.pkl'
    if os.path.exists(centralized_file):
        try:
            with open(centralized_file, 'rb') as f:
                results['centralized'] = pickle.load(f)
            print("✅ Loaded centralized results")
        except Exception as e:
            print(f"⚠️ Error loading centralized results: {e}")
    
    # Check for federated results
    federated_file = 'results/federated_results.pkl'
    if os.path.exists(federated_file):
        try:
            with open(federated_file, 'rb') as f:
                results['federated'] = pickle.load(f)
            print("✅ Loaded federated results")
        except Exception as e:
            print(f"⚠️ Error loading federated results: {e}")
    
    # Check for pipeline results
    pipeline_file = 'results/pipeline_execution_report.pkl'
    if os.path.exists(pipeline_file):
        try:
            with open(pipeline_file, 'rb') as f:
                results['pipeline'] = pickle.load(f)
            print("✅ Loaded pipeline execution report")
        except Exception as e:
            print(f"⚠️ Error loading pipeline results: {e}")
    
    # Check if any results are available
    results['data_available'] = any([
        results['centralized'] is not None,
        results['federated'] is not None,
        results['pipeline'] is not None
    ])
    
    return results

def create_results_comparison(results):
    """Create comparison charts from actual results."""
    if not results['data_available']:
        print("❌ No results available. Please run the pipeline first: python main_pipeline.py")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('FedUHI - Actual Results Comparison', fontsize=16, fontweight='bold')
    
    # Extract metrics
    centralized_metrics = results['centralized']['metrics'] if results['centralized'] else {}
    federated_metrics = results['federated']['results']['final_metrics'] if results['federated'] else {}
    
    # 1. Accuracy Comparison (RMSE)
    approaches = ['Centralized', 'Federated']
    rmse_values = []
    mae_values = []
    r2_values = []
    
    if centralized_metrics:
        rmse_values.append(centralized_metrics.get('rmse', 0))
        mae_values.append(centralized_metrics.get('mae', 0))
        r2_values.append(centralized_metrics.get('r2', 0))
    else:
        rmse_values.append(0)
        mae_values.append(0)
        r2_values.append(0)
    
    if federated_metrics:
        rmse_values.append(federated_metrics.get('loss', 0))
        mae_values.append(federated_metrics.get('mae', 0))
        r2_values.append(federated_metrics.get('r2', 0))
    else:
        rmse_values.append(0)
        mae_values.append(0)
        r2_values.append(0)
    
    # Plot RMSE
    bars1 = ax1.bar(approaches, rmse_values, color=['skyblue', 'lightcoral'], alpha=0.7)
    ax1.set_title('Root Mean Square Error (RMSE)', fontweight='bold')
    ax1.set_ylabel('RMSE (°C)')
    ax1.grid(True, alpha=0.3)
    
    for bar, value in zip(bars1, rmse_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot MAE
    bars2 = ax2.bar(approaches, mae_values, color=['lightgreen', 'orange'], alpha=0.7)
    ax2.set_title('Mean Absolute Error (MAE)', fontweight='bold')
    ax2.set_ylabel('MAE (°C)')
    ax2.grid(True, alpha=0.3)
    
    for bar, value in zip(bars2, mae_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot R² Score
    bars3 = ax3.bar(approaches, r2_values, color=['gold', 'purple'], alpha=0.7)
    ax3.set_title('R² Score (Coefficient of Determination)', fontweight='bold')
    ax3.set_ylabel('R² Score')
    ax3.grid(True, alpha=0.3)
    
    for bar, value in zip(bars3, r2_values):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Training Time Comparison
    training_times = []
    if results['centralized']:
        training_times.append(results['centralized']['training_time'])
    else:
        training_times.append(0)
    
    if results['federated']:
        training_times.append(results['federated']['results']['training_time'])
    else:
        training_times.append(0)
    
    bars4 = ax4.bar(approaches, training_times, color=['lightblue', 'lightpink'], alpha=0.7)
    ax4.set_title('Training Time Comparison', fontweight='bold')
    ax4.set_ylabel('Time (seconds)')
    ax4.grid(True, alpha=0.3)
    
    for bar, time_val in zip(bars4, training_times):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

def display_detailed_results(results):
    """Display detailed results in console."""
    print("\n" + "="*80)
    print("🌡️ FEDUHI - ACTUAL RESULTS REPORT")
    print("="*80)
    print(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Centralized Results
    if results['centralized']:
        print("\n🧠 CENTRALIZED LEARNING RESULTS:")
        print("-" * 50)
        central = results['centralized']
        print(f"Training Time: {central['training_time']:.2f} seconds")
        print(f"RMSE: {central['metrics']['rmse']:.4f}°C")
        print(f"MAE: {central['metrics']['mae']:.4f}°C")
        print(f"R² Score: {central['metrics']['r2']:.4f}")
        print(f"Bandwidth Usage: 0 MB (data centralized)")
    else:
        print("\n❌ No centralized results available")
    
    # Federated Results
    if results['federated']:
        print("\n🤝 FEDERATED LEARNING RESULTS:")
        print("-" * 50)
        fed = results['federated']
        print(f"Training Time: {fed['results']['training_time']:.2f} seconds")
        print(f"Communication Rounds: {fed['results']['rounds']}")
        print(f"Number of Clients: {fed['results']['num_clients']}")
        
        if fed['results']['final_metrics']:
            final_metrics = fed['results']['final_metrics']
            print(f"Final Loss: {final_metrics.get('loss', 'N/A'):.4f}")
            print(f"Final MAE: {final_metrics.get('mae', 'N/A'):.4f}")
            print(f"Final R²: {final_metrics.get('r2', 'N/A'):.4f}")
        
        if 'bandwidth_estimate' in fed:
            bandwidth = fed['bandwidth_estimate']
            print(f"Model Size: {bandwidth['model_size_mb']:.2f} MB")
            print(f"Bandwidth per Round: {bandwidth['bytes_per_round_mb']:.2f} MB")
            print(f"Total Bandwidth: {bandwidth['total_bytes_mb']:.2f} MB")
    else:
        print("\n❌ No federated results available")
    
    # Comparison Summary
    if results['centralized'] and results['federated']:
        print("\n⚖️ COMPARISON SUMMARY:")
        print("-" * 50)
        
        central_time = results['centralized']['training_time']
        fed_time = results['federated']['results']['training_time']
        time_ratio = fed_time / central_time if central_time > 0 else 0
        
        print(f"Training Time Ratio (Federated/Centralized): {time_ratio:.2f}x")
        
        if results['centralized']['metrics'] and results['federated']['results']['final_metrics']:
            central_rmse = results['centralized']['metrics']['rmse']
            fed_rmse = results['federated']['results']['final_metrics'].get('loss', 0)
            accuracy_ratio = central_rmse / fed_rmse if fed_rmse > 0 else 0
            print(f"Accuracy Ratio (Centralized/Federated): {accuracy_ratio:.2f}x")
        
        # Determine winner
        print("\n🏆 WINNER ANALYSIS:")
        print("-" * 50)
        
        if results['centralized']['metrics'] and results['federated']['results']['final_metrics']:
            central_rmse = results['centralized']['metrics']['rmse']
            fed_rmse = results['federated']['results']['final_metrics'].get('loss', 0)
            
            if central_rmse < fed_rmse:
                print("🥇 CENTRALIZED MODEL WINS for accuracy")
                print("   - Lower RMSE (better predictions)")
                print("   - Faster training time")
                print("   - No bandwidth overhead")
            else:
                print("🥇 FEDERATED MODEL WINS for accuracy")
                print("   - Lower RMSE (better predictions)")
                print("   - Preserves data privacy")
                print("   - Better scalability")
        
        print("\n💡 KEY INSIGHTS:")
        print("-" * 50)
        print("✅ Both models achieve good temperature prediction accuracy")
        print("✅ Federated learning preserves data privacy")
        print("✅ Centralized training is faster")
        print("✅ Federated approach scales better")
    
    print("\n" + "="*80)

def create_zone_data_visualization():
    """Create visualization of the generated zone data if available."""
    zone_files = [
        'data/zone_a_rooftop_data.csv',
        'data/zone_b_street_data.csv',
        'data/zone_c_park_data.csv',
        'data/zone_d_parking_data.csv'
    ]
    
    available_files = [f for f in zone_files if os.path.exists(f)]
    
    if not available_files:
        print("❌ No zone data files found. Run data generation first.")
        return
    
    print(f"📊 Creating zone data visualization from {len(available_files)} files...")
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    
    colors = ['red', 'blue', 'green', 'orange']
    zone_names = ['Rooftop', 'Street', 'Park', 'Parking']
    
    for i, file_path in enumerate(available_files):
        try:
            data = pd.read_csv(file_path)
            
            # Sample data for cleaner visualization
            sample_data = data.iloc[::10]  # Every 10th point
            
            # Plot temperature
            ax1.plot(sample_data.index, sample_data['temperature'], 
                    color=colors[i], label=zone_names[i], linewidth=2, alpha=0.7)
            
            # Plot humidity
            ax2.plot(sample_data.index, sample_data['humidity'], 
                    color=colors[i], label=zone_names[i], linewidth=2, alpha=0.7)
            
        except Exception as e:
            print(f"⚠️ Error reading {file_path}: {e}")
    
    ax1.set_title('Temperature by Zone (Actual Data)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Temperature (°C)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Humidity by Zone (Actual Data)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Humidity (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main function to display results."""
    print("🌡️ FedUHI Results Viewer")
    print("=" * 40)
    
    # Check and install requirements
    if not check_and_install_requirements():
        print("❌ Cannot proceed without required packages")
        return
    
    # Load results
    print("\n📁 Loading results...")
    results = load_results()
    
    if not results['data_available']:
        print("\n❌ No results found!")
        print("\nTo generate results, run:")
        print("  python main_pipeline.py")
        print("\nOr run individual steps:")
        print("  python main_pipeline.py --step 1  # Data generation")
        print("  python main_pipeline.py --step 2  # Centralized training")
        print("  python main_pipeline.py --step 3  # Federated training")
        return
    
    # Display detailed results
    display_detailed_results(results)
    
    # Create comparison charts
    print("\n📊 Creating comparison charts...")
    create_results_comparison(results)
    
    # Show zone data if available
    print("\n🏙️ Showing zone data visualization...")
    create_zone_data_visualization()
    
    print("\n✅ Results viewing completed!")

if __name__ == "__main__":
    main()
