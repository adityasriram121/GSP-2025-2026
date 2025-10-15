"""
Federated Urban Heat Island (FedUHI) - Main Pipeline

This is the main orchestrator that runs the complete FedUHI simulation pipeline:

1. Generate synthetic sensor data for 4 urban zones
2. Train centralized ML model on combined data
3. Train federated ML model using Flower framework
4. Compare results and generate comprehensive visualizations
5. Produce final analysis report

Usage: python main_pipeline.py
"""

import os
import sys
import time
import math
import pickle
import traceback
import subprocess
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def install_requirements():
    """Install required packages automatically."""
    project_root = Path(__file__).resolve().parent
    requirements_file = project_root / 'requirements.txt'
    if requirements_file.exists():
        print("📦 Installing required packages...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', str(requirements_file)])
            print("✅ All requirements installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing requirements: {e}")
            print("Please install manually: pip install -r requirements.txt")
            return False
    else:
        print(f"⚠️ Requirements file not found: {requirements_file}")
        return False
    return True

# Install requirements first
install_requirements()

# Import project modules
try:
    from data_generation import ZoneDataGenerator
    from centralized_training import train_centralized_model
    from federated_training import prepare_federated_data, run_federated_simulation, estimate_bandwidth_usage
    from metrics_visualization import FedUHIVisualizer
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure all requirements are installed: pip install -r requirements.txt")
    sys.exit(1)


class FedUHIPipeline:
    """Main pipeline orchestrator for FedUHI project."""
    
    def __init__(self, seed=42):
        """
        Initialize the FedUHI pipeline.
        
        Args:
            seed (int): Random seed for reproducibility
        """
        self.seed = seed
        self.start_time = None
        self.end_time = None
        self.results = {
            'data_generation': {},
            'centralized_training': {},
            'federated_training': {},
            'visualization': {},
            'model_selection': None
        }
        self.results_dir = 'results'

        # Create necessary directories
        self._create_directories()
        
        print("🌡️  Federated Urban Heat Island (FedUHI) Pipeline")
        print("=" * 60)
        print("A comprehensive simulation of federated learning for urban temperature prediction")
        print("=" * 60)
    
    def _create_directories(self):
        """Create necessary directories for the pipeline."""
        directories = ['data', 'models', 'results']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created directory: {directory}")
    
    def _log_step(self, step_name: str, start_time: float = None):
        """
        Log pipeline step execution.
        
        Args:
            step_name (str): Name of the pipeline step
            start_time (float): Start time of the step
        """
        if start_time:
            elapsed = time.time() - start_time
            print(f"⏱️  {step_name} completed in {elapsed:.2f} seconds")
        else:
            print(f"\n🚀 Starting: {step_name}")
            print("-" * 40)
    
    def step1_generate_data(self):
        """
        Step 1: Generate synthetic sensor data for all urban zones.
        
        Returns:
            dict: Data generation results
        """
        step_start = time.time()
        self._log_step("Step 1: Data Generation")
        
        try:
            # Initialize data generator
            generator = ZoneDataGenerator(seed=self.seed)
            
            # Generate data for all zones (7 days, 4 samples per hour)
            print("📊 Generating synthetic sensor data...")
            all_data = generator.generate_all_zones_data(days=7, samples_per_hour=4)
            
            # Save individual zone data
            print("💾 Saving zone-specific data...")
            generator.save_data_to_csv(all_data, output_dir='data')
            
            # Create and save combined dataset
            print("🔗 Creating combined dataset...")
            combined_data = generator.create_combined_dataset(all_data)
            combined_data.to_csv('data/combined_zone_data.csv', index=False)
            
            # Generate statistics
            stats = generator.get_data_statistics(all_data)
            
            # Create initial visualization
            print("📈 Creating data visualization...")
            generator.plot_zone_comparison(all_data, save_path='results/zone_comparison.png')
            
            # Store results
            self.results['data_generation'] = {
                'zones_created': list(all_data.keys()),
                'total_samples': len(combined_data),
                'stats': stats,
                'files_created': [
                    'data/zone_a_rooftop_data.csv',
                    'data/zone_b_street_data.csv', 
                    'data/zone_c_park_data.csv',
                    'data/zone_d_parking_data.csv',
                    'data/combined_zone_data.csv'
                ]
            }
            
            self._log_step("Step 1: Data Generation", step_start)
            return self.results['data_generation']
            
        except Exception as e:
            print(f"❌ Error in data generation: {str(e)}")
            traceback.print_exc()
            raise
    
    def step2_centralized_training(self):
        """
        Step 2: Train centralized ML model on combined data.
        
        Returns:
            dict: Centralized training results
        """
        step_start = time.time()
        self._log_step("Step 2: Centralized Training")
        
        try:
            # Check if data exists
            data_path = 'data/combined_zone_data.csv'
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Combined data not found at {data_path}")
            
            print("🧠 Training centralized model...")
            model, metrics, training_time = train_centralized_model(data_path, test_split=0.2)
            
            # Save centralized results
            centralized_results = {
                'model': model,
                'metrics': metrics,
                'training_time': training_time,
                'test_split': 0.2
            }
            
            with open('results/centralized_results.pkl', 'wb') as f:
                pickle.dump(centralized_results, f)
            
            self.results['centralized_training'] = centralized_results
            
            self._log_step("Step 2: Centralized Training", step_start)
            return centralized_results
            
        except Exception as e:
            print(f"❌ Error in centralized training: {str(e)}")
            traceback.print_exc()
            raise
    
    def step3_federated_training(self):
        """
        Step 3: Train federated ML model using Flower framework.
        
        Returns:
            dict: Federated training results
        """
        step_start = time.time()
        self._log_step("Step 3: Federated Training")
        
        try:
            # Check if zone data exists
            data_dir = 'data'
            required_files = [
                'zone_a_rooftop_data.csv',
                'zone_b_street_data.csv',
                'zone_c_park_data.csv', 
                'zone_d_parking_data.csv'
            ]
            
            for file in required_files:
                if not os.path.exists(os.path.join(data_dir, file)):
                    raise FileNotFoundError(f"Required file not found: {file}")
            
            print("🤝 Preparing federated data...")
            training_data, test_data = prepare_federated_data(data_dir)
            
            print("📊 Estimating bandwidth usage...")
            bandwidth_estimate = estimate_bandwidth_usage(training_data, rounds=5)

            print("🔄 Running federated simulation...")
            federated_results = run_federated_simulation(training_data, test_data, rounds=5)

            # Package federated results for downstream visualizations
            federated_package = {
                'results': federated_results,
                'bandwidth_estimate': bandwidth_estimate
            }

            # Save federated results
            with open('results/federated_results.pkl', 'wb') as f:
                pickle.dump(federated_package, f)

            # Store flattened view for pipeline reporting
            self.results['federated_training'] = {
                **federated_results,
                'bandwidth_estimate': bandwidth_estimate
            }

            self._log_step("Step 3: Federated Training", step_start)
            return self.results['federated_training']
            
        except Exception as e:
            print(f"❌ Error in federated training: {str(e)}")
            traceback.print_exc()
            raise
    
    def step4_visualization(self):
        """
        Step 4: Generate comprehensive visualizations and analysis.
        
        Returns:
            dict: Visualization results
        """
        step_start = time.time()
        self._log_step("Step 4: Visualization & Analysis")
        
        try:
            print("📊 Initializing visualizer...")
            visualizer = FedUHIVisualizer()
            
            print("📁 Loading results...")
            visualizer.load_results()
            
            if not visualizer.centralized_results or not visualizer.federated_results:
                raise ValueError("Missing results files for visualization")
            
            print("📈 Generating comparison summary...")
            summary_df = visualizer.create_comparison_summary('results/comparison_summary.csv')
            
            print("📊 Creating accuracy comparison plot...")
            visualizer.plot_accuracy_comparison('results/accuracy_comparison.png')
            
            print("⏱️ Creating training analysis plot...")
            visualizer.plot_training_analysis('results/training_analysis.png')
            
            print("🏙️ Creating zone-specific analysis...")
            visualizer.plot_zone_specific_analysis('results/zone_analysis.png')
            
            print("📋 Creating comprehensive report...")
            visualizer.create_comprehensive_report('results/comprehensive_report.png')
            
            print("📝 Generating console report...")
            visualizer.generate_console_report()
            
            # Store visualization results
            self.results['visualization'] = {
                'summary_dataframe': summary_df,
                'plots_created': [
                    'results/comparison_summary.csv',
                    'results/accuracy_comparison.png',
                    'results/training_analysis.png', 
                    'results/zone_analysis.png',
                    'results/comprehensive_report.png'
                ]
            }
            
            self._log_step("Step 4: Visualization & Analysis", step_start)
            return self.results['visualization']
            
        except Exception as e:
            print(f"❌ Error in visualization: {str(e)}")
            traceback.print_exc()
            raise

    def determine_preferred_training(self):
        """Determine which training approach currently performs best."""
        centralized = self.results.get('centralized_training') or {}
        federated = self.results.get('federated_training') or {}

        if not centralized or not federated:
            print("⚠️ Unable to determine preferred model: missing results")
            return None

        centralized_metrics = centralized.get('metrics') or {}
        federated_metrics = federated.get('final_metrics') or {}

        if not centralized_metrics and not federated_metrics:
            print("⚠️ No metrics available to compare models")
            return None

        # Extract comparable metrics
        centralized_mae = centralized_metrics.get('mae')
        centralized_rmse = centralized_metrics.get('rmse')
        centralized_time = centralized.get('training_time')

        federated_mae = federated_metrics.get('mae')
        federated_rmse = None
        if 'mse' in federated_metrics:
            federated_rmse = math.sqrt(federated_metrics['mse'])
        federated_time = federated.get('training_time')
        federated_bandwidth = None
        bandwidth_details = federated.get('bandwidth_estimate') or {}
        if bandwidth_details:
            federated_bandwidth = bandwidth_details.get('total_bytes_mb')

        preferred = None
        rationale = []

        if centralized_mae is not None and federated_mae is not None:
            if federated_mae < centralized_mae:
                preferred = 'federated'
                improvement = (centralized_mae - federated_mae) / centralized_mae * 100 if centralized_mae else None
                if improvement is not None:
                    rationale.append(f"Federated MAE improved by {improvement:.1f}% compared to centralized")
            elif centralized_mae < federated_mae:
                preferred = 'centralized'
                degradation = (federated_mae - centralized_mae) / federated_mae * 100 if federated_mae else None
                if degradation is not None:
                    rationale.append(f"Centralized MAE is {degradation:.1f}% lower than federated")
            else:
                rationale.append("MAE is equivalent across approaches")

        if preferred is None and centralized_rmse is not None and federated_rmse is not None:
            if federated_rmse < centralized_rmse:
                preferred = 'federated'
                improvement = (centralized_rmse - federated_rmse) / centralized_rmse * 100 if centralized_rmse else None
                if improvement is not None:
                    rationale.append(f"Federated RMSE improved by {improvement:.1f}%")
            elif centralized_rmse < federated_rmse:
                preferred = 'centralized'
                improvement = (federated_rmse - centralized_rmse) / federated_rmse * 100 if federated_rmse else None
                if improvement is not None:
                    rationale.append(f"Centralized RMSE improved by {improvement:.1f}%")

        # If accuracy is tied, prefer faster training for operational simplicity
        if preferred is None and centralized_time is not None and federated_time is not None:
            if centralized_time < federated_time:
                preferred = 'centralized'
                rationale.append("Training time is faster for centralized model")
            elif federated_time < centralized_time:
                preferred = 'federated'
                rationale.append("Training time is faster for federated model")

        # Consider bandwidth savings when accuracy is similar
        if preferred == 'federated' and federated_bandwidth is not None:
            rationale.append(f"Federated training used approximately {federated_bandwidth:.2f} MB of bandwidth")

        selection_summary = {
            'preferred': preferred,
            'centralized': {
                'mae': centralized_mae,
                'rmse': centralized_rmse,
                'training_time': centralized_time,
            },
            'federated': {
                'mae': federated_mae,
                'rmse': federated_rmse,
                'training_time': federated_time,
                'bandwidth_mb': federated_bandwidth,
            },
            'rationale': rationale,
        }

        self.results['model_selection'] = selection_summary

        if preferred:
            print("\n🏆 Preferred Training Approach:")
            print(f"   → {preferred.title()} model is recommended based on current metrics")
            for detail in rationale:
                print(f"     - {detail}")
        else:
            print("\nℹ️ No clear preference detected between centralized and federated models")

        return selection_summary
    
    def generate_final_report(self):
        """Generate final pipeline execution report."""
        total_time = self.end_time - self.start_time

        # Ensure preferred model is determined before reporting
        if not self.results.get('model_selection'):
            self.determine_preferred_training()

        print("\n" + "="*80)
        print("🎯 FEDUHI PIPELINE EXECUTION REPORT")
        print("="*80)
        
        print(f"📅 Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"⏱️ Total Execution Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f"🌱 Random Seed: {self.seed}")
        
        print("\n📊 PIPELINE STEPS COMPLETED:")
        print("-" * 50)
        
        # Data Generation Summary
        if self.results['data_generation']:
            data_results = self.results['data_generation']
            print(f"✅ Data Generation: {data_results['total_samples']} samples across {len(data_results['zones_created'])} zones")
        
        # Centralized Training Summary
        if self.results['centralized_training']:
            central_results = self.results['centralized_training']
            print(f"✅ Centralized Training: {central_results['training_time']:.2f}s, RMSE: {central_results['metrics']['rmse']:.3f}°C")
        
        # Federated Training Summary
        if self.results['federated_training']:
            federated_results = self.results['federated_training']
            bandwidth = federated_results.get('bandwidth_estimate', {})
            print(f"✅ Federated Training: {federated_results['training_time']:.2f}s, {federated_results['rounds']} rounds, {bandwidth.get('total_bytes_mb', 0):.1f}MB bandwidth")
        
        # Visualization Summary
        if self.results['visualization']:
            viz_results = self.results['visualization']
            print(f"✅ Visualization: {len(viz_results['plots_created'])} plots and reports generated")

        # Preferred model summary
        selection = self.results.get('model_selection')
        if selection and selection.get('preferred'):
            print("\n🏆 RECOMMENDED TRAINING APPROACH:")
            print(f"   Preferred: {selection['preferred'].title()} model")
            for detail in selection.get('rationale', []):
                print(f"   - {detail}")
        
        print("\n📁 OUTPUT FILES:")
        print("-" * 50)
        
        # List all generated files
        output_dirs = ['data', 'models', 'results']
        for directory in output_dirs:
            if os.path.exists(directory):
                files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
                if files:
                    print(f"\n📂 {directory.upper()}/")
                    for file in sorted(files):
                        file_path = os.path.join(directory, file)
                        file_size = os.path.getsize(file_path)
                        print(f"   📄 {file} ({file_size:,} bytes)")
        
        print("\n🎉 PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        # Save pipeline results
        pipeline_results = {
            'execution_time': total_time,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'seed': self.seed,
            'results': self.results
        }
        
        with open('results/pipeline_execution_report.pkl', 'wb') as f:
            pickle.dump(pipeline_results, f)
        
        print("💾 Pipeline execution report saved to: results/pipeline_execution_report.pkl")
    
    def show_results_with_charts(self):
        """Show results with interactive charts and graphs."""
        try:
            # Import the results viewer
            from view_results import display_detailed_results, create_results_comparison, create_zone_data_visualization
            
            # Load results
            results = {
                'centralized': None,
                'federated': None,
                'data_available': False
            }
            
            # Load centralized results
            centralized_file = os.path.join(self.results_dir, 'centralized_results.pkl')
            if os.path.exists(centralized_file):
                with open(centralized_file, 'rb') as f:
                    results['centralized'] = pickle.load(f)
            
            # Load federated results
            federated_file = os.path.join(self.results_dir, 'federated_results.pkl')
            if os.path.exists(federated_file):
                with open(federated_file, 'rb') as f:
                    results['federated'] = pickle.load(f)
            
            results['data_available'] = results['centralized'] is not None or results['federated'] is not None
            
            if results['data_available']:
                print("\n🎯 REAL RESULTS ANALYSIS:")
                display_detailed_results(results)
                
                print("\n📊 CREATING COMPARISON CHARTS...")
                create_results_comparison(results)
                
                print("\n🏙️ SHOWING ZONE DATA VISUALIZATION...")
                create_zone_data_visualization()
                
                print("\n✅ All charts and graphs displayed!")
            else:
                print("⚠️ No results available for visualization")
                
        except Exception as e:
            print(f"⚠️ Error displaying results: {e}")
            print("You can manually run: python view_results.py")
    
    def run_reliable_training(self, num_iterations=3):
        """
        Run reliable training with multiple iterations for statistical significance.
        
        Args:
            num_iterations (int): Number of training iterations per model
            
        Returns:
            dict: Reliable training results
        """
        print(f"\n🔬 Running Reliable Training Analysis ({num_iterations} iterations per model)")
        print("=" * 70)
        
        try:
            from reliable_training import ReliableTrainer
            
            # Check if data exists
            if not os.path.exists('data/combined_zone_data.csv'):
                raise FileNotFoundError("Combined data not found. Run data generation first.")
            
            # Initialize reliable trainer
            trainer = ReliableTrainer(num_iterations=num_iterations, seed_base=self.seed)
            
            # Run multiple centralized training
            print(f"\n🧠 Running {num_iterations} centralized training iterations...")
            central_stats = trainer.run_multiple_centralized_training()
            
            # Run multiple federated training  
            print(f"\n🤝 Running {num_iterations} federated training iterations...")
            fed_stats = trainer.run_multiple_federated_training()
            
            # Create reliability plots
            print("\n📊 Creating reliability analysis plots...")
            trainer.create_reliability_plots('results/reliability_analysis.png')
            
            # Print reliability report
            trainer.print_reliability_report()
            
            # Save results
            trainer.save_reliability_results('results')
            
            self.results['reliable_training'] = {
                'centralized_stats': central_stats,
                'federated_stats': fed_stats,
                'trainer_results': trainer.results
            }
            
            print(f"\n✅ Reliable training analysis completed!")
            print(f"📊 Results saved with confidence intervals and statistical analysis")
            
            return self.results['reliable_training']
            
        except Exception as e:
            print(f"❌ Error in reliable training: {e}")
            traceback.print_exc()
            raise
    
    def run_complete_pipeline(self):
        """
        Run the complete FedUHI pipeline from start to finish.
        
        Returns:
            dict: Complete pipeline results
        """
        self.start_time = time.time()
        
        try:
            print(f"\n🚀 Starting FedUHI Pipeline Execution (Seed: {self.seed})")
            print("="*60)
            
            # Execute all pipeline steps
            self.step1_generate_data()
            self.step2_centralized_training() 
            self.step3_federated_training()
            self.step4_visualization()
            
            self.end_time = time.time()
            
            # Generate final report
            self.generate_final_report()
            
            # Show results with charts
            print("\n📊 Displaying results with charts and graphs...")
            self.show_results_with_charts()
            
            return self.results
            
        except Exception as e:
            print(f"\n❌ PIPELINE EXECUTION FAILED: {str(e)}")
            print("="*60)
            traceback.print_exc()
            raise
    
    def run_step(self, step_number: int):
        """
        Run a specific pipeline step.
        
        Args:
            step_number (int): Step number to run (1-4)
            
        Returns:
            dict: Results from the specified step
        """
        steps = {
            1: self.step1_generate_data,
            2: self.step2_centralized_training,
            3: self.step3_federated_training,
            4: self.step4_visualization
        }
        
        if step_number not in steps:
            raise ValueError(f"Invalid step number: {step_number}. Must be 1-4.")
        
        print(f"\n🎯 Running Step {step_number} only...")
        return steps[step_number]()


def main():
    """Main function to run the FedUHI pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='FedUHI Pipeline - Federated Urban Heat Island Simulation')
    parser.add_argument('--step', type=int, choices=[1, 2, 3, 4], 
                       help='Run specific step only (1: Data, 2: Centralized, 3: Federated, 4: Visualization)')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick version with fewer epochs/rounds')
    parser.add_argument('--reliable', action='store_true',
                       help='Run multiple training iterations for reliable results')
    parser.add_argument('--iterations', type=int, default=3,
                       help='Number of training iterations for reliable mode (default: 3)')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FedUHIPipeline(seed=args.seed)
    
    try:
        if args.reliable:
            # Run reliable training with multiple iterations
            result = pipeline.run_reliable_training(args.iterations)
            print(f"\n🎉 Reliable training with {args.iterations} iterations completed successfully!")
        elif args.step:
            # Run specific step
            result = pipeline.run_step(args.step)
            print(f"\n✅ Step {args.step} completed successfully!")
        else:
            # Run complete pipeline
            result = pipeline.run_complete_pipeline()
            print("\n🎉 Complete pipeline executed successfully!")
            
    except KeyboardInterrupt:
        print("\n⚠️ Pipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
