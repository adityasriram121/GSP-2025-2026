import os
import sys
import time
import pickle
import traceback
import subprocess
from datetime import datetime

# Ensure output is flushed immediately (important for Windows)
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

# Add repository root and code directory to Python path
paths_to_add = [REPO_ROOT, CURRENT_DIR]
for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)



# Import project modules (canonical `code` package)
try:
    from code.data_generation import ZoneDataGenerator
    from code.centralized_training import train_centralized_model
    from code.federated_training import prepare_federated_data, run_federated_simulation, estimate_bandwidth_usage
    from code.metrics_visualization import FedUHIVisualizer
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
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
            'visualization': {}
        }
        
        # Create necessary directories
        self._create_directories()
        
        print("  Federated Urban Heat Island (FedUHI) Pipeline")
        print("=" * 60)
        print("A comprehensive simulation of federated learning for urban temperature prediction")
        print("=" * 60)
    
    def _create_directories(self):
        """Create necessary directories for the pipeline."""
        directories = ['data', 'models', 'results']
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"[OK] Created directory: {directory}")
    
    def _log_step(self, step_name: str, start_time: float = None):
        """
        Log pipeline step execution.
        
        Args:
            step_name (str): Name of the pipeline step
            start_time (float): Start time of the step
        """
        if start_time:
            elapsed = time.time() - start_time
            print(f"[TIME]  {step_name} completed in {elapsed:.2f} seconds")
        else:
            print(f"\n[RUN] Starting: {step_name}")
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
            print("[PLOT] Generating synthetic sensor data...")
            all_data = generator.generate_all_zones_data(days=7, samples_per_hour=4)
            
            # Save individual zone data
            print(" Saving zone-specific data...")
            generator.save_data_to_csv(all_data, output_dir='data')
            
            # Create and save combined dataset
            print("[LINK] Creating combined dataset...")
            combined_data = generator.create_combined_dataset(all_data)
            combined_data.to_csv('data/combined_zone_data.csv', index=False)
            
            # Generate statistics
            stats = generator.get_data_statistics(all_data)
            
            # Create initial visualization
            print("[CHART] Creating data visualization...")
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
            print(f"[ERROR] Error in data generation: {str(e)}")
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
            
            print("[ML] Training centralized model...")
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
            print(f"[ERROR] Error in centralized training: {str(e)}")
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
            
            print(" Preparing federated data...")
            training_data, test_data = prepare_federated_data(data_dir)
            
            print("[PLOT] Estimating bandwidth usage...")
            bandwidth_estimate = estimate_bandwidth_usage(training_data, rounds=10)
            
            print("[LOOP] Running federated simulation...")
            federated_results = run_federated_simulation(training_data, test_data, rounds=10)
            
            # Combine federated results
            federated_results['bandwidth_estimate'] = bandwidth_estimate
            
            # Save federated results
            with open('results/federated_results.pkl', 'wb') as f:
                pickle.dump({
                    'results': federated_results,
                    'bandwidth_estimate': bandwidth_estimate
                }, f)
            
            self.results['federated_training'] = federated_results
            
            self._log_step("Step 3: Federated Training", step_start)
            return federated_results
            
        except Exception as e:
            print(f"[ERROR] Error in federated training: {str(e)}")
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
            print("[PLOT] Initializing visualizer...")
            visualizer = FedUHIVisualizer(results_dir='results')
            
            print(" Loading results...")
            visualizer.load_results()
            
            if not visualizer.centralized_results or not visualizer.federated_results:
                raise ValueError("Missing results files for visualization")
            
            print("[CHART] Generating comparison summary...")
            summary_df = visualizer.create_comparison_summary('results/comparison_summary.csv')
            
            print("[PLOT] Creating accuracy comparison plot...")
            visualizer.plot_accuracy_comparison('results/accuracy_comparison.png')
            
            print("[TIME] Creating training analysis plot...")
            visualizer.plot_training_analysis('results/training_analysis.png')
            
            print(" Creating zone-specific analysis...")
            visualizer.plot_zone_specific_analysis('results/zone_analysis.png')
            
            print(" Creating comprehensive report...")
            visualizer.create_comprehensive_report('results/comprehensive_report.png')
            
            print("[REPORT] Generating console report...")
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
            print(f"[ERROR] Error in visualization: {str(e)}")
            traceback.print_exc()
            raise
    
    def generate_final_report(self):
        """Generate final pipeline execution report."""
        total_time = self.end_time - self.start_time
        
        print("\n" + "="*80)
        print("[REPORT] FEDUHI PIPELINE EXECUTION REPORT")
        print("="*80)
        
        print(f" Execution Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"[TIME] Total Execution Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
        print(f" Random Seed: {self.seed}")
        
        print("\n[PLOT] PIPELINE STEPS COMPLETED:")
        print("-" * 50)
        
        # Data Generation Summary
        if self.results['data_generation']:
            data_results = self.results['data_generation']
            print(f"[OK] Data Generation: {data_results['total_samples']} samples across {len(data_results['zones_created'])} zones")
        
        # Centralized Training Summary
        if self.results['centralized_training']:
            central_results = self.results['centralized_training']
            print(f"[OK] Centralized Training: {central_results['training_time']:.2f}s, RMSE: {central_results['metrics']['rmse']:.3f}C")
        
        # Federated Training Summary
        if self.results['federated_training']:
            federated_results = self.results['federated_training']
            bandwidth = federated_results.get('bandwidth_estimate', {})
            print(f"[OK] Federated Training: {federated_results['training_time']:.2f}s, {federated_results['rounds']} rounds, {bandwidth.get('total_bytes_mb', 0):.1f}MB bandwidth")
        
        # Visualization Summary
        if self.results['visualization']:
            viz_results = self.results['visualization']
            print(f"[OK] Visualization: {len(viz_results['plots_created'])} plots and reports generated")
        
        print("\n OUTPUT FILES:")
        print("-" * 50)
        
        # List all generated files
        output_dirs = ['data', 'models', 'results']
        for directory in output_dirs:
            if os.path.exists(directory):
                files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
                if files:
                    print(f"\n {directory.upper()}/")
                    for file in sorted(files):
                        file_path = os.path.join(directory, file)
                        file_size = os.path.getsize(file_path)
                        print(f"    {file} ({file_size:,} bytes)")
        
        print("\n PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
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
        
        print(" Pipeline execution report saved to: results/pipeline_execution_report.pkl")
    
    def show_results_with_charts(self):
        """Show results with interactive charts and graphs."""
        try:
            # Load results and use the built-in visualizer
            visualizer = FedUHIVisualizer(results_dir='results')
            visualizer.load_results()
            
            if visualizer.centralized_results and visualizer.federated_results:
                print("\n[REPORT] REAL RESULTS ANALYSIS:")
                visualizer.generate_console_report()
                
                print("\n[PLOT] CREATING COMPARISON CHARTS...")
                visualizer.plot_accuracy_comparison('results/accuracy_comparison.png')
                visualizer.plot_training_analysis('results/training_analysis.png')
                visualizer.plot_zone_specific_analysis('results/zone_analysis.png')
                
                print("\n SHOWING ZONE DATA VISUALIZATION...")
                visualizer.plot_zone_data_from_files('data', 'results/zone_data_visualization.png')
                
                print("\n[OK] All charts and graphs displayed!")
            else:
                print("[WARN] No results available for visualization")
                
        except Exception as e:
            print(f"[WARN] Error displaying results: {e}")
            import traceback
            traceback.print_exc()
    
    def run_comprehensive_trials(self, num_iterations=10):
        """
        Run reliable training with multiple iterations for statistical significance.
        
        Args:
            num_iterations (int): Number of training iterations per model
            
        Returns:
            dict: Reliable training results
        """
        print(f"\n Running Reliable Training Analysis ({num_iterations} iterations per model)")
        print("=" * 70)
        
        try:
            # Use comprehensive_trials module instead
            from code.comprehensive_trials import ComprehensiveTrials
            
            # Check if data exists
            if not os.path.exists('data/combined_zone_data.csv'):
                print("[INFO] Data not found, running data generation step...")
                self.step1_generate_data()
            
            # Initialize comprehensive trials
            trials_system = ComprehensiveTrials(
                num_trials=num_iterations,
                seed_base=self.seed,
                results_dir='results',
                models_dir='models'
            )
            
            # Run all trials
            trials_system.run_all_trials(
                data_path='data/combined_zone_data.csv',
                data_dir='data'
            )
            
            # Extract results
            centralized_avg = trials_system._calculate_averages(trials_system.centralized_trials, 'centralized')
            federated_avg = trials_system._calculate_averages(trials_system.federated_trials, 'federated')
            
            self.results['reliable_training'] = {
                'centralized_stats': centralized_avg,
                'federated_stats': federated_avg,
                'trainer_results': {
                    'centralized_trials': trials_system.centralized_trials,
                    'federated_trials': trials_system.federated_trials
                }
            }
            
            print(f"\n[OK] Reliable training analysis completed!")
            print(f"[PLOT] Results saved with confidence intervals and statistical analysis")
            
            return self.results['reliable_training']
            
        except Exception as e:
            print(f"[ERROR] Error in reliable training: {e}")
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
            print(f"\n[RUN] Starting FedUHI Pipeline Execution (Seed: {self.seed})")
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
            print("\n[PLOT] Displaying results with charts and graphs...")
            self.show_results_with_charts()
            
            return self.results
            
        except Exception as e:
            print(f"\n[ERROR] PIPELINE EXECUTION FAILED: {str(e)}")
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
        
        print(f"\n[REPORT] Running Step {step_number} only...")
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
    parser.add_argument('--iterations', type=int, default=10,
                       help='Number of training iterations for comprehensive trials (default: 10)')
    parser.add_argument('--skip-install', action='store_true',
                       help='Skip automatic requirements installation check')
    parser.add_argument('--single-run', action='store_true',
                       help='Run a single complete pipeline instead of comprehensive trials')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = FedUHIPipeline(seed=args.seed)
    
    try:
        if args.step:
            # Run specific step
            result = pipeline.run_step(args.step)
            print(f"\n[OK] Step {args.step} completed successfully!")
        elif args.single_run:
            # Run single complete pipeline
            result = pipeline.run_complete_pipeline()
            print("\n Complete pipeline executed successfully!")
        else:
            # Default: Run comprehensive trials with multiple iterations
            result = pipeline.run_comprehensive_trials(args.iterations)
            print(f"\n Comprehensive trials with {args.iterations} iterations completed successfully!")
            
    except KeyboardInterrupt:
        print("\n[WARN] Pipeline execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Pipeline execution failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()