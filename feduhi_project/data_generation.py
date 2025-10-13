"""
Federated Urban Heat Island (FedUHI) - Data Generation Module

This module generates synthetic sensor data for four different urban zones:
- Zone A: Sunny rooftop (high temp, low humidity)
- Zone B: Shaded street (moderate temp, stable humidity)  
- Zone C: Urban park (cool, humid)
- Zone D: Asphalt parking lot (hot, dry)

Each zone generates temperature and humidity data with realistic patterns
using sinusoidal functions plus noise to simulate real sensor data.
"""

import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


class ZoneDataGenerator:
    """Generates synthetic sensor data for different urban zones."""
    
    def __init__(self, seed=42):
        """Initialize the data generator with random seed for reproducibility."""
        np.random.seed(seed)
        self.zones = {
            'Zone_A_Rooftop': {
                'base_temp': 28.0,  # Higher baseline temperature
                'temp_amplitude': 8.0,  # Large daily temperature variation
                'base_humidity': 45.0,  # Lower humidity
                'humidity_amplitude': 15.0,  # Moderate humidity variation
                'description': 'Sunny rooftop - high temperature range, low humidity'
            },
            'Zone_B_Street': {
                'base_temp': 24.0,  # Moderate baseline temperature
                'temp_amplitude': 5.0,  # Smaller temperature variation
                'base_humidity': 60.0,  # Moderate humidity
                'humidity_amplitude': 10.0,  # Stable humidity
                'description': 'Shaded street - moderate temps, stable humidity'
            },
            'Zone_C_Park': {
                'base_temp': 22.0,  # Cooler baseline temperature
                'temp_amplitude': 4.0,  # Smallest temperature variation
                'base_humidity': 75.0,  # High humidity
                'humidity_amplitude': 12.0,  # Moderate humidity variation
                'description': 'Urban park - cool, humid microclimate'
            },
            'Zone_D_Parking': {
                'base_temp': 32.0,  # Highest baseline temperature
                'temp_amplitude': 10.0,  # Largest temperature variation
                'base_humidity': 35.0,  # Lowest humidity
                'humidity_amplitude': 20.0,  # High humidity variation
                'description': 'Asphalt parking lot - hot, dry, high heat absorption'
            }
        }
    
    def generate_zone_data(self, zone_name, days=7, samples_per_hour=4):
        """
        Generate synthetic sensor data for a specific zone.
        
        Args:
            zone_name (str): Name of the zone (e.g., 'Zone_A_Rooftop')
            days (int): Number of days to generate data for
            samples_per_hour (int): Number of samples per hour
            
        Returns:
            pd.DataFrame: DataFrame with timestamp, temperature, humidity columns
        """
        if zone_name not in self.zones:
            raise ValueError(f"Unknown zone: {zone_name}")
        
        zone_config = self.zones[zone_name]
        
        # Generate timestamps
        total_samples = days * 24 * samples_per_hour
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        timestamps = [start_time + timedelta(hours=i/samples_per_hour) for i in range(total_samples)]
        
        # Generate temperature data with sinusoidal pattern + noise
        hours = np.array([t.hour + t.minute/60.0 for t in timestamps])
        
        # Daily temperature cycle (peak at 2 PM, minimum at 6 AM)
        temp_cycle = np.sin(2 * np.pi * (hours - 6) / 24)
        temp_noise = np.random.normal(0, 1.5, total_samples)  # Random noise
        
        temperature = (zone_config['base_temp'] + 
                      zone_config['temp_amplitude'] * temp_cycle + 
                      temp_noise)
        
        # Generate humidity data (inverse relationship with temperature + noise)
        humidity_cycle = -0.3 * temp_cycle  # Humidity decreases as temp increases
        humidity_noise = np.random.normal(0, 2.0, total_samples)
        
        humidity = (zone_config['base_humidity'] + 
                   zone_config['humidity_amplitude'] * humidity_cycle + 
                   humidity_noise)
        
        # Ensure humidity stays within reasonable bounds (20-95%)
        humidity = np.clip(humidity, 20, 95)
        
        # Create DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'temperature': temperature,
            'humidity': humidity,
            'zone': zone_name
        })
        
        return data
    
    def generate_all_zones_data(self, days=7, samples_per_hour=4):
        """
        Generate synthetic data for all four zones.
        
        Args:
            days (int): Number of days to generate data for
            samples_per_hour (int): Number of samples per hour
            
        Returns:
            dict: Dictionary with zone names as keys and DataFrames as values
        """
        all_data = {}
        
        for zone_name in self.zones.keys():
            print(f"Generating data for {zone_name}...")
            data = self.generate_zone_data(zone_name, days, samples_per_hour)
            all_data[zone_name] = data
            
        return all_data
    
    def save_data_to_csv(self, data_dict, output_dir='data'):
        """
        Save all zone data to CSV files.
        
        Args:
            data_dict (dict): Dictionary of zone data
            output_dir (str): Directory to save CSV files
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for zone_name, data in data_dict.items():
            filename = f"{output_dir}/{zone_name.lower()}_data.csv"
            data.to_csv(filename, index=False)
            print(f"Saved {zone_name} data to {filename}")
    
    def create_combined_dataset(self, data_dict):
        """
        Create a combined dataset from all zones for centralized training.
        
        Args:
            data_dict (dict): Dictionary of zone data
            
        Returns:
            pd.DataFrame: Combined dataset with all zones
        """
        combined_data = pd.concat(data_dict.values(), ignore_index=True)
        return combined_data
    
    def plot_zone_comparison(self, data_dict, save_path=None):
        """
        Create visualization comparing temperature and humidity across zones.
        
        Args:
            data_dict (dict): Dictionary of zone data
            save_path (str): Optional path to save the plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        colors = ['red', 'blue', 'green', 'orange']
        
        # Plot temperature
        for i, (zone_name, data) in enumerate(data_dict.items()):
            # Sample every 4th point for cleaner visualization
            sample_data = data.iloc[::4]
            ax1.plot(sample_data['timestamp'], sample_data['temperature'], 
                    color=colors[i], label=zone_name.replace('_', ' '), linewidth=2)
        
        ax1.set_title('Temperature Comparison Across Urban Zones', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Temperature (°C)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot humidity
        for i, (zone_name, data) in enumerate(data_dict.items()):
            sample_data = data.iloc[::4]
            ax2.plot(sample_data['timestamp'], sample_data['humidity'], 
                    color=colors[i], label=zone_name.replace('_', ' '), linewidth=2)
        
        ax2.set_title('Humidity Comparison Across Urban Zones', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_ylabel('Humidity (%)', fontsize=12)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def get_data_statistics(self, data_dict):
        """
        Calculate and display statistics for each zone's data.
        
        Args:
            data_dict (dict): Dictionary of zone data
            
        Returns:
            dict: Statistics for each zone
        """
        stats = {}
        
        print("\n" + "="*60)
        print("ZONE DATA STATISTICS")
        print("="*60)
        
        for zone_name, data in data_dict.items():
            temp_stats = data['temperature'].describe()
            humidity_stats = data['humidity'].describe()
            
            stats[zone_name] = {
                'temperature': temp_stats,
                'humidity': humidity_stats,
                'sample_count': len(data)
            }
            
            print(f"\n{zone_name.replace('_', ' ')}:")
            print(f"  Description: {self.zones[zone_name]['description']}")
            print(f"  Sample count: {len(data)}")
            print(f"  Temperature - Mean: {temp_stats['mean']:.2f}°C, Std: {temp_stats['std']:.2f}°C")
            print(f"  Humidity - Mean: {humidity_stats['mean']:.2f}%, Std: {humidity_stats['std']:.2f}%")
        
        return stats


def main():
    """Main function to demonstrate data generation."""
    print("Federated Urban Heat Island (FedUHI) - Data Generation")
    print("="*55)
    
    # Initialize data generator
    generator = ZoneDataGenerator(seed=42)
    
    # Generate data for all zones (7 days, 4 samples per hour)
    print("\nGenerating synthetic sensor data for all zones...")
    all_data = generator.generate_all_zones_data(days=7, samples_per_hour=4)
    
    # Save data to CSV files
    print("\nSaving data to CSV files...")
    generator.save_data_to_csv(all_data, output_dir='data')
    
    # Create combined dataset
    combined_data = generator.create_combined_dataset(all_data)
    combined_data.to_csv('data/combined_zone_data.csv', index=False)
    print("Saved combined dataset to data/combined_zone_data.csv")
    
    # Display statistics
    stats = generator.get_data_statistics(all_data)
    
    # Create visualization
    print("\nCreating zone comparison visualization...")
    os.makedirs('results', exist_ok=True)
    generator.plot_zone_comparison(all_data, save_path='results/zone_comparison.png')
    
    print(f"\nData generation complete!")
    print(f"Total samples generated: {len(combined_data)}")
    print(f"Files saved in 'data/' directory")
    print(f"Visualization saved in 'results/' directory")


if __name__ == "__main__":
    main()
