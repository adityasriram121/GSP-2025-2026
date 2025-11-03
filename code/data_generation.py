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
                'base_temp': 28.0,
                'temp_amplitude': 8.0,
                'base_humidity': 45.0,
                'humidity_amplitude': 15.0,
                'description': 'Sunny rooftop - high temperature range, low humidity'
            },
            'Zone_B_Street': {
                'base_temp': 24.0,
                'temp_amplitude': 5.0,
                'base_humidity': 60.0,
                'humidity_amplitude': 10.0,
                'description': 'Shaded street - moderate temps, stable humidity'
            },
            'Zone_C_Park': {
                'base_temp': 22.0,
                'temp_amplitude': 4.0,
                'base_humidity': 75.0,
                'humidity_amplitude': 12.0,
                'description': 'Urban park - cool, humid microclimate'
            },
            'Zone_D_Parking': {
                'base_temp': 32.0,
                'temp_amplitude': 10.0,
                'base_humidity': 35.0,
                'humidity_amplitude': 20.0,
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
            
        Raises:
            ValueError: If parameters are invalid
        """
        if not isinstance(zone_name, str):
            raise ValueError("Zone name must be a string")
        if zone_name not in self.zones:
            raise ValueError(f"Unknown zone: {zone_name}. Available zones: {list(self.zones.keys())}")
            
        if not isinstance(days, int):
            raise ValueError("Days must be an integer")
        if days <= 0:
            raise ValueError("Days must be positive")
        if days > 365:
            raise ValueError("Days cannot exceed 365")
            
        if not isinstance(samples_per_hour, int):
            raise ValueError("Samples per hour must be an integer")
        if samples_per_hour <= 0:
            raise ValueError("Samples per hour must be positive")
        if samples_per_hour > 60:
            raise ValueError("Samples per hour cannot exceed 60")
            
        zone_config = self.zones[zone_name]
        
        total_samples = days * 24 * samples_per_hour
        start_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        timestamps = [start_time + timedelta(hours=i/samples_per_hour) for i in range(total_samples)]
        
        hours = np.array([t.hour + t.minute/60.0 for t in timestamps])
        
        temp_cycle = np.sin(2 * np.pi * (hours - 6) / 24)
        temp_noise = np.random.normal(0, 1.5, total_samples)
        
        temperature = (zone_config['base_temp'] + 
                      zone_config['temp_amplitude'] * temp_cycle + 
                      temp_noise)
        
        humidity_cycle = -0.3 * temp_cycle
        humidity_noise = np.random.normal(0, 2.0, total_samples)
        
        humidity = (zone_config['base_humidity'] + 
                   zone_config['humidity_amplitude'] * humidity_cycle + 
                   humidity_noise)
        
        humidity = np.clip(humidity, 20, 95)
        
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
    
    def save_data_to_csv(self, data_dict, output_dir=None):
        """
        Save all zone data to CSV files.
        
        Args:
            data_dict (dict): Dictionary of zone data
            output_dir (str): Directory to save CSV files
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        for zone_name, data in data_dict.items():
            filename = f"{output_dir or '.'}/{zone_name.lower()}_data.csv"
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
        
        for i, (zone_name, data) in enumerate(data_dict.items()):
            sample_data = data.iloc[::4]
            ax1.plot(sample_data['timestamp'], sample_data['temperature'], 
                    color=colors[i], label=zone_name.replace('_', ' '), linewidth=2)
        
        ax1.set_title('Temperature Comparison Across Urban Zones', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Temperature (°C)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
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



