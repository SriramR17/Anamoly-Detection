"""
Real-Time Network Usage Monitor
==============================
Monitors your system's network usage and generates data similar to cellular network metrics.
This simulates real-time network monitoring for testing the anomaly detection system.
"""

import psutil
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
import json
from collections import deque
import random

class NetworkMonitor:
    """
    Real-time network usage monitor that collects and formats data
    similar to cellular network metrics for testing purposes.
    """
    
    def __init__(self, collection_interval=5):
        """
        Initialize the network monitor.
        
        Args:
            collection_interval: Seconds between data collection points
        """
        self.collection_interval = collection_interval
        self.data_buffer = deque(maxlen=1000)  # Keep last 1000 readings
        self.is_monitoring = False
        self.monitor_thread = None
        
        # Simulated cell names (like in your training data)
        self.cell_names = [
            '1ALTE', '2BLTE', '3CLTE', '4ALTE', '5BLTE', 
            '6ULTE', '7ALTE', '8BLTE', '9ALTE', '10CLTE'
        ]
        
        # Initialize baseline metrics
        self._initialize_baseline()
    
    def _initialize_baseline(self):
        """Initialize baseline network metrics."""
        print("Initializing network baseline...")
        
        # Collect initial network stats
        net_io = psutil.net_io_counters()
        self.baseline = {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
            'timestamp': time.time()
        }
        
        print("✓ Baseline established")
    
    def get_current_network_stats(self):
        """Get current network statistics from the system."""
        try:
            # Network I/O statistics
            net_io = psutil.net_io_counters()
            
            # Network connections
            connections = psutil.net_connections()
            active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])
            
            # CPU and Memory (affects network performance)
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            current_time = time.time()
            time_diff = current_time - self.baseline['timestamp']
            
            # Calculate rates (bytes per second)
            if time_diff > 0:
                bytes_sent_rate = (net_io.bytes_sent - self.baseline['bytes_sent']) / time_diff
                bytes_recv_rate = (net_io.bytes_recv - self.baseline['bytes_recv']) / time_diff
                packets_sent_rate = (net_io.packets_sent - self.baseline['packets_sent']) / time_diff
                packets_recv_rate = (net_io.packets_recv - self.baseline['packets_recv']) / time_diff
            else:
                bytes_sent_rate = bytes_recv_rate = packets_sent_rate = packets_recv_rate = 0
            
            # Update baseline for next calculation
            self.baseline = {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv,
                'timestamp': current_time
            }
            
            return {
                'timestamp': datetime.now(),
                'bytes_sent_rate': max(0, bytes_sent_rate),
                'bytes_recv_rate': max(0, bytes_recv_rate),
                'packets_sent_rate': max(0, packets_sent_rate),
                'packets_recv_rate': max(0, packets_recv_rate),
                'active_connections': active_connections,
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'total_bytes': net_io.bytes_sent + net_io.bytes_recv,
                'total_packets': net_io.packets_sent + net_io.packets_recv
            }
            
        except Exception as e:
            print(f"Error collecting network stats: {e}")
            return None
    
    def convert_to_cellular_format(self, net_stats):
        """
        Convert system network stats to a format similar to cellular network data,
        using a more deterministic mapping.
        """
        if not net_stats:
            return None
        
        # Randomly assign to a cell, but keep the core logic non-random
        cell_name = random.choice(self.cell_names)
        
        # Calculate total throughput
        total_throughput_kbps = (net_stats['bytes_sent_rate'] + net_stats['bytes_recv_rate']) / 1024
        
        # --- PRB Usage (Less random, more proportional) ---
        # Max PRB usage is 100%. Let's assume a linear relationship where
        # 10,000 KB/s (10 MB/s) corresponds to 50% PRB usage.
        # We add a small amount of Gaussian noise instead of large random variations.
        prb_ul = min(100, (net_stats['bytes_sent_rate'] / 1024) / 200 + np.random.normal(2, 0.5))
        prb_dl = min(100, (net_stats['bytes_recv_rate'] / 1024) / 200 + np.random.normal(5, 1))

        # --- Throughput metrics (Direct conversion) ---
        mean_thr_ul = (net_stats['bytes_sent_rate'] * 8) / (1024 * 1024)  # Mbps
        mean_thr_dl = (net_stats['bytes_recv_rate'] * 8) / (1024 * 1024)  # Mbps
        
        # Max throughput is now just a scaling factor of mean throughput, not a large random number.
        max_thr_ul = mean_thr_ul * 1.5  
        max_thr_dl = mean_thr_dl * 1.5 
        
        # --- User Equipment (UE) metrics (Directly tied to active connections) ---
        # The number of connections is the key indicator. Add a small constant to avoid zero.
        base_users = max(1, net_stats['active_connections'])
        mean_ue_ul = base_users * 0.9  
        mean_ue_dl = base_users * 1.1
        max_ue_ul = base_users * 1.5
        max_ue_dl = base_users * 2.0
        
        # Create cellular network format data
        cellular_data = {
            'Time': net_stats['timestamp'].strftime('%H:%M'),
            'CellName': cell_name,
            'PRBUsageUL': round(prb_ul, 2),
            'PRBUsageDL': round(prb_dl, 2),
            'meanThr_DL': round(mean_thr_dl, 3),
            'meanThr_UL': round(mean_thr_ul, 3),
            'maxThr_DL': round(max_thr_dl, 3),
            'maxThr_UL': round(max_thr_ul, 3),
            'meanUE_DL': round(mean_ue_dl, 1),
            'meanUE_UL': round(mean_ue_ul, 1),
            'maxUE_DL': round(max_ue_dl, 1),
            'maxUE_UL': round(max_ue_ul, 1),
            'maxUE_UL+DL': round(max_ue_ul + max_ue_dl, 1),
            'system_cpu': net_stats['cpu_percent'],
            'system_memory': net_stats['memory_percent'],
            'raw_throughput_kbps': total_throughput_kbps
        }
        
        return cellular_data
    
    def _monitor_loop(self):
        """Main monitoring loop (runs in separate thread)."""
        print(f"🔍 Starting real-time monitoring (interval: {self.collection_interval}s)")
        
        while self.is_monitoring:
            try:
                # Get current network stats
                net_stats = self.get_current_network_stats()
                
                if net_stats:
                    # Convert to cellular format
                    cellular_data = self.convert_to_cellular_format(net_stats)
                    
                    if cellular_data:
                        # Add to buffer
                        self.data_buffer.append(cellular_data)
                        
                        # Print current stats
                        print(f"📊 {cellular_data['Time']} | {cellular_data['CellName']} | "
                              f"UL: {cellular_data['PRBUsageUL']:.1f}% | "
                              f"DL: {cellular_data['PRBUsageDL']:.1f}% | "
                              f"Thr: {cellular_data['meanThr_DL']:.2f} Mbps")
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                print(f"Error in monitoring loop: {e}")
                time.sleep(self.collection_interval)
    
    def start_monitoring(self):
        """Start real-time network monitoring."""
        if self.is_monitoring:
            print("⚠️ Monitoring is already active")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("✅ Network monitoring started")
    
    def stop_monitoring(self):
        """Stop network monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        print("🛑 Network monitoring stopped")
    
    def get_recent_data(self, minutes=10):
        """
        Get recent network data as DataFrame.
        
        Args:
            minutes: Number of minutes of recent data to return
            
        Returns:
            pandas.DataFrame: Recent network data
        """
        if not self.data_buffer:
            return pd.DataFrame()
        
        # Convert buffer to DataFrame
        df = pd.DataFrame(list(self.data_buffer))
        
        if df.empty:
            return df
        
        # Filter by time if requested
        if minutes > 0:
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            df['timestamp'] = pd.to_datetime(df['Time'], format='%H:%M')
            df['timestamp'] = df['timestamp'].apply(
                lambda x: x.replace(year=datetime.now().year, 
                                  month=datetime.now().month, 
                                  day=datetime.now().day)
            )
            df = df[df['timestamp'] >= cutoff_time]
        
        return df.drop('timestamp', axis=1, errors='ignore')
    
    def simulate_network_load(self, duration_seconds=30):
        """
        Generate artificial network load for testing.
        
        Args:
            duration_seconds: How long to generate load
        """
        print(f"🚀 Generating network load for {duration_seconds} seconds...")
        
        import requests
        import concurrent.futures
        
        def make_requests():
            try:
                # Make multiple HTTP requests to generate network traffic
                urls = [
                    'https://httpbin.org/bytes/1000',
                    'https://jsonplaceholder.typicode.com/posts',
                    'https://api.github.com/users/octocat',
                ]
                
                for url in urls:
                    response = requests.get(url, timeout=5)
                    time.sleep(0.1)
            except:
                pass  # Ignore errors, we just want to generate traffic
        
        # Run multiple threads to generate load
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            end_time = time.time() + duration_seconds
            
            while time.time() < end_time:
                future = executor.submit(make_requests)
                futures.append(future)
                time.sleep(0.5)
            
            # Wait for completion
            concurrent.futures.wait(futures, timeout=duration_seconds + 10)
        
        print("✅ Network load generation complete")
    
    def save_data_to_csv(self, filename=None, minutes=60):
        """
        Save recent network data to CSV file.
        
        Args:
            filename: Output filename (optional)
            minutes: Minutes of data to save
        """
        if filename is None:
            filename = f"network_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = self.get_recent_data(minutes=minutes)
        
        if df.empty:
            print("No data to save")
            return
        
        filepath = f"results/{filename}"
        df.to_csv(filepath, index=False)
        print(f"💾 Saved {len(df)} data points to {filepath}")
        
        return filepath


def main():
    """Demo of real-time network monitoring."""
    monitor = NetworkMonitor(collection_interval=2)  # Collect every 2 seconds
    
    try:
        print("="*60)
        print("REAL-TIME NETWORK MONITORING FOR ANOMALY DETECTION")
        print("="*60)
        print("This will monitor your system's network usage and convert it")
        print("to cellular network format for testing your anomaly detection model.")
        print()
        
        # Start monitoring
        monitor.start_monitoring()
        
        print("\n🔧 Options:")
        print("  1. Let it run to collect normal usage data")
        print("  2. Open web browsers, download files, stream videos to generate load")
        print("  3. Press Ctrl+C to stop and save data")
        print()
        
        # Optional: Generate some artificial load
        generate_load = input("Generate artificial network load? (y/n): ").lower()
        if generate_load == "y":
            monitor.simulate_network_load(duration_seconds=30)
        
            print("📈 Monitoring... Press Ctrl+C to stop")
            
            # Keep monitoring until interrupted
            while True:
                time.sleep(5)
                
                # Show stats every 30 seconds
                if len(monitor.data_buffer) % 15 == 0 and len(monitor.data_buffer) > 0:
                    recent_data = monitor.get_recent_data(minutes=5)
                    if not recent_data.empty:
                        avg_prb_ul = recent_data['PRBUsageUL'].mean()
                        avg_prb_dl = recent_data['PRBUsageDL'].mean()
                        print(f"\n📊 5-min averages: UL={avg_prb_ul:.1f}%, DL={avg_prb_dl:.1f}%")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Stopping monitoring...")
        monitor.stop_monitoring()
        
        # Save collected data
        if monitor.data_buffer:
            filepath = monitor.save_data_to_csv()
            print(filepath)
            print(f"💾 Data saved to: {filepath}")
            
            # Show summary
            df = monitor.get_recent_data()
            print(f"\n📈 Collection Summary:")
            print(f"  Total samples: {len(df)}")
            print(f"  Time range: {df['Time'].min()} - {df['Time'].max()}")
            print(f"  Cells monitored: {df['CellName'].nunique()}")
            print(f"  Avg PRB Usage UL: {df['PRBUsageUL'].mean():.1f}%")
            print(f"  Avg PRB Usage DL: {df['PRBUsageDL'].mean():.1f}%")
            print()
            print("🔮 You can now use this data to test your anomaly detection model!")
        
        else:
            print("No data collected")


if __name__ == "__main__":
    main()
