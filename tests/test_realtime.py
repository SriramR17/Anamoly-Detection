"""
Real-Time Network Testing Demo
=============================
Simple demo showing different ways to test your network anomaly detection
system with real-time data from your computer.
"""

import time
from network_monitor import NetworkMonitor
from realtime_detector import RealTimeAnomalyDetector

def test_network_monitor():
    """Test 1: Basic network monitoring."""
    print("="*60)
    print("TEST 1: BASIC NETWORK MONITORING")
    print("="*60)
    print("This will monitor your system's network for 30 seconds")
    print("and convert it to cellular network format.")
    print()
    
    monitor = NetworkMonitor(collection_interval=3)
    
    try:
        monitor.start_monitoring()
        print("📱 Monitoring network... (30 seconds)")
        print("💡 Try browsing websites, streaming videos, or downloading files!")
        print()
        
        time.sleep(30)
        
        # Get collected data
        data = monitor.get_recent_data(minutes=10)
        print(f"\n📊 Results:")
        print(f"   Samples collected: {len(data)}")
        if not data.empty:
            print(f"   Average PRB Usage UL: {data['PRBUsageUL'].mean():.1f}%")
            print(f"   Average PRB Usage DL: {data['PRBUsageDL'].mean():.1f}%")
            print(f"   Peak throughput: {data['meanThr_DL'].max():.3f} Mbps")
        
        # Save data
        filename = monitor.save_data_to_csv(minutes=10)
        print(f"   Data saved to: {filename}")
        
    finally:
        monitor.stop_monitoring()
    
    input("\nPress Enter to continue to next test...")


def test_load_generation():
    """Test 2: Network load generation."""
    print("\n" + "="*60)
    print("TEST 2: ARTIFICIAL NETWORK LOAD GENERATION")
    print("="*60)
    print("This will generate artificial network traffic and monitor it.")
    print()
    
    monitor = NetworkMonitor(collection_interval=2)
    
    try:
        monitor.start_monitoring()
        
        # Wait for baseline
        print("📈 Collecting baseline data (10 seconds)...")
        time.sleep(10)
        
        # Generate load
        print("🚀 Generating artificial network load...")
        monitor.simulate_network_load(duration_seconds=20)
        
        # Wait for more data
        print("📉 Monitoring post-load data (10 seconds)...")
        time.sleep(10)
        
        # Show results
        data = monitor.get_recent_data(minutes=5)
        print(f"\n📊 Load Test Results:")
        print(f"   Total samples: {len(data)}")
        if not data.empty:
            print(f"   Max PRB Usage UL: {data['PRBUsageUL'].max():.1f}%")
            print(f"   Max PRB Usage DL: {data['PRBUsageDL'].max():.1f}%")
            print(f"   Max throughput: {data['meanThr_DL'].max():.3f} Mbps")
            
            # Show variation
            prb_std = data['PRBUsageDL'].std()
            print(f"   PRB Usage variation: {prb_std:.2f} (higher = more activity)")
        
    finally:
        monitor.stop_monitoring()
    
    input("\nPress Enter to continue to final test...")


def test_realtime_detection():
    """Test 3: Real-time anomaly detection."""
    print("\n" + "="*60)
    print("TEST 3: REAL-TIME ANOMALY DETECTION")
    print("="*60)
    print("This will run live anomaly detection on your network usage.")
    print("The system will train a model first, then monitor for anomalies.")
    print()
    
    detector = RealTimeAnomalyDetector()
    
    # Custom alert handler
    def alert_handler(details):
        print(f"🚨 ANOMALY ALERT: {details['cell_name']} - {details['anomaly_probability']:.1%} confidence")
        # You could add email, log file, etc. here
    
    detector.set_alert_callback(alert_handler)
    
    try:
        print("🤖 Training anomaly detection model...")
        if detector.start_detection():
            print("✅ Real-time detection started!")
            print("\n💡 To trigger anomaly detection, try:")
            print("   - Download large files")
            print("   - Stream multiple videos")
            print("   - Run network speed tests") 
            print("   - Update software/apps")
            print("\n📊 Live monitoring (60 seconds):")
            print("-" * 50)
            
            # Monitor for 60 seconds
            for i in range(12):  # 5-second intervals
                time.sleep(5)
                
                stats = detector.get_statistics()
                recent_preds = detector.get_recent_predictions(count=3)
                
                if recent_preds:
                    latest = recent_preds[0]
                    status = "🔴 ANOMALY" if latest['is_anomaly'] else "🟢 NORMAL"
                    print(f"{status} | {latest['timestamp'].strftime('%H:%M:%S')} | "
                          f"Prob: {latest['anomaly_probability']:.3f}")
            
            # Final stats
            stats = detector.get_statistics()
            print(f"\n📈 Final Statistics:")
            print(f"   Predictions made: {stats['total_predictions']}")
            print(f"   Anomalies detected: {stats['anomalies_detected']}")
            print(f"   Detection rate: {stats['anomaly_rate']:.1f}%")
        
        else:
            print("❌ Failed to start real-time detection")
    
    finally:
        detector.stop_detection()
    
    print("\n✅ All tests complete!")


def main():
    """Run all real-time testing demos."""
    print("🌐 REAL-TIME NETWORK ANOMALY DETECTION TESTING")
    print("=" * 60)
    print("This demo will show you three ways to test your anomaly detection")
    print("system using your computer's real network usage:")
    print()
    print("1. 📊 Basic Network Monitoring")
    print("2. 🚀 Artificial Load Generation")  
    print("3. 🤖 Live Anomaly Detection")
    print()
    
    choice = input("Choose test (1-3, or 'all' for all tests): ").lower()
    
    if choice == '1':
        test_network_monitor()
    elif choice == '2':
        test_load_generation()
    elif choice == '3':
        test_realtime_detection()
    elif choice == 'all':
        test_network_monitor()
        test_load_generation()
        test_realtime_detection()
    else:
        print("Invalid choice. Running all tests...")
        test_network_monitor()
        test_load_generation()
        test_realtime_detection()
    
    print("\n🎉 Testing complete!")
    print("📁 Check the 'output' folder for saved data files")
    print("🔬 You can now use this real network data to improve your model")


if __name__ == "__main__":
    main()
