# Real-Time Network Anomaly Detection Testing Guide

## 🌐 Overview

This guide shows you how to test your network anomaly detection system using **real-time data from your own computer's network usage**. This is perfect for:

- **Testing your model** with live data
- **Demonstrating the system** to stakeholders  
- **Development and debugging** in a controlled environment
- **Understanding network patterns** from actual usage

## 🚀 Quick Start

### 1. **Basic Network Monitoring Test**
```bash
python network_monitor.py
```
- Monitors your network for real-time data
- Converts system metrics to cellular network format
- Saves data for analysis

### 2. **Interactive Testing Demo**
```bash
python test_realtime.py
```
- Choose from 3 different testing methods
- Guided walkthrough of each approach
- Generates network load for testing

### 3. **Full Real-Time Anomaly Detection**
```bash
python realtime_detector.py
```
- Complete live anomaly detection system
- Uses your trained model on live data
- Real-time alerts and monitoring

## 📊 Testing Methods Explained

### Method 1: Network Usage Monitoring

**What it does:**
- Monitors your computer's network traffic in real-time
- Converts bytes/packets to cellular-like metrics (PRB usage, throughput, user equipment)
- Simulates multiple cell towers using your single connection

**How to use:**
```python
from network_monitor import NetworkMonitor

monitor = NetworkMonitor(collection_interval=5)  # Every 5 seconds
monitor.start_monitoring()

# Let it run while you use your computer
# - Browse websites, stream videos, download files
# - Open multiple browser tabs, update software

# Save the collected data
data = monitor.get_recent_data(minutes=30)
monitor.save_data_to_csv("my_network_test.csv")
```

**Generated Data Format:**
```
Time,CellName,PRBUsageUL,PRBUsageDL,meanThr_DL,meanThr_UL,maxThr_DL,maxThr_UL,meanUE_DL,meanUE_UL,maxUE_DL,maxUE_UL,maxUE_UL+DL
21:30,5ALTE,12.5,34.2,0.045,0.012,0.089,0.025,8.3,5.1,15.2,9.8,25.0
21:31,2BLTE,8.1,28.7,0.032,0.008,0.067,0.018,6.7,4.2,12.1,7.9,20.0
```

### Method 2: Artificial Load Generation

**What it does:**
- Generates controlled network traffic using HTTP requests
- Creates patterns that simulate different network conditions
- Useful for testing specific scenarios

**How to trigger:**
```python
monitor = NetworkMonitor()
monitor.start_monitoring()

# Generate artificial load
monitor.simulate_network_load(duration_seconds=60)

# This will make multiple HTTP requests to create network activity
# You'll see corresponding increases in PRB usage and throughput
```

### Method 3: Real-Time Anomaly Detection

**What it does:**
- Uses your trained anomaly detection model
- Monitors network usage continuously
- Detects and alerts on anomalies in real-time

**Example output:**
```
🟢 NORMAL | 21:32:15 | 3CLTE | Prob: 0.234 | Conf: Low
🟢 NORMAL | 21:32:25 | 7ALTE | Prob: 0.156 | Conf: Low  
🔴 ANOMALY | 21:32:35 | 5BLTE | Prob: 0.847 | Conf: High

🚨 ANOMALY ALERT 🚨
Time: 2025-08-26 21:32:35
Cell: 5BLTE  
Probability: 84.7%
Confidence: High
```

## 🔧 How the System Converts Network Data

### Your Computer → Cellular Network Format

| **System Metric** | **Cellular Equivalent** | **Conversion** |
|-------------------|-------------------------|----------------|
| Bytes sent/received per second | PRB Usage UL/DL | Network activity → Resource block utilization |
| Network throughput | Mean/Max Throughput | Direct conversion to Mbps |
| Active connections | User Equipment (UE) | TCP connections → Connected devices |
| CPU/Memory usage | Efficiency metrics | System load → Network performance |
| Time of day | Business/Peak hours | Same time-based patterns |

### Simulation Logic

```python
# PRB Usage simulation (0-100%)
prb_ul = min(100, (bytes_sent_rate / 1024 / 100) * 10 + random_variation)
prb_dl = min(100, (bytes_recv_rate / 1024 / 100) * 10 + random_variation)

# Throughput conversion  
mean_thr_ul = (bytes_sent_rate * 8) / (1024 * 1024)  # To Mbps
mean_thr_dl = (bytes_recv_rate * 8) / (1024 * 1024)  # To Mbps

# User equipment simulation
mean_ue_ul = active_connections * random_factor
mean_ue_dl = active_connections * random_factor
```

## 💡 Testing Scenarios

### Scenario 1: Normal Usage Testing
**Goal:** Collect baseline normal network behavior

```bash
# Start monitoring
python -c "
from network_monitor import NetworkMonitor
monitor = NetworkMonitor()
monitor.start_monitoring()
input('Let this run for 30+ minutes while using computer normally...')
monitor.save_data_to_csv('normal_usage.csv')
"
```

**Activities:**
- Regular web browsing
- Email checking
- Light streaming
- Normal application usage

### Scenario 2: High Load Testing  
**Goal:** Generate anomaly-like behavior

```bash
# Generate high network load
python -c "
from network_monitor import NetworkMonitor
monitor = NetworkMonitor()
monitor.start_monitoring()
monitor.simulate_network_load(duration_seconds=300)  # 5 minutes
monitor.save_data_to_csv('high_load.csv')
"
```

**Activities:**
- Download large files simultaneously
- Stream multiple 4K videos
- Run network speed tests
- Update multiple applications
- Video calls + streaming

### Scenario 3: Pattern Testing
**Goal:** Test specific time-based patterns

```python
import schedule
from network_monitor import NetworkMonitor

monitor = NetworkMonitor()
monitor.start_monitoring()

# Generate load every hour to test time-based detection
schedule.every().hour.do(lambda: monitor.simulate_network_load(60))
```

## 🎯 What Triggers Anomalies

### High Probability Anomalies (80%+)
- **Massive downloads:** Multiple large file downloads
- **Video streaming spikes:** Starting several HD streams
- **Software updates:** OS or large app updates
- **Speed tests:** Running bandwidth tests
- **Gaming + streaming:** High-bandwidth gaming with streaming

### Medium Probability Anomalies (50-80%)  
- **Multiple video calls:** Several simultaneous calls
- **Cloud backups:** Large file sync operations
- **Multiple browsers:** Many tabs with media content
- **P2P applications:** BitTorrent, file sharing

### Factors That Increase Detection
- **Sudden spikes** in data usage
- **Sustained high throughput** (>5 Mbps for extended periods)
- **High connection counts** (>50 active connections)
- **Unusual time patterns** (heavy usage at 3 AM)
- **Combined metrics** (high PRB + high throughput + many users)

## 🔍 Interpreting Results

### Normal Network Usage
```
PRB Usage UL: 5-15%
PRB Usage DL: 10-30%  
Mean Throughput: 0.001-0.1 Mbps
Active Connections: 5-20
Anomaly Probability: 0-30%
```

### Anomalous Network Usage  
```
PRB Usage UL: 25-80%
PRB Usage DL: 40-95%
Mean Throughput: 1-50+ Mbps  
Active Connections: 30-100+
Anomaly Probability: 50-100%
```

## 📈 Advanced Testing Features

### Custom Alert Callbacks
```python
def my_alert_handler(details):
    # Send email notification
    send_email(f"Anomaly detected: {details['anomaly_probability']:.1%}")
    
    # Log to file
    with open('anomaly_log.txt', 'a') as f:
        f.write(f"{details['timestamp']}: {details}\n")
    
    # Trigger automated response
    if details['anomaly_probability'] > 0.9:
        restart_network_service()

detector.set_alert_callback(my_alert_handler)
```

### Continuous Learning
```python
# Retrain model with new data every 24 hours
detector = RealTimeAnomalyDetector(retrain_hours=24)

# Save models for later use
detector.save_model("output/production_model.joblib")
detector.load_model("output/production_model.joblib")
```

### Performance Monitoring
```python
stats = detector.get_statistics()
print(f"""
Runtime: {stats['uptime_hours']:.1f} hours
Total predictions: {stats['total_predictions']}
Anomalies detected: {stats['anomalies_detected']} 
Detection rate: {stats['anomaly_rate']:.1f}%
Predictions/hour: {stats['predictions_per_hour']:.1f}
""")
```

## 🛠️ Troubleshooting

### Common Issues

**Issue:** "No network activity detected"  
**Solution:** Use your computer more actively or run the load generation

**Issue:** "All predictions are normal"  
**Solution:** Generate more network load or lower the anomaly threshold

**Issue:** "Too many false positives"
**Solution:** Collect more normal usage data for training

**Issue:** "System running slow during monitoring"
**Solution:** Increase collection interval or reduce concurrent activities

### Performance Optimization

```python
# Reduce monitoring frequency
monitor = NetworkMonitor(collection_interval=30)  # Every 30 seconds

# Limit data buffer size
monitor.data_buffer = deque(maxlen=100)  # Keep only 100 recent points

# Adjust anomaly threshold
detector.anomaly_threshold = 0.7  # Less sensitive (fewer false positives)
```

## 📁 Output Files

### Generated Files
- `network_data_YYYYMMDD_HHMMSS.csv` - Raw network monitoring data
- `realtime_model.joblib` - Trained model for reuse
- `anomaly_log.txt` - Alert history log
- `exploration.png` - Network pattern visualizations

### Data Analysis
```python
import pandas as pd
import matplotlib.pyplot as plt

# Load your real network data
data = pd.read_csv('output/network_data_20250826_213000.csv')

# Compare with original training data  
print("Real network vs Training data comparison:")
print(f"Real PRB UL avg: {data['PRBUsageUL'].mean():.1f}%")
print(f"Real PRB DL avg: {data['PRBUsageDL'].mean():.1f}%")

# Plot patterns
data['Hour'] = pd.to_datetime(data['Time'], format='%H:%M').dt.hour
hourly_usage = data.groupby('Hour')['PRBUsageDL'].mean()
plt.plot(hourly_usage.index, hourly_usage.values)
plt.title('Your Network Usage Pattern by Hour')
plt.show()
```

## 🎉 Next Steps

1. **Collect baseline data** with normal computer usage (1+ hours)
2. **Generate test anomalies** using high network load
3. **Run live detection** and monitor for alerts
4. **Analyze patterns** in your real usage data
5. **Integrate with your web application** for live monitoring
6. **Set up automated alerts** for production monitoring

## 💡 Use Cases for Real Network Testing

### Development & Testing
- **Model validation** with live data streams
- **Performance testing** under real conditions  
- **Feature engineering** based on actual patterns
- **Threshold tuning** for optimal detection

### Demonstrations
- **Live demos** to stakeholders showing real-time detection
- **Proof of concept** with immediate visual feedback
- **Training presentations** with interactive examples
- **Customer demonstrations** with controllable scenarios

### Production Preparation  
- **Stress testing** the detection pipeline
- **Alert system validation** with real triggers
- **Performance benchmarking** under load
- **Integration testing** with external systems

Your network anomaly detection system is now ready for comprehensive real-time testing! 🚀
