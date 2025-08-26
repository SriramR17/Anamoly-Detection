"""
Simple Data Explorer for Network Anomaly Detection
==================================================
Basic exploration with essential insights and clean visualizations.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from config import TARGET_COL, NUMERIC_COLS, OUTPUT_DIR


def explore_data(train_data):
    """
    Perform basic data exploration.
    
    Args:
        train_data: Training dataset
        
    Returns:
        dict: Key insights from the data
    """
    print("Exploring data...")
    
    # Clean data for analysis
    train_clean = train_data.copy()
    for col in NUMERIC_COLS:
        if col in train_clean.columns:
            train_clean[col] = pd.to_numeric(train_clean[col], errors='coerce')
    
    # Basic statistics
    print(f"\nDataset Info:")
    print(f"  Shape: {train_data.shape}")
    print(f"  Missing values: {train_data.isnull().sum().sum()}")
    
    # Target distribution
    target_counts = train_data[TARGET_COL].value_counts()
    print(f"\nTarget Distribution:")
    print(f"  Normal (0): {target_counts[0]:,} ({target_counts[0]/len(train_data)*100:.1f}%)")
    print(f"  Anomaly (1): {target_counts[1]:,} ({target_counts[1]/len(train_data)*100:.1f}%)")
    
    # Time patterns
    train_clean['Hour'] = pd.to_datetime(train_clean['Time'], format='%H:%M').dt.hour
    hourly_anomalies = train_clean.groupby('Hour')[TARGET_COL].mean()
    peak_hours = hourly_anomalies.nlargest(3).index.tolist()
    print(f"\nTime Patterns:")
    print(f"  Peak anomaly hours: {peak_hours}")
    
    # Cell patterns
    cell_anomalies = train_data.groupby('CellName')[TARGET_COL].mean()
    top_anomaly_cells = cell_anomalies.nlargest(5).index.tolist()
    print(f"\nCell Patterns:")
    print(f"  Top 5 anomaly cells: {top_anomaly_cells}")
    
    # Feature correlations with target
    correlations = train_clean[NUMERIC_COLS + [TARGET_COL]].corr()[TARGET_COL].sort_values(ascending=False)
    top_features = correlations.drop(TARGET_COL).head(3)
    print(f"\nTop 3 Correlated Features:")
    for feature, corr in top_features.items():
        print(f"  {feature}: {corr:.3f}")
    
    # Create simple visualization
    _create_basic_plots(train_clean)
    
    return {
        'target_distribution': target_counts.to_dict(),
        'peak_anomaly_hours': peak_hours,
        'top_anomaly_cells': top_anomaly_cells,
        'top_correlated_features': top_features.to_dict()
    }


def _create_basic_plots(train_data):
    """Create basic visualization plots."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Network Data Exploration', fontsize=14)
    
    # 1. Target distribution
    target_counts = train_data[TARGET_COL].value_counts()
    axes[0, 0].bar(['Normal', 'Anomaly'], target_counts.values, color=['blue', 'red'])
    axes[0, 0].set_title('Target Distribution')
    axes[0, 0].set_ylabel('Count')
    
    # 2. Anomaly rate by hour
    hourly_anomalies = train_data.groupby('Hour')[TARGET_COL].mean() * 100
    axes[0, 1].plot(hourly_anomalies.index, hourly_anomalies.values, marker='o')
    axes[0, 1].set_title('Anomaly Rate by Hour')
    axes[0, 1].set_xlabel('Hour')
    axes[0, 1].set_ylabel('Anomaly Rate (%)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature distributions (example with PRB Usage)
    normal_data = train_data[train_data[TARGET_COL] == 0]['PRBUsageUL']
    anomaly_data = train_data[train_data[TARGET_COL] == 1]['PRBUsageUL']
    axes[1, 0].hist(normal_data, bins=30, alpha=0.6, label='Normal', color='blue')
    axes[1, 0].hist(anomaly_data, bins=30, alpha=0.6, label='Anomaly', color='red')
    axes[1, 0].set_title('PRB Usage UL Distribution')
    axes[1, 0].set_xlabel('PRB Usage UL')
    axes[1, 0].legend()
    
    # 4. Correlation heatmap (top features)
    top_features = ['PRBUsageUL', 'PRBUsageDL', 'meanThr_DL', 'meanThr_UL', TARGET_COL]
    corr_matrix = train_data[top_features].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1, 1])
    axes[1, 1].set_title('Feature Correlations')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = OUTPUT_DIR / "exploration.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Exploration plots saved to: {plot_path}")
    
    plt.show()


if __name__ == "__main__":
    # Test the explorer
    try:
        from data_loader import load_data
        train_data, test_data = load_data()
        results = explore_data(train_data)
        print("✓ Data exploration test successful")
    except Exception as e:
        print(f"❌ Error: {e}")
