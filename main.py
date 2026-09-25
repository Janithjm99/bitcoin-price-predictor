# main.py - Run this file to execute the entire project

import subprocess
import sys

def run_project():
    print("🚀 Starting Bitcoin Price Prediction Project\n")
    print("=" * 60)
    
    try:
        # Run the main script
        result = subprocess.run([sys.executable, 'bitcoin_predictor.py'], 
                              capture_output=False, text=True)
        
        print("\n" + "=" * 60)
        print("✅ Project completed successfully!")
        print("Check the generated files:")
        print("  📁 bitcoin_exploration.png - Initial data exploration")
        print("  📁 bitcoin_predictions.png - Model predictions")
        print("  📁 bitcoin_rf_model.pkl - Saved machine learning model")
        print("  📁 bitcoin_scaler.pkl - Saved data scaler")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    run_project()