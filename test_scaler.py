#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import joblib
import sys

def test_scaler_direct():
    """Test load scaler langsung tanpa aplikasi"""
    print("=== Testing Scaler Load ===")
    
    # Cek file exists
    scaler_path = 'scaler_btc.save'
    print(f"File path: {scaler_path}")
    print(f"File exists: {os.path.exists(scaler_path)}")
    
    if os.path.exists(scaler_path):
        print(f"File size: {os.path.getsize(scaler_path)} bytes")
        print(f"File permissions: {oct(os.stat(scaler_path).st_mode)[-3:]}")
        
        # Coba load
        try:
            scaler = joblib.load(scaler_path)
            print(f"✅ Scaler loaded successfully!")
            print(f"Scaler type: {type(scaler)}")
            print(f"Scaler fitted: {hasattr(scaler, 'mean_')}")
            
            if hasattr(scaler, 'mean_'):
                print(f"Mean: {scaler.mean_}")
                print(f"Scale: {scaler.scale_}")
            else:
                print("❌ Scaler not fitted!")
                
        except Exception as e:
            print(f"❌ Error loading scaler: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ File not found!")
        
        # Cek current directory
        print(f"\nCurrent directory: {os.getcwd()}")
        print("Files in current directory:")
        for file in os.listdir('.'):
            if file.endswith('.save') or 'scaler' in file.lower():
                print(f"  - {file}")

def test_scikit_learn_version():
    """Test scikit-learn version compatibility"""
    print("\n=== Testing Scikit-learn Version ===")
    try:
        import sklearn
        print(f"Scikit-learn version: {sklearn.__version__}")
        
        # Test StandardScaler
        from sklearn.preprocessing import StandardScaler
        test_scaler = StandardScaler()
        print("✅ StandardScaler import successful")
        
        # Test fit
        import numpy as np
        test_data = np.array([[1], [2], [3]])
        test_scaler.fit(test_data)
        print("✅ StandardScaler fit successful")
        
    except Exception as e:
        print(f"❌ Scikit-learn error: {e}")

if __name__ == "__main__":
    test_scaler_direct()
    test_scikit_learn_version() 