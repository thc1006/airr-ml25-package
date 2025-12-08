#!/usr/bin/env python3
"""
GPU XGBoost with k=4 - Memory Optimized
"""
import sys
sys.path.insert(0, '.')

# Import from existing train_gpu.py and modify k value
exec(open('scripts/train_gpu.py').read().replace('k=3', 'k=4').replace('k3_', 'k4_'))
