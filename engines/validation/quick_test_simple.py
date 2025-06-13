#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick Indicator Import Test - Windows Compatible
Tests if the restored base classes fixed the indicator import issues
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent

def test_indicator_imports():
    """Test importing various indicators to verify Unicode and import fixes work"""
    success_count = 0
    failed_count = 0
    
    # Test momentum indicators
    momentum_tests = [
        ("engines.momentum.cci", "Cci"),
        ("engines.momentum.mfi", "Mfi"),
        ("engines.momentum.roc", "Roc"),
        ("engines.momentum.williams_r", "WilliamsR"),
        ("engines.momentum.ultimate_oscillator", "UltimateOscillator")
    ]
    
    # Test trend indicators
    trend_tests = [
        ("engines.trend.adx", "Adx"),
        ("engines.trend.aroon", "Aroon"),
        ("engines.trend.ema", "Ema"),
        ("engines.trend.macd", "Macd")
    ]
    
    # Test volume indicators
    volume_tests = [
        ("engines.volume.accumulation_distribution", "AccumulationDistribution"),
        ("engines.volume.chaikin_money_flow", "ChaikinMoneyFlow"),
        ("engines.volume.on_balance_volume", "OnBalanceVolume")
    ]
    
    all_tests = momentum_tests + trend_tests + volume_tests
    
    print("Testing indicator imports after base class restoration...")
    print("=" * 60)
    
    for module_path, class_name in all_tests:
        try:
            module = __import__(module_path, fromlist=[class_name])
            indicator_class = getattr(module, class_name)
            print(f"[SUCCESS] {module_path}.{class_name}")
            success_count += 1
        except ImportError as e:
            print(f"[FAILED] {module_path}.{class_name} - ImportError: {e}")
            failed_count += 1
        except AttributeError as e:
            print(f"[FAILED] {module_path}.{class_name} - AttributeError: {e}")
            failed_count += 1
        except Exception as e:
            print(f"[FAILED] {module_path}.{class_name} - {type(e).__name__}: {e}")
            failed_count += 1
    
    print("=" * 60)
    total_tests = success_count + failed_count
    success_rate = (success_count / total_tests * 100) if total_tests > 0 else 0
    
    print(f"RESULTS:")
    print(f"  Success: {success_count}/{total_tests} ({success_rate:.1f}%)")
    print(f"  Failed:  {failed_count}/{total_tests}")
    
    if success_rate > 80:
        print(f"EXCELLENT! Indicators are importing successfully.")
        print(f"Base class restoration was successful!")
    elif success_rate > 50:
        print(f"PARTIAL SUCCESS. Some indicators still have issues.")
    else:
        print(f"MAJOR ISSUES. Most indicators still failing to import.")
    
    return success_count, failed_count

if __name__ == "__main__":
    test_indicator_imports()