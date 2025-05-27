#!/usr/bin/env python3
"""
Test script to verify Volume Analysis import dependencies fix
"""

import sys
import os
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_volume_analysis_imports():
    """Test Volume Analysis import dependencies"""
    
    try:
        print("🧪 Testing Volume Analysis import dependencies fix...")
        
        # Test 1: Import TickVolumeIndicators with corrected classes
        print("\n🧪 Test 1: Import TickVolumeIndicators")
        try:
            sys.path.append('services/analytics-service/src/engines/volume')
            from TickVolumeIndicators import (
                TickVolumeIndicators,
                VolumeSignal,
                TickVolumeResult,
                VolumeStrength,
                TickVolumeMetrics,
                VolumeConfirmation
            )
            print("✅ TickVolumeIndicators import successful")
        except Exception as e:
            print(f"❌ TickVolumeIndicators import failed: {e}")
            return False
        
        # Test 2: Import VolumeSpreadAnalysis
        print("\n🧪 Test 2: Import VolumeSpreadAnalysis")
        try:
            from VolumeSpreadAnalysis import (
                VolumeSpreadAnalysis,
                VSAAnalysisResult,
                VSABar,
                VSASignalType,
                VolumeStrength as VSAVolumeStrength,
                SpreadSize
            )
            print("✅ VolumeSpreadAnalysis import successful")
        except Exception as e:
            print(f"❌ VolumeSpreadAnalysis import failed: {e}")
            return False
        
        # Test 3: Import OrderFlowImbalance
        print("\n🧪 Test 3: Import OrderFlowImbalance")
        try:
            from OrderFlowImbalance import (
                OrderFlowImbalance,
                OrderFlowAnalysisResult,
                OrderFlowBar,
                ImbalanceType,
                ImbalanceStrength
            )
            print("✅ OrderFlowImbalance import successful")
        except Exception as e:
            print(f"❌ OrderFlowImbalance import failed: {e}")
            return False
        
        # Test 4: Import VolumeProfiles
        print("\n🧪 Test 4: Import VolumeProfiles")
        try:
            from VolumeProfiles import (
                VolumeProfiles,
                VolumeProfileAnalysisResult,
                VolumeProfile,
                VolumeNode,
                ValueArea,
                TradingSession,
                VolumeNodeType
            )
            print("✅ VolumeProfiles import successful")
        except Exception as e:
            print(f"❌ VolumeProfiles import failed: {e}")
            return False
        
        # Test 5: Import SmartMoneyIndicators
        print("\n🧪 Test 5: Import SmartMoneyIndicators")
        try:
            from SmartMoneyIndicators import (
                SmartMoneyIndicators,
                SmartMoneyAnalysisResult,
                SmartMoneySignal,
                InstitutionalFootprint,
                SmartMoneyActivity,
                InstitutionalBehavior,
                FlowStrength
            )
            print("✅ SmartMoneyIndicators import successful")
        except Exception as e:
            print(f"❌ SmartMoneyIndicators import failed: {e}")
            return False
        
        # Test 6: Test volume module __init__.py imports
        print("\n🧪 Test 6: Test volume module __init__.py imports")
        try:
            # Reset path to test module imports
            sys.path.append('services/analytics-service/src/engines')
            from volume import (
                TickVolumeIndicators,
                VolumeSignal,
                TickVolumeResult,
                VolumeStrength,
                TickVolumeMetrics,
                VolumeConfirmation,
                VolumeSpreadAnalysis,
                VSAAnalysisResult,
                VSABar,
                VSASignalType,
                OrderFlowImbalance,
                VolumeProfiles,
                SmartMoneyIndicators
            )
            print("✅ Volume module __init__.py imports successful")
        except Exception as e:
            print(f"❌ Volume module __init__.py imports failed: {e}")
            return False
        
        # Test 7: Test instantiation of volume analysis classes
        print("\n🧪 Test 7: Test instantiation of volume analysis classes")
        try:
            # Test TickVolumeIndicators instantiation
            tick_volume = TickVolumeIndicators()
            print(f"✅ TickVolumeIndicators instantiated")
            
            # Test VolumeSpreadAnalysis instantiation
            vsa = VolumeSpreadAnalysis()
            print(f"✅ VolumeSpreadAnalysis instantiated")
            
            # Test VolumeProfiles instantiation
            volume_profiles = VolumeProfiles()
            print(f"✅ VolumeProfiles instantiated")
            
            # Test SmartMoneyIndicators instantiation
            smart_money = SmartMoneyIndicators()
            print(f"✅ SmartMoneyIndicators instantiated")
            
        except Exception as e:
            print(f"❌ Volume analysis instantiation failed: {e}")
            return False
        
        # Test 8: Test enum functionality
        print("\n🧪 Test 8: Test enum functionality")
        try:
            # Test VolumeSignal enum
            signal = VolumeSignal.BULLISH
            print(f"✅ VolumeSignal enum: {signal.value}")
            
            # Test VSASignalType enum
            vsa_signal = VSASignalType.ACCUMULATION
            print(f"✅ VSASignalType enum: {vsa_signal.value}")
            
            # Test ImbalanceType enum
            imbalance = ImbalanceType.BUY_IMBALANCE
            print(f"✅ ImbalanceType enum: {imbalance.value}")
            
            # Test TradingSession enum
            session = TradingSession.LONDON
            print(f"✅ TradingSession enum: {session.value}")
            
            # Test SmartMoneyActivity enum
            activity = SmartMoneyActivity.ACCUMULATION
            print(f"✅ SmartMoneyActivity enum: {activity.value}")
            
        except Exception as e:
            print(f"❌ Enum functionality test failed: {e}")
            return False
        
        print("\n🎉 All Volume Analysis import dependency tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing Volume Analysis Import Dependencies Fix")
    print("=" * 70)
    
    success = test_volume_analysis_imports()
    
    if success:
        print("\n✅ Volume Analysis import dependencies fix verified successfully!")
        print("   - All 4 volume analysis indicators can be imported")
        print("   - TickVolumeSignal dependency issues resolved")
        print("   - Volume analysis functionality restored")
        sys.exit(0)
    else:
        print("\n❌ Volume Analysis import dependencies fix verification failed!")
        sys.exit(1)
