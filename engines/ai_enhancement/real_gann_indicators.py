"""
Real Gann Indicators Integration for Platform3
Connects all 7 real Gann modules to the indicator registry with fixes for compatibility
"""

# Import all real Gann modules with fixes
try:
    from engines.ai_enhancement.gann_indicator_fixes import (
        GannAnglesCalculatorFixed,
        GannPatternDetectorWrapper,
        GannSquareOfNineFixed,
        GannTimeCyclesWrapper,
    )
    from engines.gann.GannGrid import GannGridIndicator
    from engines.gann.gann_fan_lines import GannFanLines
    from engines.gann.gann_time_cycles import GannTimeCycles
    from engines.gann.price_time_relationships import PriceTimeRelationships

    GANN_IMPORT_SUCCESS = True
    print("[OK] All real Gann modules imported successfully (with fixes)")
except ImportError as e:
    print(f"[WARN] Gann import issue: {e}")
    GANN_IMPORT_SUCCESS = False

# Real Gann Indicators Registry
GANN_INDICATORS = {}

if GANN_IMPORT_SUCCESS:
    # Clean Gann Indicators Registry - PRIMARY NAMES ONLY
    # Each indicator maps to exactly one unique implementation
    GANN_INDICATORS = {
        # 1. Gann Angles and Calculations (Fixed)
        "gann_angles_calculator": GannAnglesCalculatorFixed,
        # 2. Gann Grid Analysis
        "gann_grid": GannGridIndicator,
        # 3. Gann Pattern Recognition (Fixed Wrapper)
        "gann_pattern_detector": GannPatternDetectorWrapper,
        # 4. Gann Fan Lines
        "gann_fan_lines": GannFanLines,
        # 5. Gann Square of Nine (Fixed)
        "gann_square_of_nine": GannSquareOfNineFixed,
        # 6. Gann Time Cycles (Wrapper)
        "gann_time_cycles": GannTimeCyclesWrapper,
        # 7. Price-Time Relationships
        "price_time_relationships": PriceTimeRelationships,
    }

    print(f"[OK] {len(GANN_INDICATORS)} unique Gann indicators registered (no aliases)")
    print("[OK] Loaded 7 real Gann indicators")
else:
    # Fallback to simple stubs if imports fail
    class GannAngleStub:
        def __init__(self, period=20, **kwargs):
            self.period = period
            self.kwargs = kwargs

        def calculate(self, data):
            return None

    class GannPatternStub:
        def __init__(self, period=20, **kwargs):
            self.period = period
            self.kwargs = kwargs

        def calculate(self, data):
            return None

    GANN_INDICATORS = {
        "gann_angle": GannAngleStub,
        "gann_pattern": GannPatternStub,
    }
    print(f"[WARN] Using {len(GANN_INDICATORS)} Gann stubs due to import issues")

# Export for registry
__all__ = ["GANN_INDICATORS"]
