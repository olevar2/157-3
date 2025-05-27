#!/usr/bin/env python3
"""
Test script to verify AutoencoderFeatures parameter initialization fixes
"""

import sys
import os
import numpy as np
import logging

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_autoencoder_features_initialization():
    """Test AutoencoderFeatures initialization with various parameter combinations"""

    try:
        # Import the fixed AutoencoderFeatures
        sys.path.append('services/analytics-service/src/engines/indicators/advanced')
        from AutoencoderFeatures import (
            AutoencoderFeatures, AutoencoderType, AnomalyLevel
        )

        print("✅ AutoencoderFeatures import successful")

        # Test 1: Valid initialization with minimum parameters
        print("\n🧪 Test 1: Valid initialization with minimum parameters")
        try:
            autoencoder = AutoencoderFeatures(input_dim=10)
            print(f"✅ Basic initialization successful: input_dim={autoencoder.input_dim}, encoding_dim={autoencoder.encoding_dim}")
        except Exception as e:
            print(f"❌ Basic initialization failed: {e}")
            return False

        # Test 2: Valid initialization with all parameters
        print("\n🧪 Test 2: Valid initialization with all parameters")
        try:
            autoencoder = AutoencoderFeatures(
                input_dim=20,
                encoding_dim=5,
                autoencoder_type=AutoencoderType.DENOISING,
                noise_factor=0.2,
                sparsity_regularizer=1e-4,
                feature_names=['price', 'volume', 'rsi'] + [f'feature_{i}' for i in range(17)]
            )
            print(f"✅ Full initialization successful: input_dim={autoencoder.input_dim}, encoding_dim={autoencoder.encoding_dim}")
        except Exception as e:
            print(f"❌ Full initialization failed: {e}")
            return False

        # Test 3: Invalid input_dim (None)
        print("\n🧪 Test 3: Invalid input_dim (None)")
        try:
            autoencoder = AutoencoderFeatures(input_dim=None)
            print("❌ Should have failed with None input_dim")
            return False
        except ValueError as e:
            print(f"✅ Correctly rejected None input_dim: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

        # Test 4: Invalid input_dim (negative)
        print("\n🧪 Test 4: Invalid input_dim (negative)")
        try:
            autoencoder = AutoencoderFeatures(input_dim=-5)
            print("❌ Should have failed with negative input_dim")
            return False
        except ValueError as e:
            print(f"✅ Correctly rejected negative input_dim: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

        # Test 5: Invalid input_dim (zero)
        print("\n🧪 Test 5: Invalid input_dim (zero)")
        try:
            autoencoder = AutoencoderFeatures(input_dim=0)
            print("❌ Should have failed with zero input_dim")
            return False
        except ValueError as e:
            print(f"✅ Correctly rejected zero input_dim: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

        # Test 6: Invalid input_dim (string)
        print("\n🧪 Test 6: Invalid input_dim (string)")
        try:
            autoencoder = AutoencoderFeatures(input_dim="10")
            print("❌ Should have failed with string input_dim")
            return False
        except TypeError as e:
            print(f"✅ Correctly rejected string input_dim: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

        # Test 7: Invalid encoding_dim (negative)
        print("\n🧪 Test 7: Invalid encoding_dim (negative)")
        try:
            autoencoder = AutoencoderFeatures(input_dim=10, encoding_dim=-2)
            print("❌ Should have failed with negative encoding_dim")
            return False
        except ValueError as e:
            print(f"✅ Correctly rejected negative encoding_dim: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

        # Test 8: Warning for encoding_dim >= input_dim
        print("\n🧪 Test 8: Warning for encoding_dim >= input_dim")
        try:
            autoencoder = AutoencoderFeatures(input_dim=5, encoding_dim=5)
            print(f"✅ Accepted encoding_dim >= input_dim with warning: input_dim={autoencoder.input_dim}, encoding_dim={autoencoder.encoding_dim}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

        # Test 9: Invalid noise_factor (negative)
        print("\n🧪 Test 9: Invalid noise_factor (negative)")
        try:
            autoencoder = AutoencoderFeatures(input_dim=10, noise_factor=-0.1)
            print("❌ Should have failed with negative noise_factor")
            return False
        except ValueError as e:
            print(f"✅ Correctly rejected negative noise_factor: {e}")
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False

        # Test 10: Auto-calculated encoding_dim for various input sizes
        print("\n🧪 Test 10: Auto-calculated encoding_dim for various input sizes")
        test_sizes = [3, 6, 15, 30, 100]
        for size in test_sizes:
            try:
                autoencoder = AutoencoderFeatures(input_dim=size)
                print(f"✅ input_dim={size} -> encoding_dim={autoencoder.encoding_dim}")
            except Exception as e:
                print(f"❌ Failed for input_dim={size}: {e}")
                return False

        # Test 11: Test basic functionality
        print("\n🧪 Test 11: Test basic functionality")
        try:
            autoencoder = AutoencoderFeatures(input_dim=5)

            # Test with mock data
            test_data = np.random.rand(10, 5)
            result = autoencoder.transform(test_data)

            print(f"✅ Transform successful: anomaly_level={result.anomaly_level.value}, "
                  f"compression_ratio={result.compression_ratio:.3f}")
        except Exception as e:
            print(f"❌ Basic functionality test failed: {e}")
            return False

        print("\n🎉 All AutoencoderFeatures parameter validation tests passed!")
        return True

    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing AutoencoderFeatures Parameter Initialization Fixes")
    print("=" * 70)

    success = test_autoencoder_features_initialization()

    if success:
        print("\n✅ AutoencoderFeatures parameter validation fix verified successfully!")
        sys.exit(0)
    else:
        print("\n❌ AutoencoderFeatures parameter validation fix verification failed!")
        sys.exit(1)
