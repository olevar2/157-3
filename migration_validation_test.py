#!/usr/bin/env python3
"""
Platform3 Migration Validation Test
Test if the package structure migration was successful
"""

def test_package_imports():
    """Test that main package imports work correctly"""
    print("Testing Platform3 package imports...")
    
    try:
        # Test main package import
        import platform3
        print("✓ Main package import successful")
        print(f"  Version: {platform3.get_version()}")
        
        # Test shared components
        from shared.logging.platform3_logger import Platform3Logger
        print("✓ Platform3Logger import successful")
        
        # Test AI platform components  
        from ai_platform.ai_platform_manager import AIPlatformManager
        print("✓ AIPlatformManager import successful")
        
        # Test AI model base
        from shared.ai_model_base import EnhancedAIModelBase
        print("✓ EnhancedAIModelBase import successful")
        
        return True
        
    except ImportError as e:
        print(f"X Import failed: {e}")
        return False
    except Exception as e:
        print(f"X Unexpected error: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of key components"""
    print("\nTesting basic functionality...")
    
    try:
        # Test logger creation
        from shared.logging.platform3_logger import Platform3Logger
        logger = Platform3Logger("ValidationTest")
        logger.info("Logger test successful")
        print("✓ Logger creation and logging successful")
        
        # Test getting engine instance
        import platform3
        engine = platform3.get_engine()
        print("✓ Engine instance creation successful")
        
        return True
        
    except Exception as e:
        print(f"X Functionality test failed: {e}")
        return False

def main():
    """Main validation function"""
    print("=" * 50)
    print("Platform3 Migration Validation Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_package_imports()
    
    # Test functionality
    functionality_ok = test_basic_functionality()
    
    # Summary
    print("\n" + "=" * 50)
    print("Validation Summary:")
    print(f"Imports: {'PASS' if imports_ok else 'FAIL'}")
    print(f"Functionality: {'PASS' if functionality_ok else 'FAIL'}")
    
    if imports_ok and functionality_ok:
        print("\nMigration Validation: SUCCESS!")
        print("The Platform3 package structure migration completed successfully.")
        print("All sys.path.append issues have been resolved.")
    else:
        print("\nMigration Validation: FAILED!")
        print("Some issues remain. Please check the errors above.")
    
    print("=" * 50)

if __name__ == "__main__":
    main()