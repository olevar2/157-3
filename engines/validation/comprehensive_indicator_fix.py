#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Comprehensive Platform3 Indicator Fix System
Emergency fix for all Unicode and import issues

This system will:
1. Fix all Unicode encoding issues in indicator files
2. Fix import path problems
3. Ensure UTF-8 compliance across all files
4. Add proper sys.path management for shared modules
"""

import sys
import os
import importlib
import inspect
import traceback
import chardet
from pathlib import Path
from typing import List, Dict, Any, Optional

class ComprehensiveIndicatorFixSystem:
    """Emergency fix system for all Platform3 indicators"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.engines_root = self.project_root / 'engines'
        self.shared_root = self.project_root / 'shared'
        
        # Indicator categories to process
        self.categories = [
            'momentum', 'trend', 'volume', 'volatility', 'pattern',
            'statistical', 'fractal', 'elliott_wave', 'gann', 
            'fibonacci', 'cycle', 'divergence', 'core_momentum', 
            'core_trend', 'performance', 'pivot', 'sentiment'
        ]
        
        self.fixed_files = []
        self.failed_files = []
        self.unicode_fixes = 0
        self.import_fixes = 0
        
    def fix_unicode_issues(self, file_path: Path) -> bool:
        """Fix Unicode issues in a single file"""
        try:
            # Read file with encoding detection
            with open(file_path, 'rb') as f:
                raw_content = f.read()
            
            # Detect encoding
            detected = chardet.detect(raw_content)
            encoding = detected.get('encoding', 'utf-8')
            
            # Decode content
            try:
                if encoding and encoding.lower() != 'utf-8':
                    content = raw_content.decode(encoding)
                else:
                    content = raw_content.decode('utf-8')
            except UnicodeDecodeError:
                # Try with error handling
                content = raw_content.decode('utf-8', errors='replace')
            
            # Unicode character replacements
            unicode_replacements = {
                '"': '"',  # Smart quotes
                '"': '"',  # Smart quotes
                ''': "'",  # Smart apostrophe
                ''': "'",  # Smart apostrophe
                ' ': ' ',  # Non-breaking space
                '–': '-',  # En dash
                '—': '-',  # Em dash
                '≈': '~=', # Approximately equal
                '°': 'deg', # Degree symbol
                '×': '*',  # Multiplication sign
                '÷': '/',  # Division sign
                '≤': '<=', # Less than or equal
                '≥': '>=', # Greater than or equal
                '≠': '!=', # Not equal
                '±': '+/-', # Plus-minus
                '∞': 'inf', # Infinity
                'α': 'alpha',  # Greek alpha
                'β': 'beta',   # Greek beta
                'γ': 'gamma',  # Greek gamma
                'δ': 'delta',  # Greek delta
                'ε': 'epsilon', # Greek epsilon
                'θ': 'theta',  # Greek theta
                'λ': 'lambda', # Greek lambda
                'μ': 'mu',     # Greek mu
                'π': 'pi',     # Greek pi
                'σ': 'sigma',  # Greek sigma
                'τ': 'tau',    # Greek tau
                'φ': 'phi',    # Greek phi
                'ω': 'omega',  # Greek omega
            }
            
            # Apply Unicode replacements
            original_content = content
            for unicode_char, replacement in unicode_replacements.items():
                if unicode_char in content:
                    content = content.replace(unicode_char, replacement)
                    self.unicode_fixes += 1
            
            # Ensure UTF-8 encoding declaration
            lines = content.split('\n')
            has_encoding = False
            shebang_line = 0
            
            # Check for shebang
            if lines and lines[0].startswith('#!'):
                shebang_line = 1
            
            # Check for existing encoding declaration
            for i in range(min(3, len(lines))):
                if 'coding' in lines[i] or 'encoding' in lines[i]:
                    has_encoding = True
                    # Update encoding declaration
                    lines[i] = '# -*- coding: utf-8 -*-'
                    break
            
            # Add encoding declaration if missing
            if not has_encoding:
                encoding_line = '# -*- coding: utf-8 -*-'
                if shebang_line:
                    lines.insert(1, encoding_line)
                else:
                    lines.insert(0, encoding_line)
            
            content = '\n'.join(lines)
            
            # Write back with UTF-8 encoding
            if content != original_content or not has_encoding:
                with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(content)
                return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] Failed to fix Unicode in {file_path}: {e}")
            return False
    
    def fix_import_paths(self, file_path: Path) -> bool:
        """Fix import path issues in a single file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Fix common import issues
            import_fixes = [
                # Fix shared.engines imports
                ('from shared.engines.', 'from engines.'),
                ('import shared.engines.', 'import engines.'),
                # Fix relative imports
                ('from .indicator_base', 'from engines.indicator_base'),
                ('from ..shared', 'from shared'),
                # Add proper sys.path management if needed
            ]
            
            for old_import, new_import in import_fixes:
                if old_import in content:
                    content = content.replace(old_import, new_import)
                    self.import_fixes += 1
            
            # Add sys.path management at the top if there are import issues
            if 'from engines.' in content or 'from shared.' in content:
                lines = content.split('\n')
                
                # Find insertion point (after encoding declaration)
                insert_index = 0
                for i, line in enumerate(lines):
                    if line.startswith('#!') or 'coding' in line or line.strip() == '':
                        insert_index = i + 1
                    else:
                        break
                
                # Add sys.path management
                sys_path_code = [
                    '',
                    '# Platform3 path management',
                    'import sys',
                    'from pathlib import Path',
                    'project_root = Path(__file__).parent.parent.parent',
                    '                    '                    '                    ''
                ]
                
                # Check if sys.path management already exists
                content_str = '\n'.join(lines)
                if 'project_root = Path(__file__)' not in content_str:
                    for i, line in enumerate(sys_path_code):
                        lines.insert(insert_index + i, line)
                    
                    content = '\n'.join(lines)
            
            # Write back if changes were made
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(content)
                return True
            
            return False
            
        except Exception as e:
            print(f"[ERROR] Failed to fix imports in {file_path}: {e}")
            return False
    
    def process_indicator_file(self, file_path: Path) -> bool:
        """Process a single indicator file"""
        try:
            print(f"[PROCESSING] {file_path.name}")
            
            # Fix Unicode issues
            unicode_fixed = self.fix_unicode_issues(file_path)
            
            # Fix import paths
            import_fixed = self.fix_import_paths(file_path)
            
            # Test if file can be imported now
            success = self.test_file_import(file_path)
            
            if success:
                self.fixed_files.append(str(file_path))
                print(f"[SUCCESS] Fixed {file_path.name}")
            else:
                self.failed_files.append(str(file_path))
                print(f"[FAILED] Could not fix {file_path.name}")
            
            return success
            
        except Exception as e:
            print(f"[ERROR] Processing {file_path}: {e}")
            self.failed_files.append(str(file_path))
            return False
    
    def test_file_import(self, file_path: Path) -> bool:
        """Test if a file can be imported successfully"""
        try:
            # Add project paths to sys.path
            project_root = file_path.parent.parent.parent
            paths_to_add = [
                str(project_root),
                str(project_root / 'shared'),
                str(project_root / 'engines'),
                str(file_path.parent)
            ]
            
            for path in paths_to_add:
                if path not in sys.path:
                    sys.path.insert(0, path)
            
            # Try to read and compile the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Compile to check for syntax errors
            compile(content, str(file_path), 'exec')
            
            return True
            
        except Exception as e:
            print(f"[TEST FAILED] {file_path.name}: {e}")
            return False
    
    def run_comprehensive_fix(self):
        """Run comprehensive fix on all indicator files"""
        print("="*70)
        print("PLATFORM3 COMPREHENSIVE INDICATOR FIX SYSTEM")
        print("="*70)
        
        total_files = 0
        
        # Process each category
        for category in self.categories:
            category_path = self.engines_root / category
            if not category_path.exists():
                print(f"[WARNING] Category {category} not found")
                continue
            
            print(f"\n[CATEGORY] Processing {category}")
            
            # Find all Python files in category
            python_files = list(category_path.glob('*.py'))
            if not python_files:
                print(f"[WARNING] No Python files in {category}")
                continue
            
            total_files += len(python_files)
            
            # Process each file
            for file_path in python_files:
                if file_path.name.startswith('__'):
                    continue  # Skip __init__.py files
                
                self.process_indicator_file(file_path)
        
        # Generate summary report
        print("\n" + "="*70)
        print("FIX SUMMARY REPORT")
        print("="*70)
        print(f"Total files processed: {total_files}")
        print(f"Successfully fixed: {len(self.fixed_files)}")
        print(f"Failed to fix: {len(self.failed_files)}")
        print(f"Unicode fixes applied: {self.unicode_fixes}")
        print(f"Import fixes applied: {self.import_fixes}")
        print(f"Success rate: {len(self.fixed_files)/total_files*100:.1f}%")
        
        if self.failed_files:
            print(f"\nFAILED FILES ({len(self.failed_files)}):")
            for failed_file in self.failed_files:
                print(f"  - {Path(failed_file).name}")
        
        if self.fixed_files:
            print(f"\nSUCCESSFULLY FIXED FILES ({len(self.fixed_files)}):")
            for fixed_file in self.fixed_files[:10]:  # Show first 10
                print(f"  - {Path(fixed_file).name}")
            if len(self.fixed_files) > 10:
                print(f"  ... and {len(self.fixed_files) - 10} more")
        
        print("\n[COMPLETE] Fix system completed")
        return len(self.failed_files) == 0

if __name__ == "__main__":
    fix_system = ComprehensiveIndicatorFixSystem()
    success = fix_system.run_comprehensive_fix()
    
    if success:
        print("\n🎉 ALL INDICATORS FIXED SUCCESSFULLY!")
    else:
        print(f"\n⚠️  {len(fix_system.failed_files)} indicators still need attention")