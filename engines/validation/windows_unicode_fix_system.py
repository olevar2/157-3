#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Windows-Compatible Unicode Fix System
Platform3 - Humanitarian Trading System

Specifically addresses Windows encoding issues with 'charmap' codec
and ensures proper UTF-8 handling for all indicator files.
"""

import os
import sys
import re
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set

class WindowsUnicodeFixer:
    """Windows-specific Unicode and import path fixer"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.engines_path = self.project_root / "engines"
        self.fixed_files = 0
        self.failed_files = 0
        self.encoding_fixes = 0
        self.import_fixes = 0
        
        # Comprehensive Unicode character mappings for Windows compatibility
        self.unicode_replacements = {
            # Greek letters commonly used in financial formulas
            'β': 'beta',
            'α': 'alpha', 
            'γ': 'gamma',
            'δ': 'delta',
            'σ': 'sigma',
            'μ': 'mu',
            'π': 'pi',
            'λ': 'lambda',
            'Δ': 'Delta',
            'Σ': 'Sigma',
            'Γ': 'Gamma',
            'Θ': 'Theta',
            'Φ': 'Phi',
            'Ψ': 'Psi',
            'Ω': 'Omega',
            
            # Mathematical symbols
            '×': '*',
            '±': '+/-',
            '÷': '/',
            '≈': '~=',
            '≠': '!=',
            '≤': '<=',
            '≥': '>=',
            '∞': 'inf',
            '∑': 'sum',
            '∏': 'prod',
            '∫': 'integral',
            '∂': 'partial',
            '∇': 'nabla',
            '√': 'sqrt',
            '∆': 'delta',
            
            # Arrows and special symbols
            '→': '->',
            '←': '<-',
            '↑': '^',
            '↓': 'v',
            '↔': '<->',
            '⇒': '=>',
            '⇐': '<=',
            '⇔': '<=>',
            
            # Degree and other symbols
            '°': 'deg',
            '′': "'",
            '″': '"',
            '‰': 'per_mille',
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY',
            '$': 'USD',
            
            # Quotation marks
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '„': '"',
            '‚': "'",
            
            # Dashes and spaces
            '—': '--',
            '–': '-',
            '…': '...',
            ' ': ' ',  # Non-breaking space
            '\u2009': ' ',  # Thin space
            '\u200a': ' ',  # Hair space
            '\u2028': '\n',  # Line separator
            '\u2029': '\n\n',  # Paragraph separator
        }
        
        # Import path fixes for Platform3 structure
        self.import_patterns = [
            # Shared engines imports
            (r'from\s+\.\.indicator_base\s+import', 'from shared.engines.indicator_base import'),
            (r'from\s+\.\.\.\s*shared\s*\.\s*engines\s+import', 'from shared.engines import'),
            (r'from\s+shared\.engines\s+import', 'from shared.engines.indicator_base import'),
            
            # Relative imports to absolute
            (r'from\s+\.\.\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+import', r'from engines.\1 import'),
            (r'from\s+\.\s*([a-zA-Z_][a-zA-Z0-9_]*)\s+import', r'from engines.\1 import'),
            
            # Framework imports
            (r'from\s+shared\.logging\s+import', 'from shared.logging.platform3_logger import'),
            (r'from\s+shared\.error_handling\s+import', 'from shared.error_handling.platform3_error_system import'),
            (r'from\s+shared\.database\s+import', 'from shared.database.platform3_database_manager import'),
            (r'from\s+shared\.communication\s+import', 'from shared.communication.platform3_communication_framework import'),
        ]
        
        print("Windows Unicode Fix System initialized")
        print(f"Project root: {self.project_root}")
        print(f"Engines path: {self.engines_path}")
    
    def detect_unicode_issues(self, file_path: Path) -> List[str]:
        """Detect Unicode characters that cause Windows encoding issues"""
        issues = []
        
        try:
            # Try reading with different encodings to detect issues
            encodings_to_try = ['utf-8', 'utf-8-sig', 'cp1252', 'latin1']
            
            for encoding in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    
                    # Check for problematic Unicode characters
                    for char, replacement in self.unicode_replacements.items():
                        if char in content:
                            issues.append(f"Unicode character '{char}' found (will replace with '{replacement}')")
                    
                    # Successfully read file
                    break
                    
                except UnicodeDecodeError as e:
                    issues.append(f"Encoding issue with {encoding}: {str(e)}")
                    continue
                    
        except Exception as e:
            issues.append(f"File access error: {str(e)}")
            
        return issues
    
    def fix_unicode_in_file(self, file_path: Path) -> bool:
        """Fix Unicode characters in a single file with Windows compatibility"""
        try:
            print(f"[INFO] Processing {file_path.name}...")
            
            # Try multiple encodings to read the file
            content = None
            original_encoding = None
            
            for encoding in ['utf-8', 'utf-8-sig', 'cp1252', 'latin1', 'iso-8859-1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    original_encoding = encoding
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                print(f"[ERROR] Could not read {file_path.name} with any encoding")
                self.failed_files += 1
                return False
            
            # Track changes
            original_content = content
            changes_made = 0
            
            # Fix Unicode characters
            for unicode_char, replacement in self.unicode_replacements.items():
                if unicode_char in content:
                    content = content.replace(unicode_char, replacement)
                    changes_made += 1
                    self.encoding_fixes += 1
                    print(f"  - Replaced '{unicode_char}' with '{replacement}'")
            
            # Fix import patterns
            for pattern, replacement in self.import_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    content = re.sub(pattern, replacement, content)
                    changes_made += len(matches)
                    self.import_fixes += 1
                    print(f"  - Fixed import pattern: {pattern}")
            
            # Ensure UTF-8 encoding declaration
            if '# -*- coding: utf-8 -*-' not in content:
                # Remove any existing encoding declarations
                content = re.sub(r'#.*?-\*-.*?coding[:=].*?-\*-.*?\n', '', content)
                
                # Add UTF-8 declaration after shebang if present
                lines = content.split('\n')
                if lines and lines[0].startswith('#!'):
                    lines.insert(1, '# -*- coding: utf-8 -*-')
                else:
                    lines.insert(0, '# -*- coding: utf-8 -*-')
                content = '\n'.join(lines)
                changes_made += 1
                print(f"  - Added UTF-8 encoding declaration")
            
            # Remove duplicate encoding declarations
            encoding_pattern = r'# -\*- coding: utf-8 -\*-'
            encoding_matches = re.findall(encoding_pattern, content)
            if len(encoding_matches) > 1:
                # Keep only the first occurrence
                content = re.sub(encoding_pattern, '', content)
                content = re.sub(r'^(#!/usr/bin/env python3\n)', r'\1# -*- coding: utf-8 -*-\n', content)
                changes_made += 1
                print(f"  - Removed duplicate encoding declarations")
            
            # Write file with explicit UTF-8 encoding
            if changes_made > 0:
                try:
                    with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                        f.write(content)
                    
                    print(f"[SUCCESS] Fixed {file_path.name} - {changes_made} changes made")
                    self.fixed_files += 1
                    return True
                    
                except Exception as e:
                    print(f"[ERROR] Failed to write {file_path.name}: {str(e)}")
                    self.failed_files += 1
                    return False
            else:
                print(f"[SKIP] No changes needed for {file_path.name}")
                return True
                
        except Exception as e:
            print(f"[ERROR] Exception processing {file_path.name}: {str(e)}")
            self.failed_files += 1
            return False
    
    def get_indicator_directories(self) -> List[Path]:
        """Get all indicator category directories"""
        directories = []
        
        try:
            for item in self.engines_path.iterdir():
                if item.is_dir() and not item.name.startswith('.') and not item.name.startswith('__'):
                    # Skip validation directory (our tools)
                    if item.name != 'validation':
                        directories.append(item)
                        
        except Exception as e:
            print(f"[ERROR] Failed to read engines directory: {str(e)}")
            
        return sorted(directories)
    
    def find_indicator_files(self, directory: Path) -> List[Path]:
        """Find all Python indicator files in a directory"""
        files = []
        
        try:
            for item in directory.iterdir():
                if item.is_file() and item.suffix == '.py' and not item.name.startswith('__'):
                    files.append(item)
                    
        except Exception as e:
            print(f"[ERROR] Failed to read directory {directory}: {str(e)}")
            
        return sorted(files)
    
    def fix_all_indicators(self) -> Dict[str, any]:
        """Fix Unicode and import issues in all indicator files"""
        start_time = time.time()
        
        print("\n" + "="*70)
        print("WINDOWS UNICODE FIX SYSTEM - STARTING COMPREHENSIVE FIX")
        print("="*70)
        
        # Reset counters
        self.fixed_files = 0
        self.failed_files = 0
        self.encoding_fixes = 0
        self.import_fixes = 0
        
        total_files = 0
        category_results = {}
        
        # Get all indicator directories
        directories = self.get_indicator_directories()
        
        print(f"Found {len(directories)} indicator categories")
        
        for directory in directories:
            print(f"\n[CATEGORY] Processing {directory.name}...")
            
            # Find all indicator files
            files = self.find_indicator_files(directory)
            print(f"  Found {len(files)} indicator files")
            
            category_fixes = 0
            category_failures = 0
            
            for file_path in files:
                total_files += 1
                
                if self.fix_unicode_in_file(file_path):
                    category_fixes += 1
                else:
                    category_failures += 1
            
            category_results[directory.name] = {
                'total': len(files),
                'fixed': category_fixes,
                'failed': category_failures,
                'success_rate': (category_fixes / len(files) * 100) if files else 0
            }
            
            print(f"  Category {directory.name}: {category_fixes}/{len(files)} files fixed ({(category_fixes/len(files)*100 if files else 0):.1f}%)")
        
        # Calculate final results
        end_time = time.time()
        duration = end_time - start_time
        
        overall_success_rate = (self.fixed_files / total_files * 100) if total_files > 0 else 0
        
        print("\n" + "="*70)
        print("WINDOWS UNICODE FIX SYSTEM - RESULTS")
        print("="*70)
        print(f"Total Files Processed: {total_files}")
        print(f"Successfully Fixed: {self.fixed_files}")
        print(f"Failed to Fix: {self.failed_files}")
        print(f"Overall Success Rate: {overall_success_rate:.1f}%")
        print(f"Unicode Character Fixes: {self.encoding_fixes}")
        print(f"Import Path Fixes: {self.import_fixes}")
        print(f"Processing Time: {duration:.2f} seconds")
        
        print(f"\nCategory Breakdown:")
        for category, results in category_results.items():
            print(f"  {category}: {results['fixed']}/{results['total']} ({results['success_rate']:.1f}%)")
        
        return {
            'total_files': total_files,
            'fixed_files': self.fixed_files,
            'failed_files': self.failed_files,
            'success_rate': overall_success_rate,
            'encoding_fixes': self.encoding_fixes,
            'import_fixes': self.import_fixes,
            'duration': duration,
            'category_results': category_results
        }

def main():
    """Main execution function"""
    fixer = WindowsUnicodeFixer()
    results = fixer.fix_all_indicators()
    
    if results['success_rate'] >= 95:
        print("\n[SUCCESS] Unicode fix system completed successfully!")
        print("All indicator files should now be compatible with Windows encoding")
    else:
        print(f"\n[WARNING] Fix system completed with {results['success_rate']:.1f}% success rate")
        print("Some files may still have issues - manual review recommended")
    
    return results

if __name__ == "__main__":
    main()