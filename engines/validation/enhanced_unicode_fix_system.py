# -*- coding: utf-8 -*-
"""
Enhanced Unicode Fix System for Platform3 Indicators
Addresses specific charmap codec issues and import path problems
"""

import os
import re
import sys
import logging
from pathlib import Path

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class EnhancedUnicodeFixSystem:
    def __init__(self, base_path=None):
        if base_path is None:
            base_path = str(Path(__file__).parent.parent.parent)
        self.base_path = base_path
        self.engines_path = os.path.join(base_path, "engines")
        self.fixed_files = 0
        self.unicode_fixes = 0
        self.import_fixes = 0
        self.encoding_fixes = 0
        
        # Enhanced Unicode character mapping
        self.unicode_replacements = {
            # Greek letters
            'β': 'beta',
            'α': 'alpha', 
            'γ': 'gamma',
            'δ': 'delta',
            'σ': 'sigma',
            'μ': 'mu',
            'π': 'pi',
            'λ': 'lambda',
            'θ': 'theta',
            'φ': 'phi',
            'ψ': 'psi',
            'ω': 'omega',
            'Δ': 'Delta',
            'Σ': 'Sigma',
            'Φ': 'Phi',
            'Ω': 'Omega',
            # Mathematical symbols
            '×': '*',
            '÷': '/',
            '±': '+/-',
            '≥': '>=',
            '≤': '<=',
            '≠': '!=',
            '≈': '~=',
            '∞': 'inf',
            '∑': 'sum',
            '∏': 'prod',
            '∫': 'integral',
            '√': 'sqrt',
            '²': '**2',
            '³': '**3',
            '°': 'deg',
            '→': '->',
            '←': '<-',
            '↑': '^',
            '↓': 'v',
            # Currency and symbols
            '€': 'EUR',
            '£': 'GBP',
            '¥': 'JPY',
            '₹': 'INR',
            '§': 'section',
            '©': '(c)',
            '®': '(r)',
            '™': '(tm)',
            # Quotation marks and dashes
            '"': '"',
            '"': '"',
            ''': "'",
            ''': "'",
            '–': '-',
            '—': '--',
            '…': '...',
            # Additional problematic characters
            'ü': 'u',
            'ä': 'a',
            'ö': 'o',
            'ß': 'ss',
            'ñ': 'n',
            'ç': 'c',
            'é': 'e',
            'è': 'e',
            'ê': 'e',
            'à': 'a',
            'á': 'a',
            'í': 'i',
            'ó': 'o',
            'ú': 'u'
        }
        
        # Import pattern fixes
        self.import_patterns = [
            (r'from\s+\.\.indicator_base', 'from shared.engines.indicator_base'),
            (r'from\s+\.\.\s*indicator_base', 'from shared.engines.indicator_base'),
            (r'from\s+\.indicator_base', 'from shared.engines.indicator_base'),
            (r'from\s+indicator_base', 'from shared.engines.indicator_base'),
            (r'from\s+shared\.engines\.indicator_base', 'from shared.engines.indicator_base'),
            (r'import\s+\.\.indicator_base', 'from shared.engines import indicator_base'),
            (r'from\s+shared\.engines\s+import', 'from shared.engines import'),
        ]

    def fix_unicode_characters(self, content):
        """Fix Unicode characters that cause charmap codec issues"""
        original_content = content
        
        for unicode_char, replacement in self.unicode_replacements.items():
            if unicode_char in content:
                content = content.replace(unicode_char, replacement)
                self.unicode_fixes += 1
                
        return content

    def fix_import_patterns(self, content):
        """Fix import path issues"""
        original_content = content
        
        for pattern, replacement in self.import_patterns:
            new_content = re.sub(pattern, replacement, content)
            if new_content != content:
                content = new_content
                self.import_fixes += 1
                
        return content

    def ensure_utf8_encoding(self, content):
        """Ensure UTF-8 encoding declaration is present"""
        if not content.startswith('# -*- coding: utf-8 -*-'):
            if content.startswith('#'):
                # Insert after first line if it's a shebang
                lines = content.split('\n', 1)
                if len(lines) > 1:
                    content = lines[0] + '\n# -*- coding: utf-8 -*-\n' + lines[1]
                else:
                    content = lines[0] + '\n# -*- coding: utf-8 -*-\n'
            else:
                content = '# -*- coding: utf-8 -*-\n' + content
            self.encoding_fixes += 1
            
        return content

    def fix_file(self, file_path):
        """Fix a single Python file"""
        try:
            # Read file with multiple encoding attempts
            content = None
            encodings = ['utf-8', 'latin1', 'cp1252', 'ascii']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
                    
            if content is None:
                print(f"[ERROR] Could not read {file_path} with any encoding")
                return False
                
            # Apply fixes
            original_content = content
            content = self.ensure_utf8_encoding(content)
            content = self.fix_unicode_characters(content)
            content = self.fix_import_patterns(content)
            
            # Write back with UTF-8 encoding
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.fixed_files += 1
                print(f"[SUCCESS] Fixed {os.path.basename(file_path)}")
                return True
            else:
                print(f"[SKIP] No changes needed for {os.path.basename(file_path)}")
                return True
                
        except Exception as e:
            print(f"[ERROR] Failed to fix {file_path}: {e}")
            return False

    def process_all_indicators(self):
        """Process all indicator files in all categories"""
        categories = [
            'momentum', 'trend', 'volume', 'volatility', 'pattern', 
            'statistical', 'fractal', 'elliott_wave', 'gann', 
            'fibonacci', 'cycle', 'divergence'
        ]
        
        total_files = 0
        successful_fixes = 0
        
        print("Enhanced Unicode Fix System - Processing All Indicators")
        print("=" * 60)
        
        for category in categories:
            category_path = os.path.join(self.engines_path, category)
            if not os.path.exists(category_path):
                print(f"[SKIP] Category {category} not found")
                continue
                
            print(f"\nProcessing {category} indicators...")
            
            for file in os.listdir(category_path):
                if file.endswith('.py') and not file.startswith('__'):
                    file_path = os.path.join(category_path, file)
                    total_files += 1
                    
                    if self.fix_file(file_path):
                        successful_fixes += 1
                        
        print("\n" + "=" * 60)
        print("ENHANCED UNICODE FIX SYSTEM RESULTS")
        print("=" * 60)
        print(f"Total Files Processed: {total_files}")
        print(f"Successful File Fixes: {successful_fixes}")
        print(f"Success Rate: {(successful_fixes/total_files*100):.1f}%")
        print(f"Unicode Character Fixes: {self.unicode_fixes}")
        print(f"Import Pattern Fixes: {self.import_fixes}")
        print(f"Encoding Declaration Fixes: {self.encoding_fixes}")
        print(f"Files Modified: {self.fixed_files}")
        
        return successful_fixes == total_files

if __name__ == "__main__":
    fix_system = EnhancedUnicodeFixSystem()
    success = fix_system.process_all_indicators()
    
    if success:
        print("\n[SUCCESS] All indicators fixed successfully!")
        print("Running verification audit...")
        
        # Run quick verification
        import subprocess
        try:
            result = subprocess.run([
                sys.executable, 
                "indicator_audit_system.py"
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                print("[SUCCESS] Verification audit completed successfully")
            else:
                print(f"[WARNING] Verification audit had issues: {result.stderr}")
        except Exception as e:
            print(f"[INFO] Could not run verification audit: {e}")
    else:
        print("\n[WARNING] Some indicators may still have issues")