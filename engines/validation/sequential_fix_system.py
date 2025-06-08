#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Platform3 Sequential Unicode Fix System
Humanitarian Forex Trading Platform - Critical Unicode & Import Fixes
Target: 93 indicators with 0% success rate arrow_right >90% success rate
"""

import os
import sys
import glob
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import time

# Configure UTF-8 environment
os.environ['PYTHONIOENCODING'] = 'utf-8'
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('D:\\MD\\Platform3\\logs\\sequential_fixes.log', 
                          encoding='utf-8', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class SequentialFixSystem:
    """
    Sequential Unicode and Import fixing system for Platform3 indicators
    Processes files one by one to avoid threading issues
    """
    
    def __init__(self, platform_root: str = "D:\\MD\\Platform3"):
        self.platform_root = Path(platform_root)
        self.engines_path = self.platform_root / "engines"
        self.fixed_count = 0
        self.error_count = 0
        self.results = []
        
        # Unicode replacement patterns
        self.unicode_replacements = [
            ('beta', 'beta'), ('alpha', 'alpha'), ('gamma', 'gamma'), ('delta', 'delta'),
            ('sigma', 'sigma'), ('mu', 'mu'), ('pi', 'pi'), ('lambda', 'lambda'),
            ('Delta', 'Delta'), ('Sigma', 'Sigma'), ('Sum', 'Sum'), ('Delta', 'Delta'),
            ('sqrt', 'sqrt'), ('infinity', 'infinity'), ('le', 'le'), ('ge', 'ge'),
            ('ne', 'ne'), ('plus_minus', 'plus_minus'), ('multiply', 'multiply'), ('divide', 'divide'),
            ('degree', 'degree'), ('trademark', 'trademark'), ('copyright', 'copyright'), ('registered', 'registered'),
            ('EUR', 'EUR'), ('GBP', 'GBP'), ('JPY', 'JPY'), ('cent', 'cent'),
            ('arrow_right', 'arrow_right'), ('arrow_left', 'arrow_left'), ('arrow_up', 'arrow_up'), ('arrow_down', 'arrow_down')
        ]
        
        # Import fix patterns
        self.import_patterns = [
            (r'from\s+\.\.indicator_base\s+import', 'from shared.engines.indicator_base import'),
            (r'from\s+\.\.\s+import\s+indicator_base', 'from shared.engines import indicator_base'),
            (r'from\s+\.indicator_base\s+import', 'from shared.engines.indicator_base import'),
            (r'from\s+\.\s+import\s+indicator_base', 'from shared.engines import indicator_base'),
            (r'import\s+\.\.indicator_base', 'import shared.engines.indicator_base'),
        ]
    
    def find_all_indicator_files(self) -> List[Path]:
        """Find all Python indicator files"""
        indicator_files = []
        
        for engine_dir in self.engines_path.iterdir():
            if engine_dir.is_dir() and not engine_dir.name.startswith('.'):
                py_files = list(engine_dir.glob("*.py"))
                indicator_files.extend(py_files)
                logger.info(f"Found {len(py_files)} files in {engine_dir.name}")
        
        logger.info(f"Total indicator files found: {len(indicator_files)}")
        return indicator_files
    
    def fix_single_file(self, file_path: Path) -> Dict:
        """Fix Unicode and import issues in a single file"""
        result = {
            'file': str(file_path),
            'unicode_fixes': 0,
            'import_fixes': 0,
            'encoding_fixed': False,
            'success': False,
            'errors': []
        }
        
        try:
            # Read file with multiple encoding attempts
            content = None
            original_encoding = None
            
            for encoding in ['utf-8', 'cp1252', 'latin1', 'ascii']:
                try:
                    with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                        content = f.read()
                    original_encoding = encoding
                    break
                except Exception:
                    continue
            
            if content is None:
                result['errors'].append("Could not read file with any encoding")
                return result
            
            # Create backup
            backup_path = file_path.with_suffix('.py.backup_sequential')
            try:
                with open(backup_path, 'w', encoding='utf-8', errors='replace') as f:
                    f.write(content)
            except Exception as e:
                logger.warning(f"Could not create backup for {file_path.name}: {e}")
            
            # Apply fixes
            modified_content = content
            
            # 1. Fix Unicode characters
            for unicode_char, replacement in self.unicode_replacements:
                if unicode_char in modified_content:
                    modified_content = modified_content.replace(unicode_char, replacement)
                    result['unicode_fixes'] += 1
                    logger.info(f"Replaced '{unicode_char}' with '{replacement}' in {file_path.name}")
            
            # 2. Fix relative imports
            for pattern, replacement in self.import_patterns:
                if re.search(pattern, modified_content):
                    modified_content = re.sub(pattern, replacement, modified_content)
                    result['import_fixes'] += 1
                    logger.info(f"Fixed import pattern in {file_path.name}")
            
            # 3. Ensure UTF-8 encoding declaration
            lines = modified_content.split('\n')
            has_utf8_declaration = False
            
            for i, line in enumerate(lines[:3]):
                if '# -*- coding:' in line or '# coding:' in line:
                    if 'utf-8' not in line:
                        lines[i] = '# -*- coding: utf-8 -*-'
                        result['encoding_fixed'] = True
                    has_utf8_declaration = True
                    break
            
            if not has_utf8_declaration:
                if lines and lines[0].startswith('#!'):
                    lines.insert(1, '# -*- coding: utf-8 -*-')
                else:
                    lines.insert(0, '# -*- coding: utf-8 -*-')
                result['encoding_fixed'] = True
            
            # Write fixed content
            final_content = '\n'.join(lines)
            with open(file_path, 'w', encoding='utf-8', newline='\n', errors='replace') as f:
                f.write(final_content)
            
            total_fixes = result['unicode_fixes'] + result['import_fixes'] + (1 if result['encoding_fixed'] else 0)
            result['success'] = True
            
            if total_fixes > 0:
                logger.info(f"Successfully applied {total_fixes} fixes to {file_path.name}")
                self.fixed_count += 1
            
        except Exception as e:
            result['errors'].append(f"Processing error: {str(e)}")
            logger.error(f"Error processing {file_path.name}: {str(e)}")
            self.error_count += 1
        
        return result
    
    def run_sequential_fixes(self) -> Dict:
        """Run sequential fixes on all indicators"""
        logger.info("=" * 60)
        logger.info("STARTING SEQUENTIAL UNICODE & IMPORT FIX SYSTEM")
        logger.info("Platform3 Humanitarian Forex Trading Platform")
        logger.info("Target: 93 indicators arrow_right >90% success rate")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Find all indicator files
        indicator_files = self.find_all_indicator_files()
        
        if not indicator_files:
            return {'success': False, 'message': 'No files found'}
        
        # Process files sequentially
        for i, file_path in enumerate(indicator_files, 1):
            logger.info(f"Processing {i}/{len(indicator_files)}: {file_path.name}")
            result = self.fix_single_file(file_path)
            self.results.append(result)
            
            # Progress update every 10 files
            if i % 10 == 0 or i == len(indicator_files):
                logger.info(f"Progress: {i}/{len(indicator_files)} files processed")
        
        # Generate report
        total_files = len(self.results)
        successful_fixes = len([r for r in self.results if r['success']])
        success_rate = (successful_fixes / total_files) * 100 if total_files > 0 else 0
        
        total_unicode_fixes = sum(r['unicode_fixes'] for r in self.results)
        total_import_fixes = sum(r['import_fixes'] for r in self.results)
        total_encoding_fixes = sum(1 for r in self.results if r['encoding_fixed'])
        
        elapsed_time = time.time() - start_time
        
        logger.info("=" * 60)
        logger.info("SEQUENTIAL FIX SYSTEM RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Files Processed: {total_files}")
        logger.info(f"Successful File Fixes: {successful_fixes}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Unicode Character Fixes: {total_unicode_fixes}")
        logger.info(f"Import Pattern Fixes: {total_import_fixes}")
        logger.info(f"Encoding Declaration Fixes: {total_encoding_fixes}")
        logger.info(f"Processing Time: {elapsed_time:.2f} seconds")
        logger.info("=" * 60)
        
        return {
            'success': success_rate >= 90.0,
            'total_files': total_files,
            'successful_fixes': successful_fixes,
            'success_rate': success_rate,
            'unicode_fixes': total_unicode_fixes,
            'import_fixes': total_import_fixes,
            'encoding_fixes': total_encoding_fixes,
            'processing_time': elapsed_time,
            'detailed_results': self.results
        }

def main():
    """Main execution function"""
    fixer = SequentialFixSystem()
    results = fixer.run_sequential_fixes()
    
    if results['success']:
        print(f"\n✅ SUCCESS: Sequential fixes completed with {results['success_rate']:.1f}% success rate!")
        print(f"Fixed {results['unicode_fixes']} Unicode issues, {results['import_fixes']} import issues")
        print("Platform3 indicators are now ready for AI-driven humanitarian trading!")
    else:
        print(f"\n⚠️  PARTIAL SUCCESS: {results['success_rate']:.1f}% success rate")
        print("Additional fixes may be required for optimal performance.")
    
    return results

if __name__ == "__main__":
    main()