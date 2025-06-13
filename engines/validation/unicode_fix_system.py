#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Platform3 Unicode Fix System
Humanitarian Forex Trading Platform - Critical Unicode Encoding Fixes
Target: 72 indicators with 0% success rate arrow_right >90% success rate
"""

import os
import sys
import codecs
import glob
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import logging
import time

# Configure UTF-8 environment
os.environ['PYTHONIOENCODING'] = 'utf-8'
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Setup logging with UTF-8 support
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs', 'unicode_fixes.log'), 
                          encoding='utf-8', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class UnicodeFixSystem:
    """
    Comprehensive Unicode fixing system for Platform3 indicators
    Fixes 'charmap' codec issues and ensures UTF-8 compliance
    """
    
    def __init__(self, platform_root: str = None):
        if platform_root is None:
            platform_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        self.platform_root = Path(platform_root)
        self.engines_path = self.platform_root / "engines"
        self.fixed_count = 0
        self.error_count = 0
        self.processed_files = []
        self.unicode_patterns = [
            # Common Unicode characters causing charmap issues
            ('beta', 'beta'),
            ('alpha', 'alpha'), 
            ('gamma', 'gamma'),
            ('delta', 'delta'),
            ('sigma', 'sigma'),
            ('mu', 'mu'),
            ('pi', 'pi'),
            ('lambda', 'lambda'),
            ('Delta', 'Delta'),
            ('Sigma', 'Sigma'),
            ('Sum', 'Sum'),
            ('Delta', 'Delta'),
            ('sqrt', 'sqrt'),
            ('infinity', 'infinity'),
            ('le', 'le'),
            ('ge', 'ge'),
            ('ne', 'ne'),
            ('plus_minus', 'plus_minus'),
            ('multiply', 'multiply'),
            ('divide', 'divide'),
            ('degree', 'degree'),
            ('trademark', 'trademark'),
            ('copyright', 'copyright'),
            ('registered', 'registered'),
            # Financial symbols
            ('EUR', 'EUR'),
            ('GBP', 'GBP'),
            ('JPY', 'JPY'),
            ('cent', 'cent'),
            ('₽', 'RUB'),
            ('₹', 'INR'),
            # Mathematical operators
            ('arrow_right', 'arrow_right'),
            ('arrow_left', 'arrow_left'),
            ('arrow_up', 'arrow_up'),
            ('arrow_down', 'arrow_down'),
            ('∫', 'integral'),
            ('∂', 'partial'),
            ('∇', 'nabla'),
            ('∈', 'in'),
            ('∉', 'not_in'),
            ('∅', 'empty_set'),
            ('∪', 'union'),
            ('∩', 'intersection'),
        ]
        
    def find_all_indicator_files(self) -> List[Path]:
        """Find all Python indicator files in engines directory"""
        indicator_files = []
        
        # Search all engine subdirectories
        engine_dirs = [
            'momentum', 'trend', 'volatility', 'volume', 'statistical',
            'pattern', 'fibonacci', 'elliott_wave', 'gann', 'fractal',
            'cycle', 'divergence', 'pivot', 'sentiment', 'performance',
            'ai_enhancement', 'ml_advanced', 'core_trend', 'core_momentum'
        ]
        
        for engine_dir in engine_dirs:
            engine_path = self.engines_path / engine_dir
            if engine_path.exists():
                # Find all .py files
                py_files = list(engine_path.glob("*.py"))
                indicator_files.extend(py_files)
                logger.info(f"Found {len(py_files)} files in {engine_dir}")
        
        logger.info(f"Total indicator files found: {len(indicator_files)}")
        return indicator_files
    
    def detect_encoding_issues(self, file_path: Path) -> Tuple[bool, str, List[str]]:
        """Detect encoding issues in a Python file"""
        issues = []
        encoding_used = 'unknown'
        has_issues = False
        
        try:
            # Try to read with different encodings
            encodings_to_try = ['utf-8', 'cp1252', 'latin1', 'ascii']
            
            for encoding in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    encoding_used = encoding
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if encoding_used == 'unknown':
                issues.append("Could not decode file with any standard encoding")
                has_issues = True
                return has_issues, encoding_used, issues
            
            # Check for problematic Unicode characters
            for unicode_char, replacement in self.unicode_patterns:
                if unicode_char in content:
                    issues.append(f"Found problematic Unicode character: {unicode_char}")
                    has_issues = True
            
            # Check for encoding declaration
            lines = content.split('\n')
            has_utf8_declaration = False
            for i, line in enumerate(lines[:3]):  # Check first 3 lines
                if '# -*- coding:' in line or '# coding:' in line or '#coding:' in line:
                    if 'utf-8' in line:
                        has_utf8_declaration = True
                    else:
                        issues.append(f"Non-UTF-8 encoding declaration: {line.strip()}")
                        has_issues = True
            
            if not has_utf8_declaration:
                issues.append("Missing UTF-8 encoding declaration")
                has_issues = True
                
        except Exception as e:
            issues.append(f"Error analyzing file: {str(e)}")
            has_issues = True
            
        return has_issues, encoding_used, issues
    
    def fix_unicode_issues(self, file_path: Path) -> bool:
        """Fix Unicode issues in a single file"""
        try:
            # Read file content with best available encoding
            content = None
            original_encoding = 'utf-8'
            
            for encoding in ['utf-8', 'cp1252', 'latin1']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    original_encoding = encoding
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue
            
            if content is None:
                logger.error(f"Could not read file: {file_path}")
                return False
            
            # Create backup
            backup_path = file_path.with_suffix('.py.unicode_backup')
            with open(backup_path, 'w', encoding='utf-8', errors='replace') as f:
                f.write(content)
            
            # Apply Unicode fixes
            fixed_content = content
            replacements_made = 0
            
            # Replace problematic Unicode characters
            for unicode_char, replacement in self.unicode_patterns:
                if unicode_char in fixed_content:
                    fixed_content = fixed_content.replace(unicode_char, replacement)
                    replacements_made += 1
                    logger.info(f"Replaced '{unicode_char}' with '{replacement}' in {file_path.name}")
            
            # Ensure UTF-8 encoding declaration
            lines = fixed_content.split('\n')
            has_utf8_declaration = False
            
            # Check if UTF-8 declaration exists
            for i, line in enumerate(lines[:3]):
                if '# -*- coding:' in line or '# coding:' in line or '#coding:' in line:
                    if 'utf-8' not in line:
                        lines[i] = '# -*- coding: utf-8 -*-'
                        replacements_made += 1
                    has_utf8_declaration = True
                    break
            
            # Add UTF-8 declaration if missing
            if not has_utf8_declaration:
                if lines[0].startswith('#!'):
                    # Insert after shebang
                    lines.insert(1, '# -*- coding: utf-8 -*-')
                else:
                    # Insert at beginning
                    lines.insert(0, '# -*- coding: utf-8 -*-')
                replacements_made += 1
            
            # Write fixed content with UTF-8 encoding
            fixed_content = '\n'.join(lines)
            with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(fixed_content)
            
            if replacements_made > 0:
                logger.info(f"Fixed {replacements_made} Unicode issues in {file_path.name}")
                self.fixed_count += 1
            
            return True
            
        except Exception as e:
            logger.error(f"Error fixing Unicode issues in {file_path}: {str(e)}")
            self.error_count += 1
            return False
    
    def process_file(self, file_path: Path) -> Dict:
        """Process a single indicator file"""
        result = {
            'file': str(file_path),
            'has_issues': False,
            'issues': [],
            'fixed': False,
            'encoding': 'unknown'
        }
        
        try:
            # Detect issues
            has_issues, encoding_used, issues = self.detect_encoding_issues(file_path)
            result.update({
                'has_issues': has_issues,
                'issues': issues,
                'encoding': encoding_used
            })
            
            # Fix issues if found
            if has_issues:
                result['fixed'] = self.fix_unicode_issues(file_path)
            else:
                result['fixed'] = True  # No issues to fix
                
            self.processed_files.append(result)
            
        except Exception as e:
            result['issues'].append(f"Processing error: {str(e)}")
            result['fixed'] = False
            self.error_count += 1
            
        return result
    
    def run_comprehensive_unicode_fixes(self, max_workers: int = 8) -> Dict:
        """Run comprehensive Unicode fixes on all indicators"""
        logger.info("=" * 60)
        logger.info("STARTING COMPREHENSIVE UNICODE FIX SYSTEM")
        logger.info("Platform3 Humanitarian Forex Trading Platform")
        logger.info("Target: 72 indicators arrow_right >90% success rate")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Find all indicator files
        indicator_files = self.find_all_indicator_files()
        
        if not indicator_files:
            logger.error("No indicator files found!")
            return {'success': False, 'message': 'No files found'}
        
        # Process files in parallel
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {
                executor.submit(self.process_file, file_path): file_path 
                for file_path in indicator_files
            }
            
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.get(timeout=30)
                    results.append(result)
                    
                    # Log progress
                    if len(results) % 10 == 0:
                        logger.info(f"Processed {len(results)}/{len(indicator_files)} files...")
                        
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {str(e)}")
                    results.append({
                        'file': str(file_path),
                        'has_issues': True,
                        'issues': [f"Processing timeout: {str(e)}"],
                        'fixed': False,
                        'encoding': 'unknown'
                    })
        
        # Generate comprehensive report
        total_files = len(results)
        files_with_issues = len([r for r in results if r['has_issues']])
        files_fixed = len([r for r in results if r['fixed']])
        success_rate = (files_fixed / total_files) * 100 if total_files > 0 else 0
        
        elapsed_time = time.time() - start_time
        
        # Detailed reporting
        logger.info("=" * 60)
        logger.info("UNICODE FIX SYSTEM RESULTS")
        logger.info("=" * 60)
        logger.info(f"Total Files Processed: {total_files}")
        logger.info(f"Files with Unicode Issues: {files_with_issues}")
        logger.info(f"Files Successfully Fixed: {files_fixed}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Processing Time: {elapsed_time:.2f} seconds")
        logger.info("=" * 60)
        
        # Issue breakdown
        issue_summary = {}
        for result in results:
            for issue in result['issues']:
                issue_key = issue.split(':')[0] if ':' in issue else issue
                issue_summary[issue_key] = issue_summary.get(issue_key, 0) + 1
        
        if issue_summary:
            logger.info("ISSUE BREAKDOWN:")
            for issue, count in sorted(issue_summary.items(), key=lambda x: x[1], reverse=True):
                logger.info(f"  {issue}: {count} files")
        
        return {
            'success': success_rate >= 90.0,
            'total_files': total_files,
            'files_with_issues': files_with_issues,
            'files_fixed': files_fixed,
            'success_rate': success_rate,
            'processing_time': elapsed_time,
            'issue_summary': issue_summary,
            'detailed_results': results
        }

def main():
    """Main execution function"""
    unicode_fixer = UnicodeFixSystem()
    results = unicode_fixer.run_comprehensive_unicode_fixes()
    
    if results['success']:
        print(f"\n✅ SUCCESS: Unicode fixes completed with {results['success_rate']:.1f}% success rate!")
        print("Platform3 indicators are now ready for AI-driven humanitarian trading!")
    else:
        print(f"\n❌ PARTIAL SUCCESS: {results['success_rate']:.1f}% success rate")
        print("Additional fixes may be required for optimal performance.")
    
    return results

if __name__ == "__main__":
    main()