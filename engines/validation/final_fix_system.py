#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Final Fix System for Failed Files and AI Services Integration
Platform3 - Humanitarian Trading System
"""

import os
import sys
import re
import time
from pathlib import Path

def fix_specific_files():
    """Fix the 2 files that failed in the previous run"""
    project_root = Path(__file__).parent.parent.parent
    
    # Problematic files from previous run
    failed_files = [
        project_root / "engines" / "core_trend" / "ADX.py",
        project_root / "engines" / "core_trend" / "Ichimoku.py"
    ]
    
    print("FIXING SPECIFIC FAILED FILES")
    print("="*50)
    
    for file_path in failed_files:
        if not file_path.exists():
            print(f"[SKIP] File not found: {file_path}")
            continue
            
        print(f"[INFO] Processing {file_path.name}...")
        
        try:
            # Read file with binary mode to handle any encoding
            with open(file_path, 'rb') as f:
                content_bytes = f.read()
            
            # Try to decode with UTF-8, fall back to latin1
            try:
                content = content_bytes.decode('utf-8')
            except UnicodeDecodeError:
                content = content_bytes.decode('latin1')
            
            # Replace problematic Unicode characters
            unicode_replacements = {
                '\u2248': '~=',  # ≈ (approximately equal)
                '\u2260': '!=',  # ≠ (not equal)
                '\u2264': '<=',  # ≤ (less than or equal)
                '\u2265': '>=',  # ≥ (greater than or equal)
                '\u00b1': '+/-', # ± (plus-minus)
                '\u00d7': '*',   # × (multiplication)
                '\u00f7': '/',   # ÷ (division)
                '\u03b1': 'alpha',  # α
                '\u03b2': 'beta',   # β
                '\u03b3': 'gamma',  # γ
                '\u03b4': 'delta',  # δ
                '\u03c3': 'sigma',  # σ
                '\u03bc': 'mu',     # μ
                '\u03c0': 'pi',     # π
                '\u03bb': 'lambda', # λ
                '\u0394': 'Delta',  # Δ
                '\u03a3': 'Sigma',  # Σ
                '\u201c': '"',      # "
                '\u201d': '"',      # "
                '\u2018': "'",      # '
                '\u2019': "'",      # '
                '\u2013': '-',      # –
                '\u2014': '--',     # —
                '\u2026': '...',    # …
                '\u00a0': ' ',      # Non-breaking space
            }
            
            changes_made = 0
            for unicode_char, replacement in unicode_replacements.items():
                if unicode_char in content:
                    content = content.replace(unicode_char, replacement)
                    changes_made += 1
                    print(f"  - Replaced '{unicode_char}' with '{replacement}'")
            
            # Ensure UTF-8 encoding declaration
            if '# -*- coding: utf-8 -*-' not in content:
                lines = content.split('\n')
                if lines and lines[0].startswith('#!'):
                    lines.insert(1, '# -*- coding: utf-8 -*-')
                else:
                    lines.insert(0, '# -*- coding: utf-8 -*-')
                content = '\n'.join(lines)
                changes_made += 1
                print(f"  - Added UTF-8 encoding declaration")
            
            # Write file with UTF-8 encoding
            with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                f.write(content)
            
            print(f"[SUCCESS] Fixed {file_path.name} - {changes_made} changes made")
            
        except Exception as e:
            print(f"[ERROR] Failed to fix {file_path.name}: {str(e)}")

def fix_ai_services_import():
    """Fix AI services import path in the system"""
    project_root = Path(__file__).parent.parent.parent
    ai_platform_path = project_root / "ai-platform"
    
    print("\nFIXING AI SERVICES IMPORT")
    print("="*30)
    
    # Check if ai-platform directory exists
    if not ai_platform_path.exists():
        print(f"[ERROR] AI platform directory not found: {ai_platform_path}")
        return
    
    # Add __init__.py to ai-platform if missing
    ai_platform_init = ai_platform_path / "__init__.py"
    if not ai_platform_init.exists():
        with open(ai_platform_init, 'w', encoding='utf-8') as f:
            f.write('# -*- coding: utf-8 -*-\n"""AI Platform Module"""\n')
        print(f"[SUCCESS] Created {ai_platform_init}")
    
    # Check ai-services directory
    ai_services_path = ai_platform_path / "ai-services"
    if not ai_services_path.exists():
        print(f"[ERROR] AI services directory not found: {ai_services_path}")
        return
    
    # Update the audit system to use correct import path
    audit_system_path = project_root / "engines" / "validation" / "indicator_audit_system.py"
    
    if audit_system_path.exists():
        print(f"[INFO] Updating audit system imports...")
        
        with open(audit_system_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix AI services import
        old_import = "from ai_services import"
        new_import = "        
        if old_import in content and new_import not in content:
            content = content.replace(
                "try:\n    from ai_services import",
                f"try:\n    {new_import}"
            )
            
            with open(audit_system_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print(f"[SUCCESS] Updated AI services import in audit system")
        else:
            print(f"[INFO] AI services import already correct or not found")
    
    print(f"[INFO] AI platform structure:")
    for item in ai_platform_path.iterdir():
        print(f"  - {item.name}")

def run_quick_audit():
    """Run a quick audit to test the fixes"""
    print("\nRUNNING QUICK AUDIT TEST")
    print("="*25)
    
    import subprocess
    import sys
    
    try:
        # Change to project directory and run audit
        os.chdir(r"D:\MD\Platform3")
        result = subprocess.run([
            sys.executable, 
            "engines/validation/indicator_audit_system.py"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            # Extract key metrics from output
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'Success Rate:' in line or 'Total Indicators Tested:' in line:
                    print(f"[RESULT] {line.strip()}")
        else:
            print(f"[ERROR] Audit failed with return code {result.returncode}")
            if result.stderr:
                print(f"Error: {result.stderr[:200]}...")
                
    except subprocess.TimeoutExpired:
        print("[WARNING] Audit test timed out")
    except Exception as e:
        print(f"[ERROR] Failed to run audit test: {str(e)}")

def main():
    """Main execution function"""
    print("FINAL FIX SYSTEM FOR PLATFORM3 INDICATORS")
    print("="*50)
    
    start_time = time.time()
    
    # Fix specific failed files
    fix_specific_files()
    
    # Fix AI services import
    fix_ai_services_import()
    
    # Run quick audit test
    run_quick_audit()
    
    end_time = time.time()
    duration = end_time - start_time
    
    print(f"\nFINAL FIX SYSTEM COMPLETED in {duration:.2f} seconds")
    print("="*50)

if __name__ == "__main__":
    main()