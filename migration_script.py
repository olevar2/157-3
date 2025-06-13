#!/usr/bin/env python3
"""
Platform3 Migration Script - Fix This script converts all using the new package structure.
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple

class Platform3Migrator:
    """Migrates Platform3 codebase to proper package structure"""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.backup_dir = self.root_path / "migration_backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Import mapping from old         self.import_mappings = {
            # Platform3 logging
            "from shared.logging.platform3_logger import": "from shared.logging.platform3_logger import",
            "from shared.logging.platform3_logger import": "from shared.logging.platform3_logger import",
            
            # Error handling
            "from shared.error_handling.platform3_error_system import": "from shared.error_handling.platform3_error_system import",
            
            # Database
            "from shared.database.platform3_database_manager import": "from shared.database.platform3_database_manager import",
            
            # Communication
            "from shared.communication.platform3_communication_framework import": "from shared.communication.platform3_communication_framework import",
            
            # AI Platform components
            "from ai_platform.ai_platform_manager import": "from ai_platform.ai_platform_manager import",
            "from ai_platform.ai_services.model_registry import": "from ai_platform.ai_services.model_registry import",
            
            # Shared components
            "from shared.logging.platform3_logger import": "from shared.logging.platform3_logger import",
            "from shared.platform3_module_loader import": "from shared.platform3_module_loader import",
        }
        
        # Files processed
        self.processed_files = []
        self.errors = []
    
    def backup_file(self, file_path: Path) -> Path:
        """Create backup of file before modification"""
        backup_path = self.backup_dir / file_path.name
        counter = 1
        while backup_path.exists():
            backup_path = self.backup_dir / f"{file_path.stem}_{counter}{file_path.suffix}"
            counter += 1
        
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def fix_imports_in_file(self, file_path: Path) -> bool:
        """Fix         try:
            # Read file content
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Remove             content = re.sub(r'sys\.path\.append.*?\n', '', content)
            
            # Fix import statements using our mapping
            for old_import, new_import in self.import_mappings.items():
                content = content.replace(old_import, new_import)
            
            # Add proper sys import if it's used elsewhere but not imported
            if 'sys.' in content and 'import sys' not in content:
                # Find the first import and add sys import there
                import_match = re.search(r'^(import |from )', content, re.MULTILINE)
                if import_match:
                    content = content[:import_match.start()] + "import sys\n" + content[import_match.start():]
            
            # Only write if content changed
            if content != original_content:
                # Create backup first
                backup_path = self.backup_file(file_path)
                print(f"Backup created: {backup_path}")
                
                # Write fixed content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.processed_files.append(str(file_path))
                return True
            
            return False
            
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            self.errors.append(error_msg)
            print(f"ERROR: {error_msg}")
            return False
    
    def find_python_files_with_syspath(self) -> List[Path]:
        """Find all Python files that contain         python_files = []
        
        for root, dirs, files in os.walk(self.root_path):
            # Skip backup and cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'migration_backups']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if '                                python_files.append(file_path)
                    except Exception as e:
                        print(f"Warning: Could not read {file_path}: {e}")
        
        return python_files
    
    def migrate(self) -> Tuple[int, int]:
        """Run the full migration process"""
        print("Starting Platform3 Migration...")
        print(f"Root path: {self.root_path}")
        print(f"Backup directory: {self.backup_dir}")
        
        # Find files to process
        files_to_fix = self.find_python_files_with_syspath()
        print(f"Found {len(files_to_fix)} files with         
        if not files_to_fix:
            print("No files need migration!")
            return 0, 0
        
        # Process each file
        successful = 0
        failed = 0
        
        for file_path in files_to_fix:
            print(f"\nProcessing: {file_path.relative_to(self.root_path)}")
            if self.fix_imports_in_file(file_path):
                successful += 1
                print(f"Fixed successfully")
            else:
                failed += 1
                print(f"No changes needed or failed")
        
        # Summary
        print(f"\nMigration Summary:")
        print(f"Successfully processed: {successful}")
        print(f"Failed or no changes: {failed}")
        print(f"Backup files created in: {self.backup_dir}")
        
        if self.errors:
            print(f"\nErrors encountered:")
            for error in self.errors:
                print(f"   {error}")
        
        return successful, failed

def main():
    """Main migration function"""
    root_path = "E:/MD/Platform3"
    
    migrator = Platform3Migrator(root_path)
    successful, failed = migrator.migrate()
    
    if successful > 0:
        print(f"\nMigration completed! {successful} files updated.")
        print("Next steps:")
        print("1. Install the package: pip install -e .")
        print("2. Run tests to verify everything works")
        print("3. Update imports in any remaining files if needed")
    else:
        print("\nNo migration needed - package structure looks good!")

if __name__ == "__main__":
    main()