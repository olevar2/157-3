#!/usr/bin/env python3
"""
Platform3 Code Deduplication Script
Removes duplicate AIModelPerformanceMonitor and EnhancedAIModelBase classes
and replaces them with imports from shared.ai_model_base
"""

import os
import re
from pathlib import Path
from typing import List, Set

class Platform3Deduplicator:
    """Removes duplicate code and adds proper imports"""
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.backup_dir = self.root_path / "deduplication_backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        self.processed_files = []
        self.errors = []
    
    def backup_file(self, file_path: Path) -> Path:
        """Create backup before modification"""
        backup_path = self.backup_dir / file_path.name
        counter = 1
        while backup_path.exists():
            backup_path = self.backup_dir / f"{file_path.stem}_{counter}{file_path.suffix}"
            counter += 1
        
        import shutil
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def find_duplicate_classes(self, content: str) -> Set[str]:
        """Find duplicate class definitions in content"""
        duplicates = set()
        
        # Look for AIModelPerformanceMonitor class definition
        if re.search(r'class AIModelPerformanceMonitor.*?:', content, re.DOTALL):
            duplicates.add('AIModelPerformanceMonitor')
        
        # Look for EnhancedAIModelBase class definition
        if re.search(r'class EnhancedAIModelBase.*?:', content, re.DOTALL):
            duplicates.add('EnhancedAIModelBase')
        
        return duplicates
    
    def extract_class_definition(self, content: str, class_name: str) -> str:
        """Extract full class definition from content"""
        pattern = rf'class {class_name}.*?(?=\nclass |\n\n\S|\Z)'
        match = re.search(pattern, content, re.DOTALL)
        return match.group(0) if match else ""
    
    def remove_duplicate_classes(self, content: str, duplicates: Set[str]) -> str:
        """Remove duplicate class definitions from content"""
        for class_name in duplicates:
            # Remove the entire class definition
            pattern = rf'class {class_name}.*?(?=\nclass |\n\n\S|\Z)'
            content = re.sub(pattern, '', content, flags=re.DOTALL)
        
        return content
    
    def add_proper_imports(self, content: str, duplicates: Set[str]) -> str:
        """Add proper imports for the removed classes"""
        if not duplicates:
            return content
        
        # Check if import already exists
        if 'from shared.ai_model_base import' in content:
            return content
        
        # Find where to insert the import
        import_line = "from shared.ai_model_base import " + ", ".join(sorted(duplicates))
        
        # Find the best place to insert - after other imports
        lines = content.split('\n')
        insert_index = 0
        
        # Find last import line
        for i, line in enumerate(lines):
            if line.strip().startswith(('import ', 'from ')):
                insert_index = i + 1
        
        # Insert the new import
        lines.insert(insert_index, import_line)
        
        return '\n'.join(lines)
    
    def process_file(self, file_path: Path) -> bool:
        """Process a single file to remove duplicates"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # Find duplicate classes
            duplicates = self.find_duplicate_classes(content)
            
            if not duplicates:
                return False  # No duplicates found
            
            print(f"  Found duplicates: {', '.join(duplicates)}")
            
            # Remove duplicate class definitions
            content = self.remove_duplicate_classes(content, duplicates)
            
            # Add proper imports
            content = self.add_proper_imports(content, duplicates)
            
            # Clean up extra whitespace
            content = re.sub(r'\n\n\n+', '\n\n', content)
            
            if content != original_content:
                # Create backup
                backup_path = self.backup_file(file_path)
                
                # Write cleaned content
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                self.processed_files.append(str(file_path))
                return True
            
            return False
            
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            self.errors.append(error_msg)
            print(f"  ERROR: {error_msg}")
            return False
    
    def find_files_with_duplicates(self) -> List[Path]:
        """Find all Python files containing duplicate classes"""
        files_with_duplicates = []
        
        for root, dirs, files in os.walk(self.root_path / "ai-platform"):
            # Skip backup and cache directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'migration_backups', 'deduplication_backups']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            duplicates = self.find_duplicate_classes(content)
                            if duplicates:
                                files_with_duplicates.append(file_path)
                    except Exception as e:
                        print(f"Warning: Could not read {file_path}: {e}")
        
        return files_with_duplicates
    
    def deduplicate(self) -> tuple[int, int]:
        """Run the deduplication process"""
        print("Starting Platform3 Code Deduplication...")
        print(f"Root path: {self.root_path}")
        print(f"Backup directory: {self.backup_dir}")
        
        # Find files with duplicates
        files_to_process = self.find_files_with_duplicates()
        print(f"Found {len(files_to_process)} files with duplicate classes")
        
        if not files_to_process:
            print("No duplicate classes found!")
            return 0, 0
        
        # Process each file
        successful = 0
        failed = 0
        
        for file_path in files_to_process:
            print(f"\nProcessing: {file_path.relative_to(self.root_path)}")
            if self.process_file(file_path):
                successful += 1
                print(f"  SUCCESS: Duplicates removed and imports added")
            else:
                failed += 1
                print(f"  SKIPPED: No changes needed")
        
        # Summary
        print(f"\nDeduplication Summary:")
        print(f"Successfully processed: {successful}")
        print(f"Failed or no changes: {failed}")
        print(f"Backup files created in: {self.backup_dir}")
        
        if self.errors:
            print(f"\nErrors encountered:")
            for error in self.errors:
                print(f"   {error}")
        
        return successful, failed

def main():
    """Main deduplication function"""
    root_path = "E:/MD/Platform3"
    
    deduplicator = Platform3Deduplicator(root_path)
    successful, failed = deduplicator.deduplicate()
    
    if successful > 0:
        print(f"\nCode deduplication completed! {successful} files updated.")
        print("All duplicate classes replaced with imports from shared.ai_model_base")
    else:
        print("\nNo deduplication needed!")

if __name__ == "__main__":
    main()