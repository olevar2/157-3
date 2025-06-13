"""
File Cleanup Utility for Platform3 Refactoring
Identifies and removes outdated, temporary, and duplicate files
"""

import os
import shutil
import logging
from typing import List, Dict, Any
from datetime import datetime


class FileCleanupUtility:
    """Utility for cleaning up outdated and temporary files"""
    
    def __init__(self, base_path: str, dry_run: bool = True):
        self.base_path = base_path
        self.dry_run = dry_run
        self.logger = logging.getLogger(__name__)
        self.cleanup_report = {
            "timestamp": datetime.now().isoformat(),
            "base_path": base_path,
            "dry_run": dry_run,
            "files_processed": [],
            "directories_processed": [],
            "errors": []
        }
    
    def identify_mcp_files(self) -> List[str]:
        """Identify MCP-related files for removal"""
        mcp_files = []
        
        # Specific MCP files identified
        target_files = [
            "copilot_mcp_initializer.py",
            "MCP-COORDINATION-COMPLETE.md", 
            "MCP_JSON_ERROR_FIXED.md",
            "MCP_SERVER_STATUS.md"
        ]
        
        for filename in target_files:
            file_path = os.path.join(self.base_path, filename)
            if os.path.exists(file_path):
                mcp_files.append(file_path)
                
        return mcp_files
    
    def identify_backup_directories(self) -> List[str]:
        """Identify backup directories for removal"""
        backup_dirs = []
        
        target_dirs = [
            "backup_complete_files_final",
            "backup_consolidated_files", 
            "deduplication_backups",
            "migration_backups",
            "registry_backup"
        ]
        
        for dirname in target_dirs:
            dir_path = os.path.join(self.base_path, dirname)
            if os.path.exists(dir_path) and os.path.isdir(dir_path):
                backup_dirs.append(dir_path)
                
        return backup_dirs
    
    def identify_temporary_files(self) -> List[str]:
        """Identify temporary and validation files for removal"""
        temp_files = []
        
        # Specific temporary files
        target_files = [
            "_copilot_validation_temp.py",
            "agent_indicator_implementation.log",
            "dynamic_loader_detailed.log", 
            "genius_models_24_7_test.log"
        ]
        
        for filename in target_files:
            file_path = os.path.join(self.base_path, filename)
            if os.path.exists(file_path):
                temp_files.append(file_path)
        
        # Files with temp/backup patterns
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if any(pattern in file.lower() for pattern in ['_temp', '_backup', '_old']):
                    temp_files.append(os.path.join(root, file))
                    
        return temp_files
    
    def identify_analysis_scripts(self) -> List[str]:
        """Identify analysis scripts to consolidate"""
        analysis_scripts = []
        
        target_patterns = [
            "analyze_",
            "check_",
            "compare_",
            "debug_",
            "find_",
            "identify_", 
            "verify_",
            "count_",
            "locate_"
        ]
        
        for root, dirs, files in os.walk(self.base_path):
            for file in files:
                if file.endswith('.py'):
                    for pattern in target_patterns:
                        if file.startswith(pattern):
                            analysis_scripts.append(os.path.join(root, file))
                            break
                            
        return analysis_scripts
    
    def identify_migration_files(self) -> List[str]:
        """Identify migration-related files for archival"""
        migration_files = []
        
        target_files = [
            "migration_script.py",
            "migration_validation_test.py", 
            "fix_drive_migration.py",
            "test_e_drive_migration.py",
            "verify_e_drive_migration.py",
            "setup_e_drive.py",
            "DRIVE_MIGRATION_COMPLETED.md"
        ]
        
        for filename in target_files:
            file_path = os.path.join(self.base_path, filename)
            if os.path.exists(file_path):
                migration_files.append(file_path)
                
        return migration_files
    
    def remove_file(self, file_path: str) -> bool:
        """Remove a single file"""
        try:
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would remove file: {file_path}")
                self.cleanup_report["files_processed"].append({
                    "path": file_path,
                    "action": "would_remove",
                    "type": "file"
                })
                return True
            else:
                os.remove(file_path)
                self.logger.info(f"Removed file: {file_path}")
                self.cleanup_report["files_processed"].append({
                    "path": file_path,
                    "action": "removed",
                    "type": "file"
                })
                return True
        except Exception as e:
            self.logger.error(f"Failed to remove {file_path}: {e}")
            self.cleanup_report["errors"].append({
                "path": file_path,
                "error": str(e),
                "action": "remove_file"
            })
            return False
    
    def remove_directory(self, dir_path: str) -> bool:
        """Remove a directory and all its contents"""
        try:
            if self.dry_run:
                self.logger.info(f"[DRY RUN] Would remove directory: {dir_path}")
                self.cleanup_report["directories_processed"].append({
                    "path": dir_path,
                    "action": "would_remove",
                    "type": "directory"
                })
                return True
            else:
                shutil.rmtree(dir_path)
                self.logger.info(f"Removed directory: {dir_path}")
                self.cleanup_report["directories_processed"].append({
                    "path": dir_path,
                    "action": "removed", 
                    "type": "directory"
                })
                return True
        except Exception as e:
            self.logger.error(f"Failed to remove {dir_path}: {e}")
            self.cleanup_report["errors"].append({
                "path": dir_path,
                "error": str(e),
                "action": "remove_directory"
            })
            return False
    
    def run_cleanup(self) -> Dict[str, Any]:
        """Run the complete cleanup process"""
        self.logger.info(f"Starting cleanup process (dry_run={self.dry_run})")
        
        # Remove MCP files
        mcp_files = self.identify_mcp_files()
        self.logger.info(f"Found {len(mcp_files)} MCP files to remove")
        for file_path in mcp_files:
            self.remove_file(file_path)
        
        # Remove backup directories
        backup_dirs = self.identify_backup_directories()
        self.logger.info(f"Found {len(backup_dirs)} backup directories to remove")
        for dir_path in backup_dirs:
            self.remove_directory(dir_path)
        
        # Remove temporary files  
        temp_files = self.identify_temporary_files()
        self.logger.info(f"Found {len(temp_files)} temporary files to remove")
        for file_path in temp_files:
            self.remove_file(file_path)
        
        # Archive migration files
        migration_files = self.identify_migration_files()
        self.logger.info(f"Found {len(migration_files)} migration files")
        
        # Update summary
        self.cleanup_report.update({
            "mcp_files_count": len(mcp_files),
            "backup_dirs_count": len(backup_dirs),
            "temp_files_count": len(temp_files),
            "migration_files_count": len(migration_files),
            "total_files_processed": len(self.cleanup_report["files_processed"]),
            "total_dirs_processed": len(self.cleanup_report["directories_processed"]),
            "errors_count": len(self.cleanup_report["errors"])
        })
        
        return self.cleanup_report


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Platform3 File Cleanup Utility")
    parser.add_argument("--base-path", default=".", help="Base path for cleanup")
    parser.add_argument("--execute", action="store_true", help="Execute cleanup (not dry run)")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    cleanup = FileCleanupUtility(
        base_path=args.base_path,
        dry_run=not args.execute
    )
    
    results = cleanup.run_cleanup()
    
    print("=== FILE CLEANUP RESULTS ===")
    print(f"Mode: {'DRY RUN' if results['dry_run'] else 'EXECUTE'}")
    print(f"Base path: {results['base_path']}")
    print(f"MCP files: {results['mcp_files_count']}")
    print(f"Backup directories: {results['backup_dirs_count']}")
    print(f"Temporary files: {results['temp_files_count']}")
    print(f"Migration files: {results['migration_files_count']}")
    print(f"Total files processed: {results['total_files_processed']}")
    print(f"Total directories processed: {results['total_dirs_processed']}")
    print(f"Errors: {results['errors_count']}")
    
    if results['errors_count'] > 0:
        print("\\nErrors encountered:")
        for error in results['errors']:
            print(f"  {error['path']}: {error['error']}")


if __name__ == "__main__":
    main()