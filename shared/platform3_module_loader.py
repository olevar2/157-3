"""
Platform3 Production Module Loader
Centralized module loading with path resolution
"""

import sys
import importlib.util
from pathlib import Path

class Platform3ModuleLoader:
    """Production-ready module loader for Platform3"""
    
    @staticmethod
    def setup_paths():
        """Setup all necessary paths for Platform3 modules"""
        project_root = Path(__file__).parent.parent
        
        paths_to_add = [
            project_root,
            project_root / "shared",
            project_root / "ai-platform",
            project_root / "ai-platform" / "ai-models" / "intelligent-agents",
            project_root / "ai-platform" / "ai-models" / "intelligent-agents" / "decision-master",
            project_root / "ai-platform" / "ai-models" / "intelligent-agents" / "execution-expert",
            project_root / "ai-platform" / "intelligent-agents" / "adaptive-strategy-generator",
            project_root / "ai-platform" / "coordination",
        ]
        
        for path in paths_to_add:
            if path.exists() and str(path) not in sys.path:
                sys.path.insert(0, str(path))
    
    @staticmethod
    def load_decision_master():
        """Load DecisionMaster with proper path resolution"""
        Platform3ModuleLoader.setup_paths()
        
        try:
            decision_path = Path(__file__).parent.parent / "ai-platform" / "ai-models" / "intelligent-agents" / "decision-master" / "model.py"
            spec = importlib.util.spec_from_file_location("decision_master", decision_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"DecisionMaster load failed: {e}")
            return None
    
    @staticmethod
    def load_execution_expert():
        """Load ExecutionExpert with proper path resolution"""
        Platform3ModuleLoader.setup_paths()
        
        try:
            exec_path = Path(__file__).parent.parent / "ai-platform" / "ai-models" / "intelligent-agents" / "execution-expert" / "model.py"
            spec = importlib.util.spec_from_file_location("execution_expert", exec_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            print(f"ExecutionExpert load failed: {e}")
            return None

# Auto-setup paths when module is imported
Platform3ModuleLoader.setup_paths()
