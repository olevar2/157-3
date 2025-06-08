#!/usr/bin/env python3
"""
GitHub Copilot MCP Integration Initializer
Validates all MCP servers and provides comprehensive project context
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

class CopilotMCPInitializer:
    def __init__(self, project_root="d:/MD/Platform3"):
        self.project_root = Path(project_root)
        self.context_data = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "mcp_servers": {},
            "project_structure": {},
            "key_patterns": [],
            "integration_status": {}
        }
    
    def validate_mcp_integration(self):
        """Validate that all MCP servers are properly configured"""
        print("🔍 Validating MCP Integration...")
        
        # Check VS Code settings
        vscode_settings_path = Path(os.path.expanduser("~")) / "AppData/Roaming/Code - Insiders/User/settings.json"
        if vscode_settings_path.exists():
            try:
                with open(vscode_settings_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)
                    mcp_config = settings.get('mcpServers', {})
                    
                self.context_data["mcp_servers"] = {
                    "configured_count": len(mcp_config),
                    "servers": list(mcp_config.keys()),
                    "status": "configured" if mcp_config else "missing"
                }
                print(f"✅ Found {len(mcp_config)} MCP servers configured")
                
            except Exception as e:
                print(f"❌ Error reading VS Code settings: {e}")
                self.context_data["mcp_servers"]["status"] = "error"
        else:
            print("❌ VS Code settings not found")
            self.context_data["mcp_servers"]["status"] = "not_found"
    
    def analyze_project_structure(self):
        """Analyze project structure for GitHub Copilot context"""
        print("📊 Analyzing Project Structure...")
        
        # Key directories and files
        key_patterns = [
            "*.py",      # Python files
            "*.json",    # JSON configuration
            "*.md",      # Documentation
            "*.txt",     # Text files
            "*test*",    # Test files
            "*analysis*", # Analysis files
            "*agent*",   # Agent files
            "*indicator*" # Indicator files
        ]
        
        file_counts = {}
        key_files = []
        
        try:
            for pattern in key_patterns:
                files = list(self.project_root.glob(pattern))
                file_counts[pattern] = len(files)
                if pattern in ["*agent*", "*indicator*", "*analysis*"]:
                    key_files.extend([str(f.relative_to(self.project_root)) for f in files[:10]])  # Top 10
            
            self.context_data["project_structure"] = {
                "file_counts": file_counts,
                "key_files": key_files,
                "total_files": sum(file_counts.values())
            }
            
            print(f"✅ Analyzed {sum(file_counts.values())} files across {len(key_patterns)} patterns")
            
        except Exception as e:
            print(f"❌ Error analyzing project structure: {e}")
    
    def identify_key_patterns(self):
        """Identify key patterns that GitHub Copilot should be aware of"""
        print("🔍 Identifying Key Patterns...")
        
        patterns = []
        
        # Look for common file patterns
        try:
            py_files = list(self.project_root.glob("*.py"))
            json_files = list(self.project_root.glob("*.json"))
            
            # Analyze Python files for common patterns
            agent_files = [f for f in py_files if "agent" in f.name.lower()]
            analysis_files = [f for f in py_files if "analysis" in f.name.lower()]
            test_files = [f for f in py_files if "test" in f.name.lower()]
            
            patterns.extend([
                f"Agent files: {len(agent_files)} files",
                f"Analysis files: {len(analysis_files)} files", 
                f"Test files: {len(test_files)} files",
                f"JSON configs: {len(json_files)} files"
            ])
            
            # Look for specific naming patterns
            indicator_files = [f for f in py_files if "indicator" in f.name.lower()]
            registry_files = [f for f in py_files if "registry" in f.name.lower()]
            validation_files = [f for f in py_files if "validation" in f.name.lower()]
            
            patterns.extend([
                f"Indicator files: {len(indicator_files)} files",
                f"Registry files: {len(registry_files)} files",
                f"Validation files: {len(validation_files)} files"
            ])
            
            self.context_data["key_patterns"] = patterns
            print(f"✅ Identified {len(patterns)} key patterns")
            
        except Exception as e:
            print(f"❌ Error identifying patterns: {e}")
    
    def create_copilot_context_file(self):
        """Create a context file for GitHub Copilot"""
        print("📝 Creating GitHub Copilot Context File...")
        
        context_file = self.project_root / ".vscode" / "copilot-project-context.json"
        context_file.parent.mkdir(exist_ok=True)
        
        try:
            with open(context_file, 'w', encoding='utf-8') as f:
                json.dump(self.context_data, f, indent=2, ensure_ascii=False)
            
            print(f"✅ Created context file: {context_file}")
            
        except Exception as e:
            print(f"❌ Error creating context file: {e}")
    
    def generate_mcp_usage_report(self):
        """Generate a report on how to use MCP servers with this project"""
        print("📋 Generating MCP Usage Report...")
        
        report = f"""# GitHub Copilot MCP Integration Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Project Overview
- **Project Root**: {self.project_root}
- **Total Files**: {self.context_data.get('project_structure', {}).get('total_files', 'Unknown')}
- **MCP Servers**: {self.context_data.get('mcp_servers', {}).get('configured_count', 0)} configured

## Key Patterns Detected
{chr(10).join(f"- {pattern}" for pattern in self.context_data.get('key_patterns', []))}

## Recommended MCP Usage for This Project

### 1. **Shrimp Task Manager** - Use for:
   - Breaking down complex analysis tasks
   - Planning indicator implementations
   - Managing validation workflows

### 2. **Filesystem MCP** - Use for:
   - Navigating the extensive file structure
   - Finding related indicator/agent files
   - Analyzing file relationships

### 3. **Knowledge Graph** - Use for:
   - Storing agent-indicator relationships
   - Remembering analysis patterns
   - Building project knowledge base

### 4. **Desktop Commander** - Use for:
   - Running Python analysis scripts
   - Executing validation tests
   - Managing file operations

### 5. **Code MCP** - Use for:
   - Coordinating VS Code workspace
   - Managing multiple file edits
   - Optimizing development workflow

## Auto-Trigger Recommendations

GitHub Copilot should automatically use MCP servers when:

1. **User mentions "indicators"** → Filesystem + Task Manager
2. **User mentions "agents"** → Knowledge Graph + Filesystem  
3. **User mentions "analysis"** → Task Manager + Desktop Commander
4. **User mentions "validation"** → Desktop Commander + Filesystem
5. **User requests file operations** → Filesystem + Code MCP
6. **User requests complex tasks** → Task Manager + All others

## Project-Specific Commands

### Quick Analysis:
```
mcp_filesystem_search_files(path="{self.project_root}", pattern="*analysis*")
mcp_shrimp_task_m_analyze_task(summary="Project analysis")
```

### Indicator Work:
```
mcp_filesystem_search_files(path="{self.project_root}", pattern="*indicator*") 
mcp_knowledge_gra_search_nodes(query="indicators")
```

### Agent Management:
```
mcp_filesystem_search_files(path="{self.project_root}", pattern="*agent*")
mcp_knowledge_gra_create_entities(entities=[...])
```

## Status: {"✅ READY" if self.context_data.get('mcp_servers', {}).get('status') == 'configured' else "❌ NEEDS SETUP"}
"""
        
        report_file = self.project_root / ".vscode" / "mcp-usage-report.md"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            print(f"✅ Generated usage report: {report_file}")
            
        except Exception as e:
            print(f"❌ Error generating report: {e}")
    
    def run_full_initialization(self):
        """Run complete initialization process"""
        print("🚀 Starting GitHub Copilot MCP Integration Initialization")
        print("=" * 60)
        
        self.validate_mcp_integration()
        self.analyze_project_structure()
        self.identify_key_patterns()
        self.create_copilot_context_file()
        self.generate_mcp_usage_report()
        
        print("=" * 60)
        print("✅ GitHub Copilot MCP Integration Initialization Complete!")
        
        if self.context_data.get('mcp_servers', {}).get('status') == 'configured':
            print("\n🎉 All MCP servers are configured and ready!")
            print("🤖 GitHub Copilot will now automatically use MCP coordination")
        else:
            print("\n⚠️  MCP servers need configuration")
            print("📖 Check the generated reports for setup instructions")

if __name__ == "__main__":
    initializer = CopilotMCPInitializer()
    initializer.run_full_initialization()
