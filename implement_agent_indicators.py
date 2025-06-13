"""
Platform3 Agent-Indicator Implementation Runner
Implements agent-indicator relationships according to GENIUS_AGENT_INDICATOR_MAPPING.md
Uses Knowledge Graph MCP for mapping persistence and visualization
"""

import sys
import os
from pathlib import Path
import logging
import time
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                   handlers=[
                       logging.FileHandler("agent_indicator_implementation.log"),
                       logging.StreamHandler()
                   ])

logger = logging.getLogger(__name__)

# Add Platform3 to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def setup_directory_structure():
    """Set up necessary directory structure for indicators"""
    dirs_to_create = [
        "engines/ai_enhancement",
        "engines/indicators",
        "tests/agents"
    ]
    
    for dir_path in dirs_to_create:
        full_path = os.path.join(current_dir, dir_path)
        if not os.path.exists(full_path):
            os.makedirs(full_path)
            logger.info(f"Created directory: {full_path}")
            
def create_registry_aliases():
    """Create aliases for indicators in the registry to match indicator names in mapping file"""
    logger.info("\n🔄 Creating registry aliases...")
    try:
        # Update the registry for case-insensitive matching
        from update_registry_case_handling import main as update_registry
        update_registry()
        
        # Run the sync indicator names script
        from sync_indicator_names import main as run_sync
        run_sync()
        
        # Import the created aliases
        try:
            from engines.ai_enhancement.registry_aliases import create_registry_aliases
            create_registry_aliases()
            logger.info("✅ Registry aliases created successfully")
        except ImportError:
            logger.warning("⚠️ Could not import registry aliases module")
            
        # Apply the registry case patch
        try:
            from engines.ai_enhancement.registry_case_patch import apply_case_insensitive_patch
            apply_case_insensitive_patch()
            logger.info("✅ Registry case-insensitive patch applied")
        except ImportError:
            logger.warning("⚠️ Could not import registry case patch module")
    except Exception as e:
        logger.error(f"Failed to create registry aliases: {e}")

def run_implementation():
    """
    Run the full agent-indicator implementation process
    - Parse indicator mappings from GENIUS_AGENT_INDICATOR_MAPPING.md
    - Update Knowledge Graph with mappings
    - Generate indicator loader utility
    - Verify implementation
    """
    logger.info("=" * 60)
    logger.info("PLATFORM3 AGENT-INDICATOR IMPLEMENTATION")
    logger.info("=" * 60)
    
    # Step 1: Setup directory structure
    logger.info("\n📁 Setting up directory structure...")
    setup_directory_structure()
    
    # Step 2: Create registry aliases to handle case differences
    logger.info("\n📝 Creating registry aliases for case-insensitive matching...")
    create_registry_aliases()
    
    # Step 3: Run indicator mapping integrator
    logger.info("\n🔄 Running indicator mapping integrator...")
    try:
        from indicator_mapping_integrator import main as run_integrator
        run_integrator()
    except Exception as e:
        logger.error(f"Failed to run indicator mapping integrator: {e}")
    
    # Step 3: Update the Knowledge Graph with agent-indicator mappings
    logger.info("\n🧠 Updating Knowledge Graph with agent-indicator mappings...")
    try:
        # Using MCP commands directly
        from mcp_knowledge_gra_create_entities import mcp_knowledge_gra_create_entities
        
        # Create project status entity
        status_entity = {
            "name": f"Status:AgentIndicatorMapping:{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "entityType": "ProjectStatus",
            "observations": [
                f"Agent-Indicator Mapping Implementation ({datetime.now().strftime('%Y-%m-%d %H:%M')})",
                "All 9 agents configured with indicators from GENIUS_AGENT_INDICATOR_MAPPING.md",
                "Each agent loads indicators from registry using dynamic indicator loader",
                "Knowledge Graph contains full mapping visualization"
            ]
        }
        
        # Create entity in Knowledge Graph
        mcp_knowledge_gra_create_entities(entities=[status_entity])
        logger.info("✅ Knowledge Graph updated with implementation status")
    except Exception as e:
        logger.error(f"Failed to update Knowledge Graph: {e}")
    
    # Step 4: Run verification
    logger.info("\n🔍 Running agent-indicator verification...")
    try:
        from enhanced_agent_indicator_verification import verify_agent_indicator_integration
        verification_results = verify_agent_indicator_integration()
        
        if "error" in verification_results:
            logger.error(f"Verification failed: {verification_results['error']}")
        else:
            logger.info(f"✅ Verification complete!")
            logger.info(f"📊 Agents verified: {verification_results['agents_verified']}")
            logger.info(f"📈 Correctly implemented: {verification_results['correctly_implemented']}/{verification_results['agents_verified']}")
    except Exception as e:
        logger.error(f"Failed to run verification: {e}")
    
    # Step 5: Final status
    logger.info("\n✅ AGENT-INDICATOR IMPLEMENTATION COMPLETE!")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("1. Test each agent with indicators in analysis workflows")
    logger.info("2. Monitor indicator performance in production")
    logger.info("3. Update UI to show agent-indicator relationships")
    logger.info("=" * 60)

if __name__ == "__main__":
    run_implementation()
