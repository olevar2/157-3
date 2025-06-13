"""
Platform3 Indicator Mapping Integrator
This script reads the GENIUS_AGENT_INDICATOR_MAPPING.md file and integrates
the indicator assignments into the Knowledge Graph MCP server.
"""

import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add required paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Regular expressions for parsing the markdown file
AGENT_SECTION_PATTERN = r'###\s+\d+\.\s+([\w\s]+)\s+.*=\s+(\d+)\s+Indicators'
INDICATOR_SECTION_PATTERN = r'\*\*([\w\s&]+)\s+\((\d+)\)\*\*:\s+(.*)'

def parse_indicator_mapping(markdown_path: str) -> Dict[str, Dict[str, List[str]]]:
    """
    Parse the GENIUS_AGENT_INDICATOR_MAPPING.md file to extract agent-indicator mappings.
    
    Args:
        markdown_path: Path to the markdown file
        
    Returns:
        Dict mapping agent names to their indicator categories and lists
    """
    with open(markdown_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Dictionary to store agent-indicator mappings
    agent_indicators = {}
    
    # Split content by sections
    sections = content.split('---')
    
    current_agent = None
    
    for section in sections:
        # Find agent section
        agent_match = re.search(AGENT_SECTION_PATTERN, section)
        if agent_match:
            agent_name = agent_match.group(1).strip()
            # Convert agent name to snake_case for consistency
            agent_key = agent_name.lower().replace(' ', '_')
            agent_indicators[agent_key] = {}
            current_agent = agent_key
            
            # Extract indicator categories
            indicator_matches = re.finditer(INDICATOR_SECTION_PATTERN, section)
            for indicator_match in indicator_matches:
                category = indicator_match.group(1).strip()
                count = indicator_match.group(2)
                indicators_text = indicator_match.group(3)
                
                # Convert category to snake_case
                category_key = category.lower().replace(' & ', '_').replace(' ', '_')
                
                # Extract individual indicators and convert to lowercase for registry matching
                indicators_raw = [ind.strip() for ind in indicators_text.split(',')]
                
                # Apply case conversion to match registry format
                indicators = []
                for ind in indicators_raw:
                    # Convert PascalCase to lowercase for registry matching
                    ind_lowercase = ind.lower()
                    indicators.append(ind_lowercase)
                agent_indicators[current_agent][category_key] = indicators
    
    return agent_indicators

def update_knowledge_graph(agent_indicators: Dict[str, Dict[str, List[str]]]):
    """
    Update the Knowledge Graph MCP server with the agent-indicator mappings.
    
    Args:
        agent_indicators: Dict mapping agent names to their indicator categories and lists
    """
    try:
        # Import MCP functions (these will be available in the VS Code environment)
        from mcp.knowledge_graph import (
            create_entities,
            create_relations,
            search_nodes,
            delete_entities
        )
        
        print("🧠 Updating Knowledge Graph with agent-indicator mappings...")
        
        # Create entities for each agent
        entities = []
        for agent_name, indicator_categories in agent_indicators.items():
            # Format agent name for display
            display_name = ' '.join(word.capitalize() for word in agent_name.split('_'))
            
            # Create agent entity
            agent_entity = {
                "name": f"Agent:{display_name}",
                "entityType": "GeniusAgent",
                "observations": [
                    f"Platform3 Genius Agent: {display_name}",
                    f"Responsible for specialized {display_name} analysis",
                    f"Uses {sum(len(inds) for inds in indicator_categories.values())} indicators"
                ]
            }
            entities.append(agent_entity)
            
            # Create entities for each indicator category
            for category, indicators in indicator_categories.items():
                category_display = ' '.join(word.capitalize() for word in category.split('_'))
                
                # Create indicator entities
                for indicator in indicators:
                    indicator_entity = {
                        "name": f"Indicator:{indicator}",
                        "entityType": "TechnicalIndicator",
                        "observations": [
                            f"Technical indicator: {indicator}",
                            f"Category: {category_display}",
                            f"Used by: {display_name}"
                        ]
                    }
                    entities.append(indicator_entity)
        
        # Create entities in Knowledge Graph
        try:
            # First check if entities already exist
            existing = search_nodes(query="GeniusAgent OR TechnicalIndicator")
            existing_names = [entity.get("name") for entity in existing.get("nodes", [])]
            
            # Filter out entities that already exist
            new_entities = [entity for entity in entities if entity["name"] not in existing_names]
            
            if new_entities:
                print(f"📝 Creating {len(new_entities)} new entities in Knowledge Graph...")
                create_entities(entities=new_entities)
            else:
                print("✅ All entities already exist in Knowledge Graph")
        except Exception as e:
            print(f"⚠️ Error creating entities: {e}")
        
        # Create relations between agents and indicators
        relations = []
        for agent_name, indicator_categories in agent_indicators.items():
            # Format agent name for display
            display_name = ' '.join(word.capitalize() for word in agent_name.split('_'))
            
            for category, indicators in indicator_categories.items():
                for indicator in indicators:
                    relations.append({
                        "from": f"Agent:{display_name}",
                        "to": f"Indicator:{indicator}",
                        "relationType": "uses"
                    })
        
        # Create relations in Knowledge Graph
        try:
            print(f"🔗 Creating {len(relations)} relations in Knowledge Graph...")
            create_relations(relations=relations)
        except Exception as e:
            print(f"⚠️ Error creating relations: {e}")
        
        print("✅ Knowledge Graph updated successfully!")
    except ImportError:
        # Fall back to MCP functions available in VS Code environment
        try:
            from mcp_knowledge_gra_create_entities import mcp_knowledge_gra_create_entities
            from mcp_knowledge_gra_create_relations import mcp_knowledge_gra_create_relations
            from mcp_knowledge_gra_search_nodes import mcp_knowledge_gra_search_nodes
            
            # Convert entities to MCP format
            entities_mcp = []
            for agent_name, indicator_categories in agent_indicators.items():
                display_name = ' '.join(word.capitalize() for word in agent_name.split('_'))
                
                # Create agent entity
                agent_entity = {
                    "name": f"Agent:{display_name}",
                    "entityType": "GeniusAgent",
                    "observations": [
                        f"Platform3 Genius Agent: {display_name}",
                        f"Responsible for specialized {display_name} analysis",
                        f"Uses {sum(len(inds) for inds in indicator_categories.values())} indicators"
                    ]
                }
                entities_mcp.append(agent_entity)
                
                # Create indicator entities
                for category, indicators in indicator_categories.items():
                    category_display = ' '.join(word.capitalize() for word in category.split('_'))
                    for indicator in indicators:
                        indicator_entity = {
                            "name": f"Indicator:{indicator}",
                            "entityType": "TechnicalIndicator",
                            "observations": [
                                f"Technical indicator: {indicator}",
                                f"Category: {category_display}",
                                f"Used by: {display_name}"
                            ]
                        }
                        entities_mcp.append(indicator_entity)
            
            # Create entities in Knowledge Graph
            print(f"📝 Creating {len(entities_mcp)} entities in Knowledge Graph...")
            mcp_knowledge_gra_create_entities(entities=entities_mcp)
            
            # Create relations
            relations_mcp = []
            for agent_name, indicator_categories in agent_indicators.items():
                display_name = ' '.join(word.capitalize() for word in agent_name.split('_'))
                
                for category, indicators in indicator_categories.items():
                    for indicator in indicators:
                        relations_mcp.append({
                            "from": f"Agent:{display_name}",
                            "to": f"Indicator:{indicator}",
                            "relationType": "uses"
                        })
            
            # Create relations in Knowledge Graph
            print(f"🔗 Creating {len(relations_mcp)} relations in Knowledge Graph...")
            mcp_knowledge_gra_create_relations(relations=relations_mcp)
            
            print("✅ Knowledge Graph updated successfully!")
        except Exception as e:
            print(f"⚠️ Failed to update Knowledge Graph: {e}")

def main():
    """Main function to execute the script"""
    markdown_path = os.path.join(os.path.dirname(__file__), "GENIUS_AGENT_INDICATOR_MAPPING.md")
    
    print(f"📊 Parsing indicator mapping from {markdown_path}...")
    agent_indicators = parse_indicator_mapping(markdown_path)
    
    print(f"🔍 Found {len(agent_indicators)} agents with indicator mappings")
    for agent, categories in agent_indicators.items():
        total_indicators = sum(len(inds) for inds in categories.values())
        print(f"  - {agent}: {total_indicators} indicators in {len(categories)} categories")
    
    # Update Knowledge Graph with the mappings
    update_knowledge_graph(agent_indicators)
    
    # Save the mappings to a python file for direct import
    output_path = os.path.join(os.path.dirname(__file__), "engines", "ai_enhancement", "indicator_mappings.py")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write('"""\nPlatform3 Indicator Mappings\nAutomatically generated from GENIUS_AGENT_INDICATOR_MAPPING.md\n"""\n\n')
        f.write(f'# Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write('AGENT_INDICATOR_MAPPINGS = {\n')
        
        for agent, categories in agent_indicators.items():
            f.write(f'    "{agent}": ' + '{\n')
            
            for category, indicators in categories.items():
                indicator_str = ', '.join(f'"{ind}"' for ind in indicators)
                f.write(f'        "{category}": [{indicator_str}],\n')
            
            f.write('    },\n')
        
        f.write('}\n')
    
    print(f"✅ Indicator mappings saved to {output_path}")
    
    print("\n🎯 Indicator mapping integration complete!")
    print("-" * 50)
    print("Next steps:")
    print("1. Ensure all platform system files use these indicator mappings")
    print("2. Run verify_agent_indicator_integration.py to validate the implementation")
    print("3. Update the user interface to show agent-indicator relationships")

if __name__ == "__main__":
    from datetime import datetime
    main()
