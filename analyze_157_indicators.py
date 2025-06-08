import json

with open('complete_157_indicator_discovery_20250607_034303.json', 'r') as f:
    data = json.load(f)
    print(f'Total indicators discovered: {data["total_indicators_discovered"]}')
    print(f'Target indicators: {data["target_indicators"]}')
    print()
    
    total = 0
    all_indicators = []
    
    for category, indicators in data['indicators_by_category'].items():
        count = len(indicators)
        total += count
        print(f'{category}: {count} indicators')
        
        # Collect all indicator names
        for indicator in indicators:
            if isinstance(indicator, dict) and 'name' in indicator:
                all_indicators.append(indicator['name'])
    
    print(f'\nTotal counted: {total}')
    print(f'Unique indicator names: {len(set(all_indicators))}')
    
    # Save all indicator names to a file
    with open('all_157_indicators.txt', 'w') as f:
        for name in sorted(set(all_indicators)):
            f.write(f'{name}\n')
    
    print('All indicator names saved to all_157_indicators.txt')
