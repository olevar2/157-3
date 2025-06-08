import json
import sys # Added import

def analyze_errors(results_file_path): # Added argument
    print(f"Analyzing error results from: {results_file_path}") # Added log
    with open(results_file_path, 'r') as f: # Use argument
        data = json.load(f)
    
    # Group errors by category
    errors_by_category = {}
    for indicator_name, result in data['results'].items():
        if result['status'] == 'failed':
            category = result['error_category']
            if category not in errors_by_category:
                errors_by_category[category] = []
            errors_by_category[category].append({
                'indicator_name': indicator_name,
                'error_message': result['error'],
                'error_category': category
            })
    
    print("Error summary:")
    for category, items in errors_by_category.items():
        print(f"  {category}: {len(items)} errors")
    
    print("\nConstructor signature errors (first 10):")
    constructor_errors = errors_by_category.get('constructor_signature', [])
    for item in constructor_errors[:10]:
        print(f"  {item['indicator_name']}: {item['error_message']}")
    
    print("\nCalculation errors (first 10):")
    calc_errors = errors_by_category.get('calculation_error', [])
    for item in calc_errors[:10]:
        print(f"  {item['indicator_name']}: {item['error_message']}")
    
    print("\nImport errors:")
    import_errors = errors_by_category.get('missing_imports', [])
    for item in import_errors:
        print(f"  {item['indicator_name']}: {item['error_message']}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_file_path = sys.argv[1]
        analyze_errors(results_file_path)
    else:
        print("Usage: python analyze_errors.py <path_to_results_json_file>")
        # Fallback to default if no argument provided, for compatibility, but print a warning
        print("No results file provided, attempting to use default (may be stale): d:\\MD\\Platform3\\enhanced_validation_results_20250607_165713.json")
        analyze_errors('d:\\MD\\Platform3\\enhanced_validation_results_20250607_165713.json')
