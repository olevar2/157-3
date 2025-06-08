\
import json
import sys

def analyze_constructor_errors(results_file_path):
    print(f"Analyzing constructor errors from: {results_file_path}")
    try:
        with open(results_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Results file not found at {results_file_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {results_file_path}")
        return

    constructor_errors = []
    if 'results' in data:
        for indicator_name, result in data['results'].items():
            if result.get('status') == 'failed' and result.get('error_category') == 'constructor_signature':
                error_info = {
                    'name': indicator_name,
                    'error_message': result.get('error', 'N/A'),
                    'signature_info': result.get('signature_info', 'N/A'), # Assuming this might be present
                    'traceback': result.get('traceback', 'N/A') # Assuming this might be present
                }
                constructor_errors.append(error_info)
    
    if not constructor_errors:
        print("No constructor_signature errors found.")
        return

    print(f"\\nFound {len(constructor_errors)} indicators with constructor_signature errors:")
    for error in constructor_errors:
        print(f"\\nIndicator: {error['name']}")
        print(f"  Error Message: {error['error_message']}")
        if error['signature_info'] != 'N/A':
            print(f"  Signature Info: {error['signature_info']}")
        if error['traceback'] != 'N/A' and error['traceback'] is not None : # Check for None explicitly
            print(f"  Traceback: \\n{error['traceback']}")
        print("-" * 40)

    # Identify common patterns (manual analysis based on output for now)
    # This part will be done after observing the output.

if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_file_path = sys.argv[1]
        analyze_constructor_errors(results_file_path)
    else:
        print("Usage: python analyze_constructor_errors.py <path_to_results_json_file>")
        # Fallback for testing if needed, but ideally provide the path
        default_path = 'd:\\\\MD\\\\Platform3\\\\enhanced_validation_results_20250607_215752.json'
        print(f"No results file provided. Attempting to use default: {default_path}")
        analyze_constructor_errors(default_path)
