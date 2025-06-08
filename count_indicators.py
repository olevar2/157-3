import os

def count_indicator_files():
    count = 0
    for root, dirs, files in os.walk('engines'):
        for f in files:
            if f.endswith('.py') and not f.startswith('__'):
                count += 1
    return count

if __name__ == "__main__":
    print(f"Number of indicator files in engines directory: {count_indicator_files()}")