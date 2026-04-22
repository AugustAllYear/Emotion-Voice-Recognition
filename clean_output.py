import sys
import re

def clean_file(input_path, output_path):
    with open(input_path, 'r') as f:
        lines = f.readlines()

    cleaned = []
    for line in lines:
        # Remove inline comments (Python style)
        line = re.sub(r'#.*$', '', line)
        # Remove leading/trailing whitespace
        line = line.strip()
        if line:                     # skip empty lines
            cleaned.append(line)

    with open(output_path, 'w') as f:
        f.write('\n'.join(cleaned))

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python clean_output.py <input_file> <output_file>")
        sys.exit(1)
    clean_file(sys.argv[1], sys.argv[2])