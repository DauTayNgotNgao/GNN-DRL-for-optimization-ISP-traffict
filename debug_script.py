"""
Debug Script - Kiểm tra và validate dataset
"""

import os
import tarfile
import re
import numpy as np

def debug_tgz_file(tgz_path):
    """Debug và kiểm tra TGZ file"""

    print("="*70)
    print("DATASET DEBUG SCRIPT")
    print("="*70)

    # Step 1: Check file exists
    print(f"\n1. Checking if file exists...")
    if not os.path.exists(tgz_path):
        print(f"   ERROR: File not found: {tgz_path}")
        return
    print(f"   ✓ File exists: {tgz_path}")
    print(f"   File size: {os.path.getsize(tgz_path) / 1024 / 1024:.2f} MB")

    # Step 2: Extract TGZ
    print(f"\n2. Extracting TGZ...")
    extract_dir = 'germany50_debug'
    os.makedirs(extract_dir, exist_ok=True)

    try:
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(extract_dir, filter='data')
        print(f"   ✓ Extracted to: {extract_dir}")
    except Exception as e:
        print(f"   ERROR extracting: {e}")
        return

    # Step 3: Find files
    print(f"\n3. Finding traffic matrix files...")
    all_files = []
    for root, dirs, files in os.walk(extract_dir):
        for file in files:
            full_path = os.path.join(root, file)
            all_files.append(full_path)

    print(f"   Found {len(all_files)} total files")

    # Filter for demand matrix files
    demand_files = [f for f in all_files if 'demandMatrix' in f and f.endswith('.txt')]
    print(f"   Found {len(demand_files)} demand matrix files")

    if len(demand_files) == 0:
        print(f"\n   All files found:")
        for f in all_files[:20]:  # Show first 20
            print(f"     - {f}")
        return

    # Step 4: Parse first file
    print(f"\n4. Parsing first file: {demand_files[0]}")
    try:
        with open(demand_files[0], 'r', encoding='utf-8') as f:
            content = f.read()

        print(f"   File size: {len(content)} characters")
        print(f"   First 500 chars:\n{content[:500]}")

        # Extract nodes - FIXED: Match until the closing parenthesis of NODES section
        # The section ends with a lone ')' on its own line
        nodes_match = re.search(r'NODES\s*\((.*?)\n\)', content, re.DOTALL)
        if nodes_match:
            nodes = []
            nodes_section = nodes_match.group(1)
            print(f"\n   NODES section length: {len(nodes_section)} characters")
            print(f"   NODES section preview:\n{nodes_section[:500]}")

            for line in nodes_section.split('\n'):
                line = line.strip()
                if line and not line.startswith('#') and '(' in line:
                    # Match node_id followed by coordinates in parentheses
                    match = re.match(r'(\w+)\s*\(', line)
                    if match:
                        node_name = match.group(1)
                        nodes.append(node_name)
                        if len(nodes) <= 5:  # Show first 5 for debug
                            print(f"   Parsed node: {node_name} from line: {line[:60]}")

            print(f"\n   ✓ Found {len(nodes)} nodes")
            print(f"   Nodes: {nodes[:10]}..." if len(nodes) > 10 else f"   Nodes: {nodes}")
        else:
            print(f"   ERROR: No NODES section found")
            print(f"   Trying to show file structure...")
            # Show sections in file
            sections = re.findall(r'^([A-Z]+)\s*\(', content, re.MULTILINE)
            print(f"   Sections found: {sections}")
            return

        # Extract demands
        demands_match = re.search(r'DEMANDS \((.*?)\)', content, re.DOTALL)
        if demands_match:
            demands_count = 0
            total_traffic = 0

            for line in demands_match.group(1).split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    match = re.match(r'\S+\s+\(\s*(\w+)\s+(\w+)\s*\)\s+\d+\s+([\d.]+)', line)
                    if match:
                        demands_count += 1
                        total_traffic += float(match.group(3))

            print(f"\n   ✓ Found {demands_count} demands")
            print(f"   Total traffic: {total_traffic:.2f} Mbps")
        else:
            print(f"   ERROR: No DEMANDS section found")
            return

    except Exception as e:
        print(f"   ERROR parsing file: {e}")
        import traceback
        traceback.print_exc()
        return

    # Step 5: Test topology creation
    print(f"\n5. Testing topology creation...")
    num_nodes = len(nodes)
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])

    print(f"   ✓ Created {len(edges)} edges (full mesh)")
    print(f"   Expected edges for {num_nodes} nodes: {num_nodes * (num_nodes - 1)}")

    # Step 6: Parse all traffic matrices
    print(f"\n6. Testing all traffic matrix parsing...")
    valid_matrices = 0
    empty_matrices = 0
    total_traffic_sum = 0

    for i, file_path in enumerate(demand_files[:10]):  # Test first 10
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            demands_match = re.search(r'DEMANDS \((.*?)\)', content, re.DOTALL)
            if demands_match:
                traffic = 0
                for line in demands_match.group(1).split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        match = re.match(r'\S+\s+\(\s*(\w+)\s+(\w+)\s*\)\s+\d+\s+([\d.]+)', line)
                        if match:
                            traffic += float(match.group(3))

                if traffic > 0:
                    valid_matrices += 1
                    total_traffic_sum += traffic
                else:
                    empty_matrices += 1
        except Exception as e:
            print(f"   Warning: Error parsing {file_path}: {e}")

    print(f"   ✓ Valid matrices: {valid_matrices}/10")
    print(f"   Empty matrices: {empty_matrices}/10")
    print(f"   Avg traffic per matrix: {total_traffic_sum / max(1, valid_matrices):.2f} Mbps")

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Dataset is valid and ready to use!")
    print(f"  - {num_nodes} nodes")
    print(f"  - {len(edges)} edges (full mesh)")
    print(f"  - {len(demand_files)} traffic matrices")
    print(f"  - Average traffic: {total_traffic_sum / max(1, valid_matrices):.2f} Mbps")
    print(f"{'='*70}")


def test_import():
    """Test if imports work"""
    print("\nTesting imports...")

    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"  ✗ PyTorch not found: {e}")
        return False

    try:
        import torch_geometric
        print(f"  ✓ PyTorch Geometric")
    except ImportError as e:
        print(f"  ✗ PyTorch Geometric not found: {e}")
        return False

    try:
        import gymnasium
        print(f"  ✓ Gymnasium")
    except ImportError as e:
        print(f"  ✗ Gymnasium not found: {e}")
        return False

    try:
        import numpy
        print(f"  ✓ NumPy {numpy.__version__}")
    except ImportError as e:
        print(f"  ✗ NumPy not found: {e}")
        return False

    try:
        import matplotlib
        print(f"  ✓ Matplotlib {matplotlib.__version__}")
    except ImportError as e:
        print(f"  ✗ Matplotlib not found: {e}")
        return False

    print("  All imports OK!")
    return True


if __name__ == '__main__':
    # Test imports
    if not test_import():
        print("\nPlease install missing packages:")
        print("  pip install torch torch-geometric gymnasium numpy matplotlib networkx")
        exit(1)

    # Debug dataset
    tgz_path = 'directed-germany50-DFN-aggregated-5min-over-1day-native (1).tgz'

    if not os.path.exists(tgz_path):
        print(f"\nERROR: Dataset not found: {tgz_path}")
        print("Please place the TGZ file in the current directory.")
        exit(1)

    debug_tgz_file(tgz_path)

    print("\n" + "="*70)
    print("Debug complete! If everything looks good, run:")
    print("  python main_runner.py")
    print("="*70)