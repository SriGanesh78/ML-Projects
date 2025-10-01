#!/usr/bin/env python3
"""
This script fixes Google Colab imports in Jupyter notebooks for local execution.
It replaces 'google.colab.patches.cv2_imshow' with a matplotlib-based alternative.
"""

import json
import sys

def fix_colab_imports(notebook_path):
    """Fix Colab-specific imports in a Jupyter notebook"""

    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    # Replacement code for cv2_imshow
    replacement_code = """# For local execution (replaces Google Colab's cv2_imshow)
import matplotlib.pyplot as plt

def cv2_imshow(image):
    \"\"\"Display images in Jupyter (replaces Colab's cv2_imshow)\"\"\"
    plt.figure(figsize=(12, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
"""

    # Process each cell
    modified = False
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            new_source = []

            for line in source:
                # Check if this line imports google.colab
                if 'from google.colab.patches import cv2_imshow' in line:
                    # Replace with our custom implementation
                    if not modified:  # Only add once
                        new_source.extend(replacement_code.split('\n'))
                        new_source.append('\n')
                        modified = True
                else:
                    new_source.append(line)

            cell['source'] = new_source

    # Write back the notebook
    if modified:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"✅ Fixed Google Colab imports in: {notebook_path}")
        return True
    else:
        print(f"ℹ️  No Colab imports found in: {notebook_path}")
        return False

if __name__ == '__main__':
    notebook_path = '/Users/ganeshkanagavel/Library/CloudStorage/GoogleDrive-kganesk@gmail.com/My Drive/Ganesh/Learning/AIProjects/DeepLearning5 - RealTimeObject Detection/Object_Detection_with_YOLO.ipynb'

    try:
        success = fix_colab_imports(notebook_path)
        if success:
            print("\n✨ Your notebook has been fixed!")
            print("You can now run it without the 'google.colab' error.")
            print("\nNote: Make sure you have opencv-python and matplotlib installed:")
            print("  pip install opencv-python matplotlib")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

