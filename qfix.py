#!/usr/bin/env python3
"""
Quick fix script for PyTorch 2.6+ weights_only and CPU compatibility
Run this in your project root directory: python quick_fix_pytorch26.py
"""

import os
import re
import sys

def fix_torch_load(content):
    """Replace all torch.load() calls to include map_location and weights_only=False"""
    
    # Pattern 1: torch.load(path, map_location=torch.device("cuda" if torch.cuda.is_available(, weights_only=False) else "cpu"), weights_only=False)
    pattern1 = r'torch\.load\(([^,\)]+)\)'
    replacement1 = r'torch.load(\1, map_location=torch.device("cuda" if torch.cuda.is_available(, weights_only=False) else "cpu"), weights_only=False)'
    content = re.sub(pattern1, replacement1, content)
    
    # Pattern 2: torch.load(path, map_location=..., weights_only=False) - add weights_only
    pattern2 = r'torch\.load\(([^,]+),\s*map_location=([^,\)]+)\)'
    replacement2 = r'torch.load(\1, map_location=\2, weights_only=False)'
    content = re.sub(pattern2, replacement2, content)
    
    return content

def fix_cuda_calls(content):
    """Replace .to(device) calls with .to(device)"""
    
    # Add device detection at the top if not present
    if 'device = torch.device' not in content and 'torch' in content:
        imports_section = content.find('import torch')
        if imports_section != -1:
            # Find end of imports
            next_line = content.find('\n\n', imports_section)
            if next_line != -1:
                device_code = '\n\n# Auto-added device detection\ndevice = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
                content = content[:next_line] + device_code + content[next_line:]
    
    # Replace .to(device) with .to(device)
    content = re.sub(r'\.cuda\(\)', '.to(device)', content)
    
    return content

def process_file(filepath):
    """Process a single Python file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Apply fixes
        if 'torch.load' in content:
            content = fix_torch_load(content)
            print(f"  ✅ Fixed torch.load() calls in {filepath}")
        
        if '.to(device)' in content:
            content = fix_cuda_calls(content)
            print(f"  ✅ Fixed .to(device) calls in {filepath}")
        
        # Only write if changes were made
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
    
    except Exception as e:
        print(f"  ❌ Error processing {filepath}: {e}")
        return False

def find_python_files(directory):
    """Recursively find all Python files"""
    python_files = []
    for root, dirs, files in os.walk(directory):
        # Skip virtual environments and cache directories
        dirs[:] = [d for d in dirs if d not in ['venv', 'env', '__pycache__', '.git', 'node_modules']]
        
        for file in files:
            if file.endswith('.py'):
                python_files.append(os.path.join(root, file))
    
    return python_files

def main():
    print("🔧 PyTorch 2.6+ Compatibility Fixer")
    print("=" * 50)
    
    # Get current directory
    current_dir = os.getcwd()
    print(f"📁 Working directory: {current_dir}\n")
    
    # Find all Python files
    print("🔍 Finding Python files...")
    python_files = find_python_files(current_dir)
    print(f"   Found {len(python_files)} Python files\n")
    
    # Process each file
    print("🛠️  Processing files...")
    fixed_count = 0
    
    for filepath in python_files:
        if process_file(filepath):
            fixed_count += 1
    
    print("\n" + "=" * 50)
    print(f"✨ Complete! Fixed {fixed_count} files")
    print("\n💡 Next steps:")
    print("   1. Test your application: python backend.py")
    print("   2. If you see any errors, check the modified files")
    print("   3. Consider committing these changes to version control")

if __name__ == "__main__":
    # Confirm before running
    print("⚠️  This script will modify Python files in your project.")
    response = input("Continue? (yes/no): ").lower().strip()
    
    if response in ['yes', 'y']:
        main()
    else:
        print("❌ Cancelled")
        sys.exit(0)

