#!/usr/bin/env python3
"""
Simple Web Service Classification Project Structure Creator
Creates basic directories with src/ packages only
"""

from pathlib import Path

def create_directories():
    """Create basic directory structure"""
    
    base_dir = Path("web_services_classification")
    
    # Simple directory list
    directories = [
        # Root
        "",
        
        # Data directories  
        "data/raw",
        "data/processed",
        "data/analysis", 
        "data/features",
        
        # Source packages
        "src",
        "src/preprocessing",
        "src/modeling",
        "src/evaluation", 
        "src/visualization",
        "src/utils",
        
        # High-level directories
        "models",
        "results",
        "logs",
        "docs"
    ]
    
    # Create directories
    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created: {dir_path}")
    
    return base_dir

def create_packages(base_dir):
    """Create Python packages in src/"""
    
    # Package init files
    packages = {
        "src/__init__.py": "",
        "src/preprocessing/__init__.py": "",
        "src/modeling/__init__.py": "",
        "src/evaluation/__init__.py": "",
        "src/visualization/__init__.py": "",
        "src/utils/__init__.py": ""
    }
    
    # Create package files
    for package_file in packages:
        file_path = base_dir / package_file
        file_path.touch()
        print(f"Created: {file_path}")

def main():
    """Create simple project structure"""
    
    print("Creating simple project structure...")
    
    # Create directories
    base_dir = create_directories()
    
    # Create packages
    create_packages(base_dir)
    
    print(f"\nProject created: {base_dir.absolute()}")
    print("\nStructure:")
    print("├── src/           (packages)")
    print("├── data/          (data files)")
    print("├── models/        (saved models)")
    print("├── results/       (outputs)")
    print("├── logs/          (log files)")
    print("└── docs/          (documentation)")

if __name__ == "__main__":
    main()