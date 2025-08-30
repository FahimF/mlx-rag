#!/usr/bin/env python3
"""
Script to install tree-sitter language parsers for multi-language RAG support.

This script helps users install the necessary tree-sitter language parsers
to enable RAG processing for various programming languages.
"""

import subprocess
import sys
import importlib
from typing import Dict, List, Tuple

# Language parser packages and their import names
LANGUAGE_PACKAGES = {
    'tree-sitter-python': 'tree_sitter_python',
    'tree-sitter-javascript': 'tree_sitter_javascript', 
    'tree-sitter-typescript': 'tree_sitter_typescript',
    'tree-sitter-java': 'tree_sitter_java',
    'tree-sitter-cpp': 'tree_sitter_cpp',
    'tree-sitter-c': 'tree_sitter_c',
    'tree-sitter-go': 'tree_sitter_go',
    'tree-sitter-rust': 'tree_sitter_rust',
    'tree-sitter-bash': 'tree_sitter_bash',
}

# Language descriptions
LANGUAGE_DESCRIPTIONS = {
    'tree-sitter-python': 'Python (.py)',
    'tree-sitter-javascript': 'JavaScript (.js, .jsx)',
    'tree-sitter-typescript': 'TypeScript (.ts, .tsx)',
    'tree-sitter-java': 'Java (.java)',
    'tree-sitter-cpp': 'C++ (.cpp, .cc, .cxx, .hpp)',
    'tree-sitter-c': 'C (.c, .h)',
    'tree-sitter-go': 'Go (.go)',
    'tree-sitter-rust': 'Rust (.rs)',
    'tree-sitter-bash': 'Bash/Shell (.sh, .bash)',
    'dart-flutter': 'Dart/Flutter (.dart) - Built-in support',
}

def check_package_installed(import_name: str) -> bool:
    """Check if a package is installed and importable."""
    try:
        importlib.import_module(import_name)
        return True
    except ImportError:
        return False

def install_package(package_name: str) -> Tuple[bool, str]:
    """Install a package using pip."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', package_name],
            capture_output=True,
            text=True,
            check=True
        )
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def get_package_status() -> Dict[str, Dict[str, any]]:
    """Get the installation status of all language packages."""
    status = {}
    
    for package_name, import_name in LANGUAGE_PACKAGES.items():
        is_installed = check_package_installed(import_name)
        status[package_name] = {
            'installed': is_installed,
            'import_name': import_name,
            'description': LANGUAGE_DESCRIPTIONS.get(package_name, package_name),
        }
    
    return status

def print_status():
    """Print the current status of all language parsers."""
    print("🔍 Multi-Language RAG Parser Status:")
    print("=" * 50)
    
    status = get_package_status()
    installed_count = 0
    total_count = len(status)
    
    for package_name, info in status.items():
        status_icon = "✅" if info['installed'] else "❌"
        status_text = "INSTALLED" if info['installed'] else "NOT INSTALLED"
        
        print(f"{status_icon} {info['description']:<25} {status_text}")
        
        if info['installed']:
            installed_count += 1
    
    # Add built-in language support that doesn't require separate packages
    print(f"🎯 {'Dart/Flutter (.dart)':<25} BUILT-IN SUPPORT")
    
    print("=" * 50)
    print(f"📊 Summary: {installed_count}/{total_count} parsers installed")
    print(f"🎯 Plus: Dart/Flutter with built-in pattern-based parsing")
    
    if installed_count < total_count:
        print(f"\n💡 Run with --install-missing to install missing parsers")
        print(f"ℹ️  Note: Dart/Flutter support is already built-in and ready to use!")
    else:
        print(f"\n🎉 All parsers are installed! Your RAG system supports all languages.")
    
    return status

def install_missing_packages():
    """Install all missing language parser packages."""
    print("🚀 Installing missing tree-sitter language parsers...")
    print("=" * 50)
    
    status = get_package_status()
    missing_packages = [
        package_name for package_name, info in status.items()
        if not info['installed']
    ]
    
    if not missing_packages:
        print("✅ All language parsers are already installed!")
        return
    
    print(f"📦 Found {len(missing_packages)} missing parsers:")
    for package in missing_packages:
        print(f"   - {package}")
    print()
    
    successful_installs = []
    failed_installs = []
    
    for package_name in missing_packages:
        print(f"📥 Installing {package_name}...")
        success, output = install_package(package_name)
        
        if success:
            print(f"✅ Successfully installed {package_name}")
            successful_installs.append(package_name)
        else:
            print(f"❌ Failed to install {package_name}")
            print(f"   Error: {output}")
            failed_installs.append(package_name)
    
    print("=" * 50)
    print(f"📊 Installation Summary:")
    print(f"   ✅ Successful: {len(successful_installs)}")
    print(f"   ❌ Failed: {len(failed_installs)}")
    
    if failed_installs:
        print(f"\n⚠️  Some installations failed:")
        for package in failed_installs:
            print(f"   - {package}")
        print(f"\nYou can try installing them manually:")
        for package in failed_installs:
            print(f"   pip install {package}")

def install_specific_languages(languages: List[str]):
    """Install parsers for specific languages."""
    print(f"🎯 Installing parsers for specific languages: {', '.join(languages)}")
    print("=" * 50)
    
    # Map language names to package names
    language_to_package = {
        'python': 'tree-sitter-python',
        'javascript': 'tree-sitter-javascript',
        'js': 'tree-sitter-javascript',
        'typescript': 'tree-sitter-typescript', 
        'ts': 'tree-sitter-typescript',
        'java': 'tree-sitter-java',
        'cpp': 'tree-sitter-cpp',
        'c++': 'tree-sitter-cpp',
        'c': 'tree-sitter-c',
        'go': 'tree-sitter-go',
        'rust': 'tree-sitter-rust',
        'bash': 'tree-sitter-bash',
        'shell': 'tree-sitter-bash',
        # Dart/Flutter - built-in pattern-based support
        'dart': 'dart-flutter-builtin',
        'flutter': 'dart-flutter-builtin',
    }
    
    packages_to_install = []
    builtin_languages = []
    invalid_languages = []
    
    for lang in languages:
        lang_lower = lang.lower()
        if lang_lower in language_to_package:
            package = language_to_package[lang_lower]
            if package == 'dart-flutter-builtin':
                builtin_languages.append(lang)
            elif package not in packages_to_install:
                packages_to_install.append(package)
        else:
            invalid_languages.append(lang)
    
    if invalid_languages:
        print(f"⚠️  Unknown languages: {', '.join(invalid_languages)}")
        print(f"Available languages: {', '.join(language_to_package.keys())}")
        print()
    
    # Handle built-in languages
    if builtin_languages:
        print(f"🎯 Built-in support (no installation needed): {', '.join(builtin_languages)}")
        print(f"   Dart/Flutter support is already available with pattern-based parsing.")
        print()
    
    if not packages_to_install:
        if builtin_languages:
            print("✅ All requested languages are supported (built-in or already installed)!")
        else:
            print("❌ No valid languages specified for installation.")
        return
    
    # Check which ones are already installed
    status = get_package_status()
    already_installed = []
    to_install = []
    
    for package in packages_to_install:
        if status.get(package, {}).get('installed', False):
            already_installed.append(package)
        else:
            to_install.append(package)
    
    if already_installed:
        print(f"✅ Already installed: {', '.join(already_installed)}")
    
    if not to_install:
        print("🎉 All requested parsers are already installed!")
        return
    
    print(f"📦 Installing: {', '.join(to_install)}")
    print()
    
    for package in to_install:
        print(f"📥 Installing {package}...")
        success, output = install_package(package)
        
        if success:
            print(f"✅ Successfully installed {package}")
        else:
            print(f"❌ Failed to install {package}")
            print(f"   Error: {output}")

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Install tree-sitter language parsers for multi-language RAG support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python install_language_parsers.py                    # Show current status
  python install_language_parsers.py --install-missing # Install all missing parsers
  python install_language_parsers.py --install python java rust  # Install specific parsers
  
Available Languages:
  python, javascript/js, typescript/ts, java, cpp/c++, c, go, rust, bash/shell
        """
    )
    
    parser.add_argument(
        '--install-missing',
        action='store_true',
        help='Install all missing language parsers'
    )
    
    parser.add_argument(
        '--install',
        nargs='+',
        metavar='LANGUAGE',
        help='Install parsers for specific languages'
    )
    
    args = parser.parse_args()
    
    if args.install_missing:
        install_missing_packages()
        print()
        print_status()
    elif args.install:
        install_specific_languages(args.install)
        print()
        print_status()
    else:
        print_status()

if __name__ == '__main__':
    main()
