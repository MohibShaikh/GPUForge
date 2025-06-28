#!/usr/bin/env python3
"""
GPU Environment Creator - Main Entry Point
Automatically detects GPU and creates compatible conda environments for ML/DL work
"""

import argparse
import sys
import subprocess
from gpu_detector import GPUDetector
from compatibility_finder import CompatibilityFinder
from env_generator import EnvironmentGenerator

def check_prerequisites():
    """Check if required tools are installed"""
    print("🔍 Checking prerequisites...")
    
    # Check conda
    try:
        result = subprocess.run(['conda', '--version'], capture_output=True, text=True, check=True)
        print(f"✅ Conda: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Conda not found. Please install Anaconda or Miniconda first.")
        print("   Download from: https://docs.conda.io/en/latest/miniconda.html")
        return False
    
    # Check NVIDIA drivers (optional for detection)
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, check=True)
        print("✅ NVIDIA drivers installed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("⚠️  nvidia-smi not found. GPU detection may be limited.")
        print("   Install NVIDIA drivers from: https://www.nvidia.com/drivers")
    
    return True

def main():
    parser = argparse.ArgumentParser(
        description="GPU Environment Creator - Automate GPU environment setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Create PyTorch environment with auto-detection
  %(prog)s --framework tensorflow   # Create TensorFlow environment
  %(prog)s --name my-env             # Custom environment name
  %(prog)s --detect-only             # Only detect GPU, don't create environment
  %(prog)s --list-compatible         # Show all compatible configurations
        """
    )
    
    parser.add_argument("--name", default="gpu-ml", 
                       help="Environment name (default: gpu-ml)")
    parser.add_argument("--framework", choices=["pytorch", "tensorflow"], 
                       default="pytorch", help="ML framework (default: pytorch)")
    parser.add_argument("--detect-only", action="store_true", 
                       help="Only detect GPU, don't create environment")
    parser.add_argument("--list-compatible", action="store_true",
                       help="List all compatible configurations")
    parser.add_argument("--minimal", action="store_true",
                       help="Create minimal environment without extra packages")
    parser.add_argument("--skip-prereq-check", action="store_true",
                       help="Skip prerequisite checks")
    
    args = parser.parse_args()
    
    print("🎯 GPU Environment Creator")
    print("=" * 50)
    
    # Check prerequisites
    if not args.skip_prereq_check and not check_prerequisites():
        print("\n❌ Prerequisites not met. Please install required tools first.")
        return 1
    
    # Step 1: Detect GPU
    print(f"\n🔍 Step 1: GPU Detection")
    detector = GPUDetector()
    gpu_info = detector.detect()
    
    if not gpu_info:
        print("❌ No compatible GPU found")
        print("   This tool works best with NVIDIA GPUs")
        print("   You can still create CPU-only environments manually")
        return 1
    
    if args.detect_only:
        print("\n✅ GPU detection complete!")
        return 0
    
    # Step 2: Find compatible versions
    print(f"\n🔧 Step 2: Compatibility Analysis")
    finder = CompatibilityFinder(gpu_info)
    
    if args.list_compatible:
        print(f"\n📋 All compatible configurations for {args.framework}:")
        compatible_configs = finder.find_all_compatible(args.framework)
        
        if not compatible_configs:
            print(f"❌ No compatible configurations found for {args.framework}")
            return 1
        
        for i, config in enumerate(compatible_configs, 1):
            print(f"\n{i}. {config['framework'].capitalize()} {config['framework_version']}")
            print(f"   CUDA: {config['cuda_version']}")
            print(f"   Python: {config['recommended_python']}")
            print(f"   Compatible: {'✅' if config['driver_compatible'] else '⚠️'}")
        
        return 0
    
    compatibility = finder.find_best_match(args.framework)
    
    if not compatibility:
        print(f"❌ No compatible configuration found for {args.framework}")
        print("   Try a different framework or check your GPU compatibility")
        return 1
    
    # Step 3: Generate environment
    print(f"\n📦 Step 3: Environment Generation")
    generator = EnvironmentGenerator(compatibility)
    files = generator.create_environment(args.name, not args.minimal)
    
    print(f"\n🎉 Success! GPU environment '{args.name}' ready to install")
    print(f"\n📋 Next Steps:")
    print(f"   1. Run: {'install_' + args.name + '.bat' if sys.platform == 'win32' else './install_' + args.name + '.sh'}")
    print(f"   2. Activate: conda activate {args.name}")
    print(f"   3. Test: {'test_' + args.name + '.bat' if sys.platform == 'win32' else './test_' + args.name + '.sh'}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 