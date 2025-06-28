#!/usr/bin/env python3
"""
GPU Environment Creator
Automatically detects GPU and creates compatible conda environments for ML/DL work
"""

import subprocess
import sys
import json
import re
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import platform

class GPUEnvCreator:
    def __init__(self):
        self.gpu_info = None
        self.compatibility_matrix = None
        self.recommended_versions = {}
        
    def detect_gpu(self) -> Dict:
        """Detect GPU model and compute capability"""
        print("🔍 Detecting GPU...")
        
        try:
            # Try nvidia-smi first
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,compute_cap,driver_version', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True, check=True)
            
            lines = result.stdout.strip().split('\n')
            gpus = []
            
            for line in lines:
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 3:
                        name, compute_cap, driver_version = parts[0], parts[1], parts[2]
                        gpus.append({
                            'name': name,
                            'compute_capability': compute_cap,
                            'driver_version': driver_version
                        })
            
            if gpus:
                self.gpu_info = gpus[0]  # Use first GPU
                print(f"✅ Found GPU: {self.gpu_info['name']}")
                print(f"   Compute Capability: {self.gpu_info['compute_capability']}")
                print(f"   Driver Version: {self.gpu_info['driver_version']}")
                return self.gpu_info
                
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("❌ nvidia-smi not found or failed")
            
        # Fallback: try to detect through other means
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                self.gpu_info = {
                    'name': gpu.name,
                    'memory': f"{gpu.memoryTotal}MB",
                    'driver_version': gpu.driver
                }
                print(f"✅ Found GPU: {self.gpu_info['name']}")
                return self.gpu_info
        except ImportError:
            pass
            
        print("⚠️  No NVIDIA GPU detected or tools not available")
        return None
    
    def get_compatibility_matrix(self) -> Dict:
        """Get compatibility matrix for CUDA, PyTorch, TensorFlow versions"""
        print("📊 Loading compatibility matrix...")
        
        # This would ideally be fetched from official sources or a maintained database
        # For now, using a comprehensive static matrix based on official docs
        matrix = {
            "pytorch": {
                "2.1.0": {"cuda": ["11.8", "12.1"], "python": ["3.8", "3.9", "3.10", "3.11"]},
                "2.0.1": {"cuda": ["11.7", "11.8"], "python": ["3.8", "3.9", "3.10", "3.11"]},
                "1.13.1": {"cuda": ["11.6", "11.7"], "python": ["3.7", "3.8", "3.9", "3.10"]},
                "1.12.1": {"cuda": ["11.3", "11.6"], "python": ["3.7", "3.8", "3.9", "3.10"]}
            },
            "tensorflow": {
                "2.13.0": {"cuda": ["11.8"], "python": ["3.8", "3.9", "3.10", "3.11"]},
                "2.12.0": {"cuda": ["11.8"], "python": ["3.8", "3.9", "3.10", "3.11"]},
                "2.11.0": {"cuda": ["11.2"], "python": ["3.7", "3.8", "3.9", "3.10"]},
                "2.10.0": {"cuda": ["11.2"], "python": ["3.7", "3.8", "3.9", "3.10"]}
            },
            "cuda_compute_capability": {
                "12.1": {"min_compute": "3.5"},
                "11.8": {"min_compute": "3.5"},
                "11.7": {"min_compute": "3.5"},
                "11.6": {"min_compute": "3.5"},
                "11.3": {"min_compute": "3.5"},
                "11.2": {"min_compute": "3.5"}
            }
        }
        
        self.compatibility_matrix = matrix
        print("✅ Compatibility matrix loaded")
        return matrix
    
    def find_compatible_versions(self, framework="pytorch") -> Dict:
        """Find compatible versions based on GPU and requirements"""
        print(f"🔧 Finding compatible versions for {framework}...")
        
        if not self.compatibility_matrix:
            self.get_compatibility_matrix()
            
        compatible_configs = []
        
        # Get compute capability as float for comparison
        gpu_compute = None
        if self.gpu_info and 'compute_capability' in self.gpu_info:
            try:
                gpu_compute = float(self.gpu_info['compute_capability'])
            except ValueError:
                print(f"⚠️  Could not parse compute capability: {self.gpu_info['compute_capability']}")
        
        framework_versions = self.compatibility_matrix.get(framework, {})
        
        for fw_version, requirements in framework_versions.items():
            cuda_versions = requirements.get("cuda", [])
            python_versions = requirements.get("python", [])
            
            for cuda_version in cuda_versions:
                # Check if GPU supports this CUDA version
                cuda_info = self.compatibility_matrix["cuda_compute_capability"].get(cuda_version)
                if cuda_info and gpu_compute:
                    min_compute = float(cuda_info["min_compute"])
                    if gpu_compute < min_compute:
                        continue
                
                compatible_configs.append({
                    "framework": framework,
                    "framework_version": fw_version,
                    "cuda_version": cuda_version,
                    "python_versions": python_versions,
                    "recommended_python": python_versions[-1]  # Latest supported
                })
        
        if compatible_configs:
            # Recommend the latest framework version with latest CUDA
            recommended = max(compatible_configs, 
                            key=lambda x: (x["framework_version"], x["cuda_version"]))
            self.recommended_versions = recommended
            
            print(f"✅ Recommended configuration:")
            print(f"   {framework.capitalize()}: {recommended['framework_version']}")
            print(f"   CUDA: {recommended['cuda_version']}")
            print(f"   Python: {recommended['recommended_python']}")
            
            return recommended
        else:
            print(f"❌ No compatible configuration found for {framework}")
            return {}
    
    def generate_conda_env(self, env_name: str = "gpu-ml", framework: str = "pytorch") -> str:
        """Generate conda environment YAML file"""
        print(f"📝 Generating conda environment: {env_name}")
        
        if not self.recommended_versions:
            self.find_compatible_versions(framework)
        
        if not self.recommended_versions:
            raise ValueError("No compatible versions found")
        
        config = self.recommended_versions
        
        # Generate environment YAML
        env_content = f"""name: {env_name}
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults

dependencies:
  - python={config['recommended_python']}
  - pip
"""
        
        if framework == "pytorch":
            if config['cuda_version'] == "11.8":
                env_content += f"""  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda={config['cuda_version']}
"""
            else:
                env_content += f"""  - pytorch
  - torchvision  
  - torchaudio
  - cudatoolkit={config['cuda_version']}
"""
        
        elif framework == "tensorflow":
            env_content += f"""  - cudatoolkit={config['cuda_version']}
  - cudnn
  - pip:
    - tensorflow[and-cuda]=={config['framework_version']}
"""
        
        # Add common ML packages
        env_content += """  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - jupyter
  - ipykernel
  - pip:
    - accelerate
    - transformers
    - datasets
"""
        
        # Write to file
        env_file = f"{env_name}.yml"
        with open(env_file, 'w') as f:
            f.write(env_content)
        
        print(f"✅ Environment file created: {env_file}")
        return env_file
    
    def generate_install_script(self, env_name: str = "gpu-ml") -> str:
        """Generate installation script"""
        print("📜 Generating installation script...")
        
        script_content = f"""#!/bin/bash
# GPU Environment Setup Script
# Generated automatically by GPU Environment Creator

echo "🚀 Setting up GPU environment: {env_name}"

# Create conda environment
echo "Creating conda environment..."
conda env create -f {env_name}.yml

# Activate environment
echo "Activating environment..."
conda activate {env_name}

# Verify installation
echo "🔍 Verifying installation..."
python -c "
import sys
print(f'Python version: {{sys.version}}')

try:
    import torch
    print(f'PyTorch version: {{torch.__version__}}')
    print(f'CUDA available: {{torch.cuda.is_available()}}')
    if torch.cuda.is_available():
        print(f'CUDA version: {{torch.version.cuda}}')
        print(f'GPU count: {{torch.cuda.device_count()}}')
        print(f'Current GPU: {{torch.cuda.get_device_name(0)}}')
except ImportError:
    print('PyTorch not installed')

try:
    import tensorflow as tf
    print(f'TensorFlow version: {{tf.__version__}}')
    print(f'GPU devices: {{len(tf.config.list_physical_devices(\"GPU\"))}}')
except ImportError:
    print('TensorFlow not installed')
"

echo "✅ Setup complete! Activate with: conda activate {env_name}"
"""
        
        script_file = f"install_{env_name}.sh"
        with open(script_file, 'w') as f:
            f.write(script_content)
        
        # Make executable on Unix systems
        if platform.system() != "Windows":
            import stat
            st = Path(script_file).stat()
            Path(script_file).chmod(st.st_mode | stat.S_IEXEC)
        
        print(f"✅ Installation script created: {script_file}")
        return script_file
    
    def run_diagnostics(self):
        """Run comprehensive diagnostics"""
        print("\n🏥 Running diagnostics...")
        
        # Check conda
        try:
            result = subprocess.run(['conda', '--version'], capture_output=True, text=True)
            print(f"✅ Conda: {result.stdout.strip()}")
        except FileNotFoundError:
            print("❌ Conda not found - please install Anaconda or Miniconda first")
            return False
        
        # Check NVIDIA drivers
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ NVIDIA drivers installed")
            else:
                print("❌ NVIDIA drivers not working properly")
        except FileNotFoundError:
            print("❌ nvidia-smi not found - install NVIDIA drivers")
        
        return True
    
    def create_environment(self, env_name: str = "gpu-ml", framework: str = "pytorch"):
        """Main method to create GPU environment"""
        print("🎯 GPU Environment Creator")
        print("=" * 50)
        
        # Run diagnostics
        if not self.run_diagnostics():
            print("\n❌ Prerequisites not met. Please install conda and NVIDIA drivers first.")
            return
        
        # Detect GPU
        gpu_info = self.detect_gpu()
        
        # Get compatibility matrix
        self.get_compatibility_matrix()
        
        # Find compatible versions
        compatible = self.find_compatible_versions(framework)
        
        if not compatible:
            print(f"\n❌ Could not find compatible configuration for {framework}")
            return
        
        # Generate files
        env_file = self.generate_conda_env(env_name, framework)
        script_file = self.generate_install_script(env_name)
        
        print(f"\n🎉 Success! Files generated:")
        print(f"   📄 Environment: {env_file}")
        print(f"   📜 Install script: {script_file}")
        
        print(f"\n🚀 To create the environment, run:")
        if platform.system() == "Windows":
            print(f"   conda env create -f {env_file}")
        else:
            print(f"   ./{script_file}")
        
        print(f"\n💡 Then activate with:")
        print(f"   conda activate {env_name}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="GPU Environment Creator")
    parser.add_argument("--name", default="gpu-ml", help="Environment name")
    parser.add_argument("--framework", choices=["pytorch", "tensorflow"], 
                       default="pytorch", help="ML framework")
    parser.add_argument("--detect-only", action="store_true", 
                       help="Only detect GPU, don't create environment")
    
    args = parser.parse_args()
    
    creator = GPUEnvCreator()
    
    if args.detect_only:
        creator.detect_gpu()
    else:
        creator.create_environment(args.name, args.framework)

if __name__ == "__main__":
    main() 