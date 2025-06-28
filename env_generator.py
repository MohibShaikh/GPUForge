"""
Environment Generator Module
Generates conda environment files and installation scripts
"""

import os
import platform
import stat
from pathlib import Path
from typing import Dict, Optional

class EnvironmentGenerator:
    def __init__(self, compatibility_config: Dict):
        self.config = compatibility_config
        self.env_name = "gpu-ml"
        self.framework = compatibility_config.get("framework", "pytorch")
    
    def create_environment(self, env_name: str = None, include_extras: bool = True) -> Dict[str, str]:
        """Create complete environment setup"""
        if env_name:
            self.env_name = env_name
        
        print(f"📦 Creating environment: {self.env_name}")
        
        # Generate files
        env_file = self._generate_conda_env(include_extras)
        install_script = self._generate_install_script()
        test_script = self._generate_test_script()
        
        # Create info file
        info_file = self._generate_info_file()
        
        print(f"\n🎉 Environment files created:")
        print(f"   📄 Conda environment: {env_file}")
        print(f"   🚀 Installation script: {install_script}")
        print(f"   🧪 Test script: {test_script}")
        print(f"   ℹ️  Info file: {info_file}")
        
        return {
            "env_file": env_file,
            "install_script": install_script,
            "test_script": test_script,
            "info_file": info_file
        }
    
    def _generate_conda_env(self, include_extras: bool = True) -> str:
        """Generate conda environment YAML file"""
        print(f"📝 Generating conda environment for {self.framework}")
        
        env_content = f"""name: {self.env_name}
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults

dependencies:
  - python={self.config['recommended_python']}
  - pip
  - numpy
  - scipy
"""
        
        # Framework-specific packages
        if self.framework == "pytorch":
            env_content += self._get_pytorch_packages()
        elif self.framework == "tensorflow":
            env_content += self._get_tensorflow_packages()
        
        # Common ML packages
        if include_extras:
            env_content += self._get_common_packages()
        
        # Write environment file
        env_file = f"{self.env_name}.yml"
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        return env_file
    
    def _get_pytorch_packages(self) -> str:
        """Get PyTorch-specific package configuration"""
        cuda_version = self.config['cuda_version']
        
        if cuda_version in ["12.1", "12.2"]:
            # Latest CUDA versions use pytorch-cuda
            return f"""  - pytorch
  - torchvision
  - torchaudio
  - pytorch-cuda={cuda_version}
"""
        else:
            # Older CUDA versions use cudatoolkit
            return f"""  - pytorch
  - torchvision
  - torchaudio
  - cudatoolkit={cuda_version}
"""
    
    def _get_tensorflow_packages(self) -> str:
        """Get TensorFlow-specific package configuration"""
        cuda_version = self.config['cuda_version']
        tf_version = self.config['framework_version']
        
        return f"""  - cudatoolkit={cuda_version}
  - cudnn
  - pip:
    - tensorflow[and-cuda]=={tf_version}
"""
    
    def _get_common_packages(self) -> str:
        """Get common ML packages"""
        return """  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - plotly
  - jupyter
  - jupyterlab
  - ipykernel
  - notebook
  - ipywidgets
  - tqdm
  - pillow
  - opencv
  - pip:
    - accelerate
    - transformers
    - datasets
    - wandb
    - tensorboard
    - gradio
    - streamlit
"""
    
    def _generate_install_script(self) -> str:
        """Generate installation script"""
        print("📜 Generating installation script")
        
        is_windows = platform.system() == "Windows"
        script_ext = ".bat" if is_windows else ".sh"
        script_file = f"install_{self.env_name}{script_ext}"
        
        if is_windows:
            script_content = self._get_windows_install_script()
        else:
            script_content = self._get_unix_install_script()
        
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Make executable on Unix systems
        if not is_windows:
            self._make_executable(script_file)
        
        return script_file
    
    def _get_windows_install_script(self) -> str:
        """Generate Windows batch install script"""
        return f"""@echo off
REM GPU Environment Setup Script for Windows
REM Generated automatically by GPU Environment Creator

echo 🚀 Setting up GPU environment: {self.env_name}

REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Conda not found. Please install Anaconda or Miniconda first.
    pause
    exit /b 1
)

echo Creating conda environment...
conda env create -f {self.env_name}.yml

if %errorlevel% neq 0 (
    echo ❌ Failed to create environment
    pause
    exit /b 1
)

echo ✅ Environment created successfully!
echo.
echo 🔍 To activate the environment, run:
echo    conda activate {self.env_name}
echo.
echo 🧪 To test the installation, run:
echo    test_{self.env_name}.bat
echo.
pause
"""
    
    def _get_unix_install_script(self) -> str:
        """Generate Unix shell install script"""
        return f"""#!/bin/bash
# GPU Environment Setup Script
# Generated automatically by GPU Environment Creator

set -e  # Exit on any error

echo "🚀 Setting up GPU environment: {self.env_name}"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "❌ Conda not found. Please install Anaconda or Miniconda first."
    exit 1
fi

echo "Creating conda environment..."
conda env create -f {self.env_name}.yml

echo "✅ Environment created successfully!"
echo ""
echo "🔍 To activate the environment, run:"
echo "   conda activate {self.env_name}"
echo ""
echo "🧪 To test the installation, run:"
echo "   ./test_{self.env_name}.sh"
"""
    
    def _generate_test_script(self) -> str:
        """Generate test script to verify installation"""
        print("🧪 Generating test script")
        
        is_windows = platform.system() == "Windows"
        script_ext = ".bat" if is_windows else ".sh"
        script_file = f"test_{self.env_name}{script_ext}"
        
        if is_windows:
            test_content = self._get_windows_test_script()
        else:
            test_content = self._get_unix_test_script()
        
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(test_content)
        
        # Make executable on Unix systems
        if not is_windows:
            self._make_executable(script_file)
        
        return script_file
    
    def _get_windows_test_script(self) -> str:
        """Generate Windows test script"""
        return f"""@echo off
REM Test script for {self.env_name} environment

echo 🧪 Testing GPU environment: {self.env_name}

REM Activate environment
call conda activate {self.env_name}

if %errorlevel% neq 0 (
    echo ❌ Failed to activate environment
    pause
    exit /b 1
)

echo Testing installation...
python -c "{self._get_test_python_code()}"

pause
"""
    
    def _get_unix_test_script(self) -> str:
        """Generate Unix test script"""
        return f"""#!/bin/bash
# Test script for {self.env_name} environment

set -e

echo "🧪 Testing GPU environment: {self.env_name}"

# Activate environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate {self.env_name}

echo "Testing installation..."
python -c "{self._get_test_python_code()}"

echo "✅ All tests passed!"
"""
    
    def _get_test_python_code(self) -> str:
        """Generate Python test code"""
        if self.framework == "pytorch":
            return """
import sys
import torch
print(f'✅ Python version: {sys.version}')
print(f'✅ PyTorch version: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU count: {torch.cuda.device_count()}')
    print(f'✅ Current GPU: {torch.cuda.get_device_name(0)}')
    # Test GPU tensor operation
    x = torch.randn(2, 3).cuda()
    y = torch.randn(3, 2).cuda()
    z = torch.mm(x, y)
    print(f'✅ GPU tensor operation successful: {z.shape}')
else:
    print('⚠️  CUDA not available - check drivers and installation')
"""
        elif self.framework == "tensorflow":
            return """
import sys
import tensorflow as tf
print(f'✅ Python version: {sys.version}')
print(f'✅ TensorFlow version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
print(f'✅ GPU devices found: {len(gpus)}')
if gpus:
    for i, gpu in enumerate(gpus):
        print(f'✅ GPU {i}: {gpu}')
    # Test GPU operation
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
        c = tf.matmul(a, b)
        print(f'✅ GPU operation successful: {c.shape}')
else:
    print('⚠️  No GPU devices found - check drivers and installation')
"""
        else:
            return """
import sys
print(f'✅ Python version: {sys.version}')
print('✅ Basic environment test passed')
"""
    
    def _generate_info_file(self) -> str:
        """Generate information file about the environment"""
        info_file = f"{self.env_name}_info.txt"
        
        info_content = f"""GPU Environment Information
Generated by GPU Environment Creator
================================

Environment Name: {self.env_name}
Framework: {self.config['framework'].capitalize()}
Framework Version: {self.config['framework_version']}
CUDA Version: {self.config['cuda_version']}
Python Version: {self.config['recommended_python']}
Compute Capability: {self.config.get('compute_capability', 'Unknown')}

Compatible Python Versions: {', '.join(self.config['python_versions'])}
Driver Compatible: {self.config.get('driver_compatible', 'Unknown')}

Setup Instructions:
==================
1. Run the installation script:
   - Windows: install_{self.env_name}.bat
   - Linux/Mac: ./install_{self.env_name}.sh

2. Activate the environment:
   conda activate {self.env_name}

3. Test the installation:
   - Windows: test_{self.env_name}.bat
   - Linux/Mac: ./test_{self.env_name}.sh

Files Generated:
================
- {self.env_name}.yml - Conda environment specification
- install_{self.env_name}.* - Installation script
- test_{self.env_name}.* - Test script
- {info_file} - This information file

Troubleshooting:
===============
If you encounter issues:
1. Make sure NVIDIA drivers are installed and up to date
2. Verify conda is properly installed
3. Check that your GPU supports the required compute capability
4. Ensure sufficient disk space for package installation

For more help, visit:
- PyTorch: https://pytorch.org/get-started/locally/
- TensorFlow: https://www.tensorflow.org/install/gpu
- CUDA: https://developer.nvidia.com/cuda-toolkit
"""
        
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(info_content)
        
        return info_file
    
    def _make_executable(self, file_path: str):
        """Make a file executable on Unix systems"""
        try:
            current_permissions = Path(file_path).stat().st_mode
            Path(file_path).chmod(current_permissions | stat.S_IEXEC)
        except Exception:
            pass  # Ignore errors, file permissions are not critical
    
    def create_custom_environment(self, packages: list, env_name: str = None) -> str:
        """Create a custom environment with specific packages"""
        if env_name:
            self.env_name = env_name
        
        print(f"📦 Creating custom environment: {self.env_name}")
        
        env_content = f"""name: {self.env_name}
channels:
  - pytorch
  - nvidia
  - conda-forge
  - defaults

dependencies:
  - python={self.config['recommended_python']}
  - pip
"""
        
        # Add framework packages
        if self.framework == "pytorch":
            env_content += self._get_pytorch_packages()
        elif self.framework == "tensorflow":
            env_content += self._get_tensorflow_packages()
        
        # Add custom packages
        conda_packages = []
        pip_packages = []
        
        for package in packages:
            if package.startswith("pip:"):
                pip_packages.append(package[4:])
            else:
                conda_packages.append(package)
        
        for package in conda_packages:
            env_content += f"  - {package}\n"
        
        if pip_packages:
            env_content += "  - pip:\n"
            for package in pip_packages:
                env_content += f"    - {package}\n"
        
        env_file = f"{self.env_name}.yml"
        with open(env_file, 'w', encoding='utf-8') as f:
            f.write(env_content)
        
        print(f"✅ Custom environment created: {env_file}")
        return env_file 