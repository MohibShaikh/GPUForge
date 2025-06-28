"""
GPU Detection Module
Detects NVIDIA GPU model, compute capability, and driver version
"""

import subprocess
import re
from typing import Dict, Optional

class GPUDetector:
    def __init__(self):
        self.gpu_info = None
    
    def detect(self) -> Optional[Dict]:
        """Detect GPU information using multiple methods"""
        print("🔍 Detecting GPU...")
        
        # Method 1: nvidia-smi (most reliable)
        gpu_info = self._detect_with_nvidia_smi()
        if gpu_info:
            return gpu_info
        
        # Method 2: GPUtil library
        gpu_info = self._detect_with_gputil()
        if gpu_info:
            return gpu_info
        
        # Method 3: pynvml library
        gpu_info = self._detect_with_pynvml()
        if gpu_info:
            return gpu_info
        
        print("⚠️  No NVIDIA GPU detected or tools not available")
        return None
    
    def _detect_with_nvidia_smi(self) -> Optional[Dict]:
        """Detect GPU using nvidia-smi command"""
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=name,compute_cap,driver_version,memory.total', 
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, check=True)
            
            lines = result.stdout.strip().split('\n')
            
            for line in lines:
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 4:
                        name, compute_cap, driver_version, memory = parts[:4]
                        
                        gpu_info = {
                            'name': name,
                            'compute_capability': compute_cap,
                            'driver_version': driver_version,
                            'memory_mb': int(memory),
                            'detection_method': 'nvidia-smi'
                        }
                        
                        self.gpu_info = gpu_info
                        self._print_gpu_info(gpu_info)
                        return gpu_info
                        
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            pass
        
        return None
    
    def _detect_with_gputil(self) -> Optional[Dict]:
        """Detect GPU using GPUtil library"""
        try:
            import GPUtil
            gpus = GPUtil.getGPUs()
            
            if gpus:
                gpu = gpus[0]  # Use first GPU
                gpu_info = {
                    'name': gpu.name,
                    'memory_mb': gpu.memoryTotal,
                    'driver_version': gpu.driver,
                    'detection_method': 'GPUtil'
                }
                
                # Try to get compute capability from name
                compute_cap = self._infer_compute_capability(gpu.name)
                if compute_cap:
                    gpu_info['compute_capability'] = compute_cap
                
                self.gpu_info = gpu_info
                self._print_gpu_info(gpu_info)
                return gpu_info
                
        except ImportError:
            pass
        except Exception:
            pass
        
        return None
    
    def _detect_with_pynvml(self) -> Optional[Dict]:
        """Detect GPU using pynvml library"""
        try:
            import pynvml
            
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            
            if device_count > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                name = pynvml.nvmlDeviceGetName(handle).decode('utf-8')
                
                # Get memory info
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_mb = mem_info.total // (1024 * 1024)
                
                # Get driver version
                driver_version = pynvml.nvmlSystemGetDriverVersion().decode('utf-8')
                
                gpu_info = {
                    'name': name,
                    'memory_mb': memory_mb,
                    'driver_version': driver_version,
                    'detection_method': 'pynvml'
                }
                
                # Try to get compute capability
                try:
                    major = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[0]
                    minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)[1]
                    gpu_info['compute_capability'] = f"{major}.{minor}"
                except:
                    compute_cap = self._infer_compute_capability(name)
                    if compute_cap:
                        gpu_info['compute_capability'] = compute_cap
                
                self.gpu_info = gpu_info
                self._print_gpu_info(gpu_info)
                return gpu_info
                
        except ImportError:
            pass
        except Exception:
            pass
        
        return None
    
    def _infer_compute_capability(self, gpu_name: str) -> Optional[str]:
        """Infer compute capability from GPU name"""
        # Common GPU compute capabilities
        compute_map = {
            # RTX 40 series (Ada Lovelace)
            'RTX 4090': '8.9', 'RTX 4080': '8.9', 'RTX 4070': '8.9', 'RTX 4060': '8.9',
            
            # RTX 30 series (Ampere)
            'RTX 3090': '8.6', 'RTX 3080': '8.6', 'RTX 3070': '8.6', 'RTX 3060': '8.6',
            'RTX 3050': '8.6',
            
            # RTX 20 series (Turing)
            'RTX 2080': '7.5', 'RTX 2070': '7.5', 'RTX 2060': '7.5',
            
            # GTX 16 series (Turing)
            'GTX 1660': '7.5', 'GTX 1650': '7.5',
            
            # GTX 10 series (Pascal)
            'GTX 1080': '6.1', 'GTX 1070': '6.1', 'GTX 1060': '6.1', 'GTX 1050': '6.1',
            
            # Tesla/Quadro
            'V100': '7.0', 'P100': '6.0', 'K80': '3.7', 'T4': '7.5', 'A100': '8.0'
        }
        
        gpu_name_upper = gpu_name.upper()
        
        for model, capability in compute_map.items():
            if model.upper() in gpu_name_upper:
                return capability
        
        return None
    
    def _print_gpu_info(self, gpu_info: Dict):
        """Print formatted GPU information"""
        print(f"✅ Found GPU: {gpu_info['name']}")
        
        if 'compute_capability' in gpu_info:
            print(f"   Compute Capability: {gpu_info['compute_capability']}")
        
        if 'driver_version' in gpu_info:
            print(f"   Driver Version: {gpu_info['driver_version']}")
        
        if 'memory_mb' in gpu_info:
            memory_gb = gpu_info['memory_mb'] / 1024
            print(f"   Memory: {memory_gb:.1f} GB")
        
        print(f"   Detection Method: {gpu_info['detection_method']}")
    
    def get_gpu_info(self) -> Optional[Dict]:
        """Get the detected GPU information"""
        return self.gpu_info 