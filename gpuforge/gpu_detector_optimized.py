"""
Optimized GPU Detection Module
Features: Async detection, caching, universal vendor support
"""

import asyncio
import concurrent.futures
import subprocess
import pickle
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
import platform
import json

class GPUCache:
    def __init__(self, cache_ttl: int = 3600):  # 1 hour default
        self.cache_dir = Path.home() / ".gpu_env_creator"
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_file = self.cache_dir / "gpu_cache.pkl"
        self.cache_ttl = cache_ttl
    
    def _get_system_fingerprint(self) -> str:
        """Create a fingerprint of the system to detect hardware changes"""
        system_info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'machine': platform.machine()
        }
        
        # Add driver version if available
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                system_info['nvidia_driver'] = result.stdout.strip()
        except:
            pass
        
        fingerprint = json.dumps(system_info, sort_keys=True)
        return hashlib.md5(fingerprint.encode()).hexdigest()
    
    def get_cached_gpu_info(self) -> Optional[List[Dict]]:
        """Get cached GPU information if valid"""
        if not self.cache_file.exists():
            return None
        
        try:
            with open(self.cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # Check if cache is still valid
            if (time.time() - cache_data['timestamp'] < self.cache_ttl and
                cache_data['system_fingerprint'] == self._get_system_fingerprint()):
                
                print("⚡ Using cached GPU information")
                return cache_data['gpu_info']
                
        except Exception as e:
            print(f"Cache read error: {e}")
        
        return None
    
    def cache_gpu_info(self, gpu_info: List[Dict]):
        """Cache GPU information with system fingerprint"""
        cache_data = {
            'gpu_info': gpu_info,
            'timestamp': time.time(),
            'system_fingerprint': self._get_system_fingerprint()
        }
        
        try:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
        except Exception as e:
            print(f"Cache write error: {e}")

class UniversalGPUDetector:
    def __init__(self, use_cache: bool = True, cache_ttl: int = 3600):
        self.use_cache = use_cache
        self.cache = GPUCache(cache_ttl) if use_cache else None
        self.detected_gpus = []
    
    async def detect_all_gpus(self) -> List[Dict]:
        """Detect all GPUs from all vendors using async methods"""
        
        # Try cache first
        if self.use_cache:
            cached_info = self.cache.get_cached_gpu_info()
            if cached_info:
                self.detected_gpus = cached_info
                return cached_info
        
        print("🔍 Detecting GPUs from all vendors...")
        
        # Run all detection methods in parallel
        detection_methods = [
            self._detect_nvidia_gpus_async,
            self._detect_amd_gpus_async,
            self._detect_intel_gpus_async
        ]
        
        all_gpus = []
        
        # Use ThreadPoolExecutor for I/O bound operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Submit all detection tasks
            future_to_vendor = {
                executor.submit(method): method.__name__ 
                for method in detection_methods
            }
            
            # Collect results with timeout
            for future in concurrent.futures.as_completed(future_to_vendor, timeout=10):
                try:
                    vendor_gpus = future.result()
                    if vendor_gpus:
                        all_gpus.extend(vendor_gpus)
                except Exception as e:
                    vendor = future_to_vendor[future]
                    print(f"⚠️  {vendor} detection failed: {e}")
        
        if not all_gpus:
            return self._handle_no_gpu_found()
        
        # Cache successful detection
        if self.use_cache and all_gpus:
            self.cache.cache_gpu_info(all_gpus)
        
        self.detected_gpus = all_gpus
        self._print_detection_summary(all_gpus)
        return all_gpus
    
    def _detect_nvidia_gpus_async(self) -> List[Dict]:
        """Detect NVIDIA GPUs using multiple methods"""
        nvidia_gpus = []
        
        # Method 1: nvidia-smi (most reliable)
        try:
            result = subprocess.run([
                'nvidia-smi', 
                '--query-gpu=index,name,compute_cap,driver_version,memory.total',
                '--format=csv,noheader,nounits'
            ], capture_output=True, text=True, timeout=8, check=True)
            
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = [p.strip() for p in line.split(',')]
                    if len(parts) >= 5:
                        idx, name, compute_cap, driver, memory = parts[:5]
                        
                        gpu_info = {
                            'vendor': 'NVIDIA',
                            'index': int(idx) if idx.isdigit() else 0,
                            'name': name,
                            'compute_capability': compute_cap,
                            'driver_version': driver,
                            'memory_mb': int(memory) if memory.isdigit() else 0,
                            'detection_method': 'nvidia-smi',
                            'supported_apis': ['CUDA', 'OpenCL']
                        }
                        nvidia_gpus.append(gpu_info)
                        
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        # Method 2: GPUtil fallback
        if not nvidia_gpus:
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                
                for i, gpu in enumerate(gpus):
                    compute_cap = self._infer_nvidia_compute_capability(gpu.name)
                    gpu_info = {
                        'vendor': 'NVIDIA',
                        'index': i,
                        'name': gpu.name,
                        'compute_capability': compute_cap,
                        'driver_version': gpu.driver,
                        'memory_mb': gpu.memoryTotal,
                        'detection_method': 'GPUtil',
                        'supported_apis': ['CUDA', 'OpenCL']
                    }
                    nvidia_gpus.append(gpu_info)
                    
            except ImportError:
                pass
            except Exception:
                pass
        
        return nvidia_gpus
    
    def _detect_amd_gpus_async(self) -> List[Dict]:
        """Detect AMD GPUs using ROCm and other methods"""
        amd_gpus = []
        
        # Method 1: ROCm tools
        try:
            result = subprocess.run(['rocm-smi', '--showproductname'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for i, line in enumerate(lines):
                    if 'GPU' in line and ':' in line:
                        name = line.split(':', 1)[1].strip()
                        
                        gpu_info = {
                            'vendor': 'AMD',
                            'index': i,
                            'name': name,
                            'detection_method': 'rocm-smi',
                            'supported_apis': ['ROCm', 'OpenCL'],
                            'architecture': self._infer_amd_architecture(name)
                        }
                        amd_gpus.append(gpu_info)
                        
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return amd_gpus
    
    def _detect_intel_gpus_async(self) -> List[Dict]:
        """Detect Intel GPUs"""
        intel_gpus = []
        
        # Method 1: System detection
        if platform.system() == "Windows":
            try:
                result = subprocess.run([
                    'wmic', 'path', 'win32_VideoController', 
                    'get', 'name,AdapterRAM', '/format:csv'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'Intel' in line and 'Graphics' in line:
                            parts = line.split(',')
                            if len(parts) >= 3:
                                name = parts[2].strip()
                                
                                gpu_info = {
                                    'vendor': 'Intel',
                                    'index': len(intel_gpus),
                                    'name': name,
                                    'detection_method': 'wmic',
                                    'supported_apis': ['OpenCL', 'Level Zero']
                                }
                                intel_gpus.append(gpu_info)
                                
            except Exception:
                pass
        
        return intel_gpus
    
    def _infer_nvidia_compute_capability(self, gpu_name: str) -> Optional[str]:
        """Infer NVIDIA compute capability from GPU name"""
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
            
            # Tesla/Quadro/Professional
            'A100': '8.0', 'A40': '8.6', 'A30': '8.0', 'A10': '8.6',
            'V100': '7.0', 'P100': '6.0', 'K80': '3.7', 'T4': '7.5'
        }
        
        gpu_name_upper = gpu_name.upper()
        
        for model, capability in compute_map.items():
            if model.upper() in gpu_name_upper:
                return capability
        
        return None
    
    def _infer_amd_architecture(self, gpu_name: str) -> str:
        """Infer AMD GPU architecture"""
        gpu_name_upper = gpu_name.upper()
        
        if any(x in gpu_name_upper for x in ['RX 7', 'RX 6']):
            return 'RDNA 2/3'
        elif any(x in gpu_name_upper for x in ['RX 5']):
            return 'RDNA'
        elif any(x in gpu_name_upper for x in ['RX 580', 'RX 570']):
            return 'Polaris'
        elif 'VEGA' in gpu_name_upper:
            return 'Vega'
        else:
            return 'Unknown'
    
    def _handle_no_gpu_found(self) -> List[Dict]:
        """Handle case when no GPUs are found with helpful suggestions"""
        print("\n❌ No GPUs detected")
        print("🔧 Troubleshooting suggestions:")
        
        if platform.system() == "Windows":
            print("   💡 Windows suggestions:")
            print("      • Update GPU drivers from manufacturer website")
            print("      • Check Device Manager for GPU status")
        elif platform.system() == "Linux":
            print("   💡 Linux suggestions:")
            print("      • Install NVIDIA drivers: sudo apt install nvidia-driver-xxx")
            print("      • Check: lspci | grep -i vga")
        
        print("   🌐 Alternative options:")
        print("      • Use Google Colab for free GPU access")
        print("      • Consider cloud GPU services")
        print("      • Create CPU-only environment")
        
        return []
    
    def _print_detection_summary(self, gpus: List[Dict]):
        """Print summary of detected GPUs"""
        if not gpus:
            return
        
        print(f"\n✅ Detected {len(gpus)} GPU(s):")
        
        for i, gpu in enumerate(gpus):
            vendor = gpu.get('vendor', 'Unknown')
            name = gpu.get('name', 'Unknown')
            memory = gpu.get('memory_mb', 0)
            
            memory_str = f"{memory / 1024:.1f} GB" if memory > 0 else "Unknown"
            
            print(f"   {i+1}. {vendor} {name}")
            print(f"      Memory: {memory_str}")
            
            if vendor == 'NVIDIA' and 'compute_capability' in gpu:
                print(f"      Compute Capability: {gpu['compute_capability']}")
            
            if 'supported_apis' in gpu:
                apis = ', '.join(gpu['supported_apis'])
                print(f"      APIs: {apis}")
    
    def get_best_gpu_for_ml(self) -> Optional[Dict]:
        """Get the best GPU for machine learning"""
        if not self.detected_gpus:
            return None
        
        # Score GPUs based on ML suitability
        def score_gpu(gpu):
            score = 0
            
            # Vendor preference (NVIDIA best for ML currently)
            if gpu.get('vendor') == 'NVIDIA':
                score += 100
            elif gpu.get('vendor') == 'AMD':
                score += 60  # ROCm support
            elif gpu.get('vendor') == 'Intel':
                score += 40  # OpenVINO support
            
            # Memory (very important for ML)
            memory_gb = gpu.get('memory_mb', 0) / 1024
            score += min(memory_gb * 10, 80)  # Cap at 8GB worth of points
            
            # Compute capability for NVIDIA
            if gpu.get('vendor') == 'NVIDIA' and 'compute_capability' in gpu:
                try:
                    compute_cap = float(gpu['compute_capability'])
                    score += min(compute_cap * 10, 50)  # Newer architectures
                except:
                    pass
            
            return score
        
        # Return GPU with highest ML score
        best_gpu = max(self.detected_gpus, key=score_gpu)
        return best_gpu

# Backwards compatibility wrapper
class GPUDetector:
    def __init__(self):
        self.universal_detector = UniversalGPUDetector()
        self.gpu_info = None
    
    def detect(self) -> Optional[Dict]:
        """Detect GPU information (backwards compatible)"""
        gpus = asyncio.run(self.universal_detector.detect_all_gpus())
        
        if gpus:
            # Return the best GPU for ML (usually first NVIDIA GPU)
            self.gpu_info = self.universal_detector.get_best_gpu_for_ml()
            if self.gpu_info:
                self._print_gpu_info(self.gpu_info)
            return self.gpu_info
        
        return None
    
    def _print_gpu_info(self, gpu_info: Dict):
        """Print formatted GPU information (backwards compatible)"""
        print(f"✅ Selected GPU: {gpu_info['name']}")
        
        if 'compute_capability' in gpu_info:
            print(f"   Compute Capability: {gpu_info['compute_capability']}")
        
        if 'driver_version' in gpu_info:
            print(f"   Driver Version: {gpu_info['driver_version']}")
        
        if 'memory_mb' in gpu_info:
            memory_gb = gpu_info['memory_mb'] / 1024
            print(f"   Memory: {memory_gb:.1f} GB")
        
        print(f"   Vendor: {gpu_info.get('vendor', 'Unknown')}")
    
    def get_gpu_info(self) -> Optional[Dict]:
        """Get the detected GPU information"""
        return self.gpu_info 