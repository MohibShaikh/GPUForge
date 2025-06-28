"""
Compatibility Finder Module
Finds compatible versions of CUDA, PyTorch, TensorFlow based on GPU capabilities
"""

from typing import Dict, List, Optional, Tuple
import json

class CompatibilityFinder:
    def __init__(self, gpu_info: Dict):
        self.gpu_info = gpu_info
        self.compatibility_matrix = self._load_compatibility_matrix()
    
    def _load_compatibility_matrix(self) -> Dict:
        """Load compatibility matrix for different frameworks and CUDA versions"""
        # Based on official compatibility matrices from PyTorch, TensorFlow, and NVIDIA
        return {
            "pytorch": {
                "2.1.2": {
                    "cuda_versions": ["11.8", "12.1"],
                    "python_versions": ["3.8", "3.9", "3.10", "3.11"],
                    "min_compute_capability": "3.7"
                },
                "2.1.1": {
                    "cuda_versions": ["11.8", "12.1"],
                    "python_versions": ["3.8", "3.9", "3.10", "3.11"],
                    "min_compute_capability": "3.7"
                },
                "2.1.0": {
                    "cuda_versions": ["11.8", "12.1"],
                    "python_versions": ["3.8", "3.9", "3.10", "3.11"],
                    "min_compute_capability": "3.7"
                },
                "2.0.1": {
                    "cuda_versions": ["11.7", "11.8"],
                    "python_versions": ["3.8", "3.9", "3.10", "3.11"],
                    "min_compute_capability": "3.7"
                },
                "1.13.1": {
                    "cuda_versions": ["11.6", "11.7"],
                    "python_versions": ["3.7", "3.8", "3.9", "3.10"],
                    "min_compute_capability": "3.5"
                }
            },
            "tensorflow": {
                "2.15.0": {
                    "cuda_versions": ["12.2"],
                    "python_versions": ["3.9", "3.10", "3.11"],
                    "min_compute_capability": "3.5"
                },
                "2.14.0": {
                    "cuda_versions": ["11.8", "12.2"],
                    "python_versions": ["3.9", "3.10", "3.11"],
                    "min_compute_capability": "3.5"
                },
                "2.13.0": {
                    "cuda_versions": ["11.8"],
                    "python_versions": ["3.8", "3.9", "3.10", "3.11"],
                    "min_compute_capability": "3.5"
                },
                "2.12.0": {
                    "cuda_versions": ["11.8"],
                    "python_versions": ["3.8", "3.9", "3.10", "3.11"],
                    "min_compute_capability": "3.5"
                },
                "2.11.0": {
                    "cuda_versions": ["11.2"],
                    "python_versions": ["3.7", "3.8", "3.9", "3.10"],
                    "min_compute_capability": "3.5"
                }
            },
            "cuda_driver_compatibility": {
                # CUDA version -> minimum driver version
                "12.2": "535.54",
                "12.1": "530.30",
                "11.8": "520.61",
                "11.7": "515.43",
                "11.6": "510.39",
                "11.2": "460.27"
            }
        }
    
    def find_best_match(self, framework: str = "pytorch") -> Optional[Dict]:
        """Find the best compatible configuration"""
        print(f"🔧 Finding compatible versions for {framework}...")
        
        if framework not in self.compatibility_matrix:
            print(f"❌ Framework {framework} not supported")
            return None
        
        compatible_configs = []
        framework_versions = self.compatibility_matrix[framework]
        
        # Get GPU compute capability
        gpu_compute = self._get_compute_capability()
        if not gpu_compute:
            print("⚠️  Could not determine GPU compute capability")
            gpu_compute = 3.5  # Assume minimum for safety
        
        # Get driver version
        driver_version = self._get_driver_version()
        
        for fw_version, requirements in framework_versions.items():
            min_compute = float(requirements["min_compute_capability"])
            
            # Check if GPU meets minimum compute capability
            if gpu_compute < min_compute:
                continue
            
            for cuda_version in requirements["cuda_versions"]:
                # Check driver compatibility
                if driver_version and not self._is_driver_compatible(cuda_version, driver_version):
                    continue
                
                config = {
                    "framework": framework,
                    "framework_version": fw_version,
                    "cuda_version": cuda_version,
                    "python_versions": requirements["python_versions"],
                    "recommended_python": requirements["python_versions"][-1],  # Latest supported
                    "compute_capability": gpu_compute,
                    "driver_compatible": driver_version is None or self._is_driver_compatible(cuda_version, driver_version)
                }
                
                compatible_configs.append(config)
        
        if not compatible_configs:
            print(f"❌ No compatible configuration found for {framework}")
            return None
        
        # Sort by framework version (latest first), then CUDA version (latest first)
        compatible_configs.sort(
            key=lambda x: (x["framework_version"], x["cuda_version"]), 
            reverse=True
        )
        
        best_config = compatible_configs[0]
        self._print_recommendation(best_config)
        
        return best_config
    
    def find_all_compatible(self, framework: str = "pytorch") -> List[Dict]:
        """Find all compatible configurations"""
        print(f"🔍 Finding all compatible configurations for {framework}...")
        
        if framework not in self.compatibility_matrix:
            return []
        
        compatible_configs = []
        framework_versions = self.compatibility_matrix[framework]
        
        gpu_compute = self._get_compute_capability()
        if not gpu_compute:
            gpu_compute = 3.5
        
        driver_version = self._get_driver_version()
        
        for fw_version, requirements in framework_versions.items():
            min_compute = float(requirements["min_compute_capability"])
            
            if gpu_compute < min_compute:
                continue
            
            for cuda_version in requirements["cuda_versions"]:
                if driver_version and not self._is_driver_compatible(cuda_version, driver_version):
                    continue
                
                config = {
                    "framework": framework,
                    "framework_version": fw_version,
                    "cuda_version": cuda_version,
                    "python_versions": requirements["python_versions"],
                    "recommended_python": requirements["python_versions"][-1],
                    "compute_capability": gpu_compute,
                    "driver_compatible": driver_version is None or self._is_driver_compatible(cuda_version, driver_version)
                }
                
                compatible_configs.append(config)
        
        return compatible_configs
    
    def _get_compute_capability(self) -> Optional[float]:
        """Get GPU compute capability as float"""
        if not self.gpu_info or 'compute_capability' not in self.gpu_info:
            return None
        
        try:
            return float(self.gpu_info['compute_capability'])
        except (ValueError, TypeError):
            return None
    
    def _get_driver_version(self) -> Optional[str]:
        """Get driver version"""
        if not self.gpu_info or 'driver_version' not in self.gpu_info:
            return None
        
        return self.gpu_info['driver_version']
    
    def _is_driver_compatible(self, cuda_version: str, driver_version: str) -> bool:
        """Check if driver version is compatible with CUDA version"""
        compatibility_map = self.compatibility_matrix["cuda_driver_compatibility"]
        
        if cuda_version not in compatibility_map:
            return True  # Assume compatible if not in map
        
        min_driver = compatibility_map[cuda_version]
        
        try:
            # Extract version numbers for comparison
            driver_parts = [int(x) for x in driver_version.split('.')]
            min_parts = [int(x) for x in min_driver.split('.')]
            
            # Pad shorter version with zeros
            max_len = max(len(driver_parts), len(min_parts))
            driver_parts.extend([0] * (max_len - len(driver_parts)))
            min_parts.extend([0] * (max_len - len(min_parts)))
            
            return driver_parts >= min_parts
            
        except (ValueError, AttributeError):
            return True  # Assume compatible if parsing fails
    
    def _print_recommendation(self, config: Dict):
        """Print the recommended configuration"""
        print(f"✅ Recommended configuration:")
        print(f"   {config['framework'].capitalize()}: {config['framework_version']}")
        print(f"   CUDA: {config['cuda_version']}")
        print(f"   Python: {config['recommended_python']}")
        print(f"   Compute Capability: {config['compute_capability']}")
        
        if not config['driver_compatible']:
            print(f"   ⚠️  Driver may need update for CUDA {config['cuda_version']}")
    
    def get_framework_options(self) -> List[str]:
        """Get available framework options"""
        return list(self.compatibility_matrix.keys())
    
    def validate_configuration(self, framework: str, framework_version: str, 
                             cuda_version: str, python_version: str) -> Dict:
        """Validate a specific configuration"""
        result = {
            "valid": False,
            "issues": [],
            "recommendations": []
        }
        
        if framework not in self.compatibility_matrix:
            result["issues"].append(f"Framework {framework} not supported")
            return result
        
        framework_data = self.compatibility_matrix[framework]
        
        if framework_version not in framework_data:
            result["issues"].append(f"Framework version {framework_version} not found")
            available = list(framework_data.keys())
            result["recommendations"].append(f"Available versions: {', '.join(available)}")
            return result
        
        requirements = framework_data[framework_version]
        
        # Check CUDA compatibility
        if cuda_version not in requirements["cuda_versions"]:
            result["issues"].append(f"CUDA {cuda_version} not compatible with {framework} {framework_version}")
            result["recommendations"].append(f"Compatible CUDA versions: {', '.join(requirements['cuda_versions'])}")
        
        # Check Python compatibility
        if python_version not in requirements["python_versions"]:
            result["issues"].append(f"Python {python_version} not compatible")
            result["recommendations"].append(f"Compatible Python versions: {', '.join(requirements['python_versions'])}")
        
        # Check compute capability
        gpu_compute = self._get_compute_capability()
        if gpu_compute:
            min_compute = float(requirements["min_compute_capability"])
            if gpu_compute < min_compute:
                result["issues"].append(f"GPU compute capability {gpu_compute} below minimum {min_compute}")
        
        # Check driver compatibility
        driver_version = self._get_driver_version()
        if driver_version and not self._is_driver_compatible(cuda_version, driver_version):
            min_driver = self.compatibility_matrix["cuda_driver_compatibility"].get(cuda_version)
            result["issues"].append(f"Driver version {driver_version} may be too old for CUDA {cuda_version}")
            if min_driver:
                result["recommendations"].append(f"Minimum driver version: {min_driver}")
        
        result["valid"] = len(result["issues"]) == 0
        return result 