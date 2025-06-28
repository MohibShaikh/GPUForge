"""
GPUForge Cloud Support - Phase 1: Basic Cloud Detection
Conservative implementation with robust error handling and fallbacks.
"""

import json
import requests
import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

# Set up logging
logger = logging.getLogger(__name__)

@dataclass
class CloudInstance:
    """Basic cloud instance information"""
    provider: str
    instance_type: str
    region: Optional[str] = None
    gpu_detected: bool = False
    gpu_type: Optional[str] = None
    gpu_count: int = 0
    confidence: float = 0.0  # 0.0 to 1.0

class CloudDetector:
    """Safe cloud environment detection"""
    
    def __init__(self, timeout: float = 2.0):
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
    
    async def detect_cloud_environment(self) -> Optional[CloudInstance]:
        """
        Safely detect cloud environment with multiple fallback methods
        Returns None if no cloud detected or if detection fails
        """
        try:
            # Try AWS first (most common)
            aws_result = await self._detect_aws()
            if aws_result:
                return aws_result
            
            # Try GCP
            gcp_result = await self._detect_gcp()
            if gcp_result:
                return gcp_result
            
            # Try Azure
            azure_result = await self._detect_azure()
            if azure_result:
                return azure_result
            
            logger.info("No cloud environment detected - running locally")
            return None
            
        except Exception as e:
            logger.debug(f"Cloud detection failed: {e}")
            return None
    
    async def _detect_aws(self) -> Optional[CloudInstance]:
        """Detect AWS EC2 instance"""
        try:
            # AWS metadata service endpoint
            metadata_url = "http://169.254.169.254/latest/meta-data"
            
            # Quick test - check if metadata service is available
            response = self.session.get(f"{metadata_url}/", timeout=1)
            if response.status_code != 200:
                return None
            
            # Get instance type
            instance_response = self.session.get(f"{metadata_url}/instance-type", timeout=1)
            if instance_response.status_code != 200:
                return None
            
            instance_type = instance_response.text.strip()
            
            # Get region (optional)
            region = None
            try:
                region_response = self.session.get(f"{metadata_url}/placement/region", timeout=1)
                if region_response.status_code == 200:
                    region = region_response.text.strip()
            except:
                pass
            
            # Check if this is a GPU instance
            gpu_info = self._classify_aws_gpu_instance(instance_type)
            
            return CloudInstance(
                provider="aws",
                instance_type=instance_type,
                region=region,
                gpu_detected=gpu_info['has_gpu'],
                gpu_type=gpu_info.get('gpu_type'),
                gpu_count=gpu_info.get('gpu_count', 0),
                confidence=0.95
            )
            
        except Exception as e:
            logger.debug(f"AWS detection failed: {e}")
            return None
    
    async def _detect_gcp(self) -> Optional[CloudInstance]:
        """Detect Google Cloud Platform instance"""
        try:
            # GCP metadata service
            metadata_url = "http://metadata.google.internal/computeMetadata/v1"
            headers = {'Metadata-Flavor': 'Google'}
            
            # Test metadata service availability
            response = self.session.get(f"{metadata_url}/", headers=headers, timeout=1)
            if response.status_code != 200:
                return None
            
            # Get machine type
            machine_response = self.session.get(
                f"{metadata_url}/instance/machine-type", 
                headers=headers, 
                timeout=1
            )
            if machine_response.status_code != 200:
                return None
            
            # Parse machine type (format: projects/.../zones/.../machineTypes/TYPE)
            machine_type_full = machine_response.text.strip()
            instance_type = machine_type_full.split('/')[-1]
            
            # Get zone for region info
            region = None
            try:
                zone_response = self.session.get(
                    f"{metadata_url}/instance/zone", 
                    headers=headers, 
                    timeout=1
                )
                if zone_response.status_code == 200:
                    zone_full = zone_response.text.strip()
                    zone = zone_full.split('/')[-1]
                    region = '-'.join(zone.split('-')[:-1])  # Extract region from zone
            except:
                pass
            
            # Check for attached GPUs
            gpu_info = await self._detect_gcp_gpus(headers)
            
            return CloudInstance(
                provider="gcp",
                instance_type=instance_type,
                region=region,
                gpu_detected=gpu_info['has_gpu'],
                gpu_type=gpu_info.get('gpu_type'),
                gpu_count=gpu_info.get('gpu_count', 0),
                confidence=0.90
            )
            
        except Exception as e:
            logger.debug(f"GCP detection failed: {e}")
            return None
    
    async def _detect_azure(self) -> Optional[CloudInstance]:
        """Detect Microsoft Azure instance"""
        try:
            # Azure Instance Metadata Service
            metadata_url = "http://169.254.169.254/metadata/instance"
            headers = {'Metadata': 'true'}
            params = {'api-version': '2021-02-01'}
            
            response = self.session.get(
                metadata_url, 
                headers=headers, 
                params=params, 
                timeout=1
            )
            if response.status_code != 200:
                return None
            
            metadata = response.json()
            compute_info = metadata.get('compute', {})
            
            instance_type = compute_info.get('vmSize', 'unknown')
            region = compute_info.get('location', None)
            
            # Check if this is a GPU instance
            gpu_info = self._classify_azure_gpu_instance(instance_type)
            
            return CloudInstance(
                provider="azure",
                instance_type=instance_type,
                region=region,
                gpu_detected=gpu_info['has_gpu'],
                gpu_type=gpu_info.get('gpu_type'),
                gpu_count=gpu_info.get('gpu_count', 0),
                confidence=0.90
            )
            
        except Exception as e:
            logger.debug(f"Azure detection failed: {e}")
            return None
    
    async def _detect_gcp_gpus(self, headers: Dict) -> Dict:
        """Try to detect attached GPUs on GCP (conservative approach)"""
        try:
            # Try to get guest attributes which might contain GPU info
            gpu_response = self.session.get(
                "http://metadata.google.internal/computeMetadata/v1/instance/guest-attributes/",
                headers=headers,
                timeout=1
            )
            
            # For now, return basic info - would need more sophisticated detection
            return {'has_gpu': False, 'gpu_count': 0}
            
        except:
            return {'has_gpu': False, 'gpu_count': 0}
    
    def _classify_aws_gpu_instance(self, instance_type: str) -> Dict:
        """Classify AWS instance type for GPU capabilities"""
        gpu_instances = {
            # P3 instances (Tesla V100)
            'p3.2xlarge': {'has_gpu': True, 'gpu_type': 'Tesla V100', 'gpu_count': 1},
            'p3.8xlarge': {'has_gpu': True, 'gpu_type': 'Tesla V100', 'gpu_count': 4},
            'p3.16xlarge': {'has_gpu': True, 'gpu_type': 'Tesla V100', 'gpu_count': 8},
            'p3dn.24xlarge': {'has_gpu': True, 'gpu_type': 'Tesla V100', 'gpu_count': 8},
            
            # P4 instances (A100)
            'p4d.24xlarge': {'has_gpu': True, 'gpu_type': 'A100', 'gpu_count': 8},
            
            # G4 instances (T4)
            'g4dn.xlarge': {'has_gpu': True, 'gpu_type': 'Tesla T4', 'gpu_count': 1},
            'g4dn.2xlarge': {'has_gpu': True, 'gpu_type': 'Tesla T4', 'gpu_count': 1},
            'g4dn.4xlarge': {'has_gpu': True, 'gpu_type': 'Tesla T4', 'gpu_count': 1},
            'g4dn.8xlarge': {'has_gpu': True, 'gpu_type': 'Tesla T4', 'gpu_count': 1},
            'g4dn.12xlarge': {'has_gpu': True, 'gpu_type': 'Tesla T4', 'gpu_count': 4},
            'g4dn.16xlarge': {'has_gpu': True, 'gpu_type': 'Tesla T4', 'gpu_count': 1},
            
            # G5 instances (A10G)
            'g5.xlarge': {'has_gpu': True, 'gpu_type': 'A10G', 'gpu_count': 1},
            'g5.2xlarge': {'has_gpu': True, 'gpu_type': 'A10G', 'gpu_count': 1},
            'g5.4xlarge': {'has_gpu': True, 'gpu_type': 'A10G', 'gpu_count': 1},
            'g5.8xlarge': {'has_gpu': True, 'gpu_type': 'A10G', 'gpu_count': 1},
            'g5.12xlarge': {'has_gpu': True, 'gpu_type': 'A10G', 'gpu_count': 4},
            'g5.16xlarge': {'has_gpu': True, 'gpu_type': 'A10G', 'gpu_count': 1},
            'g5.24xlarge': {'has_gpu': True, 'gpu_type': 'A10G', 'gpu_count': 4},
            'g5.48xlarge': {'has_gpu': True, 'gpu_type': 'A10G', 'gpu_count': 8},
        }
        
        return gpu_instances.get(instance_type, {'has_gpu': False, 'gpu_count': 0})
    
    def _classify_azure_gpu_instance(self, instance_type: str) -> Dict:
        """Classify Azure instance type for GPU capabilities"""
        gpu_instances = {
            # NC series (K80)
            'Standard_NC6': {'has_gpu': True, 'gpu_type': 'Tesla K80', 'gpu_count': 1},
            'Standard_NC12': {'has_gpu': True, 'gpu_type': 'Tesla K80', 'gpu_count': 2},
            'Standard_NC24': {'has_gpu': True, 'gpu_type': 'Tesla K80', 'gpu_count': 4},
            
            # NC v2 series (P40)
            'Standard_NC6s_v2': {'has_gpu': True, 'gpu_type': 'Tesla P40', 'gpu_count': 1},
            'Standard_NC12s_v2': {'has_gpu': True, 'gpu_type': 'Tesla P40', 'gpu_count': 2},
            'Standard_NC24s_v2': {'has_gpu': True, 'gpu_type': 'Tesla P40', 'gpu_count': 4},
            
            # NC v3 series (V100)
            'Standard_NC6s_v3': {'has_gpu': True, 'gpu_type': 'Tesla V100', 'gpu_count': 1},
            'Standard_NC12s_v3': {'has_gpu': True, 'gpu_type': 'Tesla V100', 'gpu_count': 2},
            'Standard_NC24s_v3': {'has_gpu': True, 'gpu_type': 'Tesla V100', 'gpu_count': 4},
            
            # ND series (P40)
            'Standard_ND6s': {'has_gpu': True, 'gpu_type': 'Tesla P40', 'gpu_count': 1},
            'Standard_ND12s': {'has_gpu': True, 'gpu_type': 'Tesla P40', 'gpu_count': 2},
            'Standard_ND24s': {'has_gpu': True, 'gpu_type': 'Tesla P40', 'gpu_count': 4},
        }
        
        return gpu_instances.get(instance_type, {'has_gpu': False, 'gpu_count': 0})

# High-level API functions for CLI integration
async def detect_cloud() -> Optional[CloudInstance]:
    """Simple API to detect cloud environment"""
    detector = CloudDetector()
    return await detector.detect_cloud_environment()

def is_cloud_gpu_instance(instance: CloudInstance) -> bool:
    """Check if detected instance has GPUs"""
    return instance.gpu_detected if instance else False

def get_cloud_environment_info(instance: CloudInstance) -> Dict:
    """Get environment configuration for cloud instance"""
    if not instance:
        return {}
    
    env_info = {
        'cloud_provider': instance.provider,
        'instance_type': instance.instance_type,
        'has_gpu': instance.gpu_detected,
        'recommended_cuda': '12.1',  # Conservative default
        'recommended_python': '3.11'  # Conservative default
    }
    
    if instance.region:
        env_info['region'] = instance.region
    
    if instance.gpu_detected:
        env_info.update({
            'gpu_type': instance.gpu_type,
            'gpu_count': instance.gpu_count,
            'cuda_visible_devices': ','.join(map(str, range(instance.gpu_count))) if instance.gpu_count > 1 else '0'
        })
    
    return env_info

# Safe testing function
async def test_cloud_detection():
    """Test cloud detection safely"""
    try:
        print("🔍 Testing cloud detection...")
        instance = await detect_cloud()
        
        if instance:
            print(f"☁️ Cloud detected: {instance.provider.upper()}")
            print(f"   Instance: {instance.instance_type}")
            if instance.region:
                print(f"   Region: {instance.region}")
            if instance.gpu_detected:
                print(f"   GPU: {instance.gpu_count}x {instance.gpu_type}")
            print(f"   Confidence: {instance.confidence:.1%}")
        else:
            print("💻 Local environment detected (no cloud)")
        
        return instance
        
    except Exception as e:
        print(f"❌ Cloud detection test failed: {e}")
        return None

if __name__ == "__main__":
    # Safe testing
    asyncio.run(test_cloud_detection()) 