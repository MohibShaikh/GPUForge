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
        Safely detect cloud environment with detailed verification logging
        Returns None if no cloud detected or if detection fails
        """
        print("🔍 Starting cloud environment detection...")
        self._print_system_properties()
        
        try:
            # Try AWS first (most common)
            print("\n📍 Step 1: Checking for AWS environment...")
            print("   🔗 Probing AWS metadata service (169.254.169.254)...")
            aws_result = await self._detect_aws()
            if aws_result:
                print("✅ AWS environment confirmed!")
                self._print_cloud_properties(aws_result)
                return aws_result
            else:
                print("❌ AWS metadata service not available")
            
            # Try GCP
            print("\n📍 Step 2: Checking for Google Cloud environment...")
            print("   🔗 Probing GCP metadata service (metadata.google.internal)...")
            gcp_result = await self._detect_gcp()
            if gcp_result:
                print("✅ Google Cloud environment confirmed!")
                self._print_cloud_properties(gcp_result)
                return gcp_result
            else:
                print("❌ GCP metadata service not available")
            
            # Try Azure
            print("\n📍 Step 3: Checking for Azure environment...")
            print("   🔗 Probing Azure metadata service...")
            azure_result = await self._detect_azure()
            if azure_result:
                print("✅ Azure environment confirmed!")
                self._print_cloud_properties(azure_result)
                return azure_result
            else:
                print("❌ Azure metadata service not available")
            
            print("\n📍 Final Result: No cloud environment detected")
            print("   ℹ️ This appears to be a local development environment")
            print("   ℹ️ Cloud deployment features will use simulated mode")
            logger.info("No cloud environment detected - running locally")
            return None
            
        except Exception as e:
            print(f"\n❌ Cloud detection failed: {e}")
            logger.debug(f"Cloud detection failed: {e}")
            return None
    
    def _print_system_properties(self):
        """Print basic system properties for verification"""
        import socket
        import platform
        import os
        
        print("\n📊 System Properties Verification:")
        try:
            hostname = socket.gethostname()
            system_info = platform.uname()
            print(f"   🏷️ Hostname: {hostname}")
            print(f"   💻 System: {system_info.system} {system_info.release}")
            print(f"   🏗️ Architecture: {system_info.machine}")
            print(f"   📍 Node: {system_info.node}")
            
            # Check for common cloud environment variables
            cloud_vars = ['AWS_REGION', 'GOOGLE_CLOUD_PROJECT', 'AZURE_SUBSCRIPTION_ID']
            env_indicators = []
            for var in cloud_vars:
                if os.getenv(var):
                    env_indicators.append(f"{var}={os.getenv(var)}")
            
            if env_indicators:
                print(f"   🌐 Cloud Environment Variables: {', '.join(env_indicators)}")
            else:
                print(f"   🌐 Cloud Environment Variables: None detected")
                
        except Exception as e:
            print(f"   ⚠️ Could not gather system properties: {e}")
    
    def _print_cloud_properties(self, instance: CloudInstance):
        """Print detailed cloud properties for user verification"""
        print("\n📋 Detected Cloud Properties - PLEASE VERIFY:")
        print(f"   ☁️ Provider: {instance.provider.upper()}")
        print(f"   📦 Instance Type: {instance.instance_type}")
        print(f"   📍 Region: {instance.region or 'Unknown'}")
        print(f"   🎯 GPU Detected: {'Yes' if instance.gpu_detected else 'No'}")
        if instance.gpu_detected:
            print(f"   🔧 GPU Type: {instance.gpu_type or 'Unknown'}")
            print(f"   📊 GPU Count: {instance.gpu_count}")
        print(f"   📈 Detection Confidence: {instance.confidence:.1%}")
        print("   ✅ Do these properties match your expected environment?")
        print("   ⚠️ If not, please verify your cloud configuration")
    
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