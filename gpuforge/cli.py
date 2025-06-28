"""
GPUForge CLI - Command Line Interface
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Optional

# Import from same package
try:
    from .gpu_detector_optimized import UniversalGPUDetector
except ImportError:
    from .gpu_detector import GPUDetector as UniversalGPUDetector

try:
    from .environment_profiles import ProfileManager, EnvironmentProfile
except ImportError:
    ProfileManager = None

try:
    from .error_recovery import SmartErrorRecovery, ConfigurationManager
except ImportError:
    SmartErrorRecovery = None
    ConfigurationManager = None

# Cloud support (Phase 1 - optional feature)
try:
    from .cloud_support import detect_cloud, get_cloud_environment_info, is_cloud_gpu_instance
    CLOUD_SUPPORT_AVAILABLE = True
except ImportError:
    CLOUD_SUPPORT_AVAILABLE = False
    detect_cloud = None
    get_cloud_environment_info = None
    is_cloud_gpu_instance = None

from .compatibility_finder import CompatibilityFinder
from .env_generator import EnvironmentGenerator

class OptimizedGPUEnvironmentCreator:
    def __init__(self):
        self.config_manager = ConfigurationManager() if ConfigurationManager else None
        self.error_recovery = SmartErrorRecovery() if SmartErrorRecovery else None
        self.profile_manager = ProfileManager() if ProfileManager else None
        
        if hasattr(UniversalGPUDetector, 'detect_all_gpus'):
            # Use new optimized detector
            cache_enabled = self.config_manager.get('cache_enabled', True) if self.config_manager else True
            cache_ttl = self.config_manager.get('cache_ttl', 3600) if self.config_manager else 3600
            self.gpu_detector = UniversalGPUDetector(use_cache=cache_enabled, cache_ttl=cache_ttl)
        else:
            # Fallback to original detector
            self.gpu_detector = UniversalGPUDetector()
        
        self.compatibility_finder = None  # Initialize when GPU info available
        self.env_generator = None  # Initialize when compatibility config available
        
        self.start_time = time.time()
        self.performance_metrics = {}
    
    async def create_environment_async(self, args) -> bool:
        """Main async workflow"""
        
        try:
            # Cloud Detection (Phase 1 - optional)
            cloud_instance = None
            if CLOUD_SUPPORT_AVAILABLE and hasattr(args, 'detect_cloud') and args.detect_cloud:
                try:
                    print("☁️ Checking cloud environment...")
                    cloud_instance = await detect_cloud()
                    if cloud_instance:
                        print(f"☁️ Running on {cloud_instance.provider.upper()}: {cloud_instance.instance_type}")
                        if cloud_instance.gpu_detected:
                            print(f"   Cloud GPU detected: {cloud_instance.gpu_count}x {cloud_instance.gpu_type}")
                    else:
                        print("💻 Local environment detected")
                except Exception as e:
                    print(f"⚠️ Cloud detection failed (continuing locally): {e}")
            
            # GPU Detection
            print("🔍 Detecting GPUs...")
            detection_start = time.time()
            
            if hasattr(self.gpu_detector, 'detect_all_gpus'):
                detected_gpus = await self.gpu_detector.detect_all_gpus()
                selected_gpu = self.gpu_detector.get_best_gpu_for_ml()
            else:
                # Fallback to sync detection
                gpu_info = self.gpu_detector.detect()
                detected_gpus = [gpu_info] if gpu_info else []
                selected_gpu = gpu_info
            
            self.performance_metrics['gpu_detection'] = time.time() - detection_start
            
            if not detected_gpus and not args.cpu_only:
                print("❌ No GPUs found. Use --cpu-only for CPU environment")
                return False
            
            # Profile Selection
            if args.cpu_only:
                profile_name = 'lightweight'
                selected_gpu = {'vendor': 'CPU', 'name': 'CPU-only', 'memory_mb': 0}
            else:
                profile_name = args.profile or self._recommend_profile(selected_gpu)
            
            print(f"📋 Using profile: {profile_name}")
            
            # Compatibility
            compatibility_start = time.time()
            if not args.cpu_only:
                # Initialize compatibility finder with GPU info
                self.compatibility_finder = CompatibilityFinder(selected_gpu)
                compatibility = self.compatibility_finder.find_best_match(args.framework or 'pytorch')
            else:
                compatibility = {
                    'framework': 'pytorch', 
                    'framework_version': '2.1.2',
                    'cuda_version': 'cpu',
                    'python_version': '3.11',
                    'recommended_python': '3.11',
                    'python_versions': ['3.8', '3.9', '3.10', '3.11'],
                    'cuda_versions': ['cpu'],
                    'compute_capability': 0.0,
                    'driver_version': 'N/A'
                }
            
            self.performance_metrics['compatibility_check'] = time.time() - compatibility_start
            
            # Environment Generation
            generation_start = time.time()
            success = self._generate_environment(args, selected_gpu, compatibility)
            self.performance_metrics['environment_generation'] = time.time() - generation_start
            
            if args.verbose:
                self._print_performance_summary()
            
            return success
            
        except Exception as e:
            return await self._handle_error(e)
    
    def _recommend_profile(self, gpu_info):
        """Recommend profile based on GPU"""
        if not self.profile_manager:
            return 'research'  # Default fallback
        
        return self.profile_manager.recommend_profile(gpu_info)
    
    def _generate_environment(self, args, gpu_info, compatibility) -> bool:
        """Generate environment"""
        try:
            print(f"📦 Generating environment '{args.name}'...")
            
            # Initialize environment generator with compatibility config
            self.env_generator = EnvironmentGenerator(compatibility)
            
            # Use the create_environment method instead
            result = self.env_generator.create_environment(
                env_name=args.name,
                include_extras=not args.minimal
            )
            
            success = bool(result and result.get('env_file'))
            
            if success:
                print(f"✅ Environment '{args.name}' created successfully!")
                self._print_next_steps(args.name)
                return True
            else:
                print(f"❌ Failed to create environment '{args.name}'")
                return False
                
        except Exception as e:
            print(f"❌ Error during generation: {e}")
            return False
    
    async def _handle_error(self, error: Exception) -> bool:
        """Handle errors with smart recovery"""
        error_message = str(error)
        print(f"\n❌ Error: {error_message}")
        
        if self.error_recovery:
            analysis = self.error_recovery.analyze_error(error_message)
            
            if analysis['identified']:
                print(f"🔍 Issue identified: {analysis['issue']}")
                recovery_steps = self.error_recovery.suggest_recovery(analysis['issue'])
                print("\n🛠️ Suggested recovery:")
                for step in recovery_steps[:3]:  # Show first 3 steps
                    print(f"   {step}")
        
        return False
    
    def _print_performance_summary(self):
        """Print performance metrics"""
        total_time = time.time() - self.start_time
        
        print(f"\n⚡ Performance Summary:")
        print(f"   Total Time: {total_time:.2f}s")
        
        for operation, duration in self.performance_metrics.items():
            percentage = (duration / total_time) * 100
            print(f"   {operation.replace('_', ' ').title()}: {duration:.2f}s ({percentage:.1f}%)")
    
    def _print_next_steps(self, env_name: str):
        """Print next steps"""
        print(f"\n🎉 Next Steps:")
        print(f"1. Activate environment: conda activate {env_name}")
        print(f"2. Test installation: python test_{env_name}.py")
        print(f"3. Start coding!")

async def main():
    """Main CLI entry point"""
    
    parser = argparse.ArgumentParser(description='GPUForge - The Smart GPU Environment Creator')
    
    parser.add_argument('name', nargs='?', help='Environment name')
    parser.add_argument('--framework', choices=['pytorch', 'tensorflow'], 
                       help='ML framework')
    parser.add_argument('--profile', help='Environment profile')
    parser.add_argument('--minimal', action='store_true', help='Minimal environment')
    parser.add_argument('--cpu-only', action='store_true', help='CPU-only environment')
    parser.add_argument('--diagnose', action='store_true', help='Run diagnosis')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    # Cloud support (Phase 1)
    if CLOUD_SUPPORT_AVAILABLE:
        parser.add_argument('--detect-cloud', action='store_true', 
                           help='Detect and optimize for cloud environment (AWS/GCP/Azure)')
        parser.add_argument('--cloud-only', action='store_true',
                           help='Test cloud detection without creating environment')
    
    args = parser.parse_args()
    
    # Handle cloud-only detection
    if CLOUD_SUPPORT_AVAILABLE and hasattr(args, 'cloud_only') and args.cloud_only:
        try:
            print("☁️ Testing cloud detection...")
            cloud_instance = await detect_cloud()
            if cloud_instance:
                print(f"✅ Cloud environment detected!")
                print(f"   Provider: {cloud_instance.provider.upper()}")
                print(f"   Instance: {cloud_instance.instance_type}")
                if cloud_instance.region:
                    print(f"   Region: {cloud_instance.region}")
                if cloud_instance.gpu_detected:
                    print(f"   GPU: {cloud_instance.gpu_count}x {cloud_instance.gpu_type}")
                else:
                    print(f"   GPU: None detected")
                print(f"   Confidence: {cloud_instance.confidence:.1%}")
            else:
                print("💻 Local environment (no cloud detected)")
            return
        except Exception as e:
            print(f"❌ Cloud detection failed: {e}")
            sys.exit(1)
    
    if not args.name:
        parser.print_help()
        return
    
    creator = OptimizedGPUEnvironmentCreator()
    
    if args.diagnose and creator.error_recovery:
        diagnosis = creator.error_recovery.diagnose_system()
        print("🏥 System diagnosis completed")
    
    success = await creator.create_environment_async(args)
    sys.exit(0 if success else 1)

def cli_main():
    """Entry point for console scripts"""
    asyncio.run(main())

if __name__ == "__main__":
    cli_main() 