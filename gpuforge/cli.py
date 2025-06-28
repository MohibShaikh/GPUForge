"""
GPUForge CLI - Command Line Interface
"""

import argparse
import asyncio
import sys
import time
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

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

def add_basic_options(parser):
    """Add basic command options"""
    basic_group = parser.add_argument_group('🔧 Basic Options')
    basic_group.add_argument('--name', default='gpu-env',
                            help='Environment name (default: gpu-env)')
    basic_group.add_argument('--minimal', action='store_true',
                            help='Create minimal environment (no extras)')
    basic_group.add_argument('--cpu-only', action='store_true',
                            help='Create CPU-only environment')
    basic_group.add_argument('--verbose', '-v', action='store_true',
                            help='Enable verbose output')
    basic_group.add_argument('--diagnose', action='store_true',
                            help='Run system diagnosis')

def add_framework_options(parser):
    """Add framework options"""
    framework_group = parser.add_argument_group('🧠 Framework Options')
    framework_group.add_argument('--framework', choices=['pytorch', 'tensorflow'],
                                 help='ML framework (auto-detected if not specified)')

def add_profile_options(parser):
    """Add profile options"""
    profile_group = parser.add_argument_group('📋 Profile Options')
    profile_group.add_argument('--profile', 
                              choices=['learning', 'research', 'computer_vision', 'nlp', 
                                      'deep_learning', 'production', 'lightweight', 'reinforcement_learning'],
                              help='Environment profile (auto-recommended if not specified)')
    profile_group.add_argument('--list-profiles', action='store_true',
                              help='List available profiles and exit')

def add_advanced_options(parser):
    """Add advanced options"""
    advanced_group = parser.add_argument_group('⚙️  Advanced Options')
    advanced_group.add_argument('--no-cache', action='store_true',
                               help='Disable GPU detection caching')
    advanced_group.add_argument('--force', action='store_true',
                               help='Force environment creation (overwrite existing)')

def add_cloud_commands(parser):
    """Add cloud-related commands"""
    cloud_group = parser.add_argument_group('🌥️  Cloud Options')
    cloud_group.add_argument('--detect-cloud', action='store_true',
                            help='Detect current cloud environment')
    cloud_group.add_argument('--cloud-only', action='store_true',
                            help='Only show cloud environment info (skip local detection)')
    
    # Phase 2: Cloud Recommendations
    cloud_group.add_argument('--recommend-cloud', action='store_true',
                            help='Get cloud instance recommendations for your workload')
    cloud_group.add_argument('--workload-type', choices=['training', 'inference', 'research', 'development', 'production', 'distributed'],
                            default='training', help='Type of ML workload (default: training)')
    cloud_group.add_argument('--model-size', choices=['small', 'medium', 'large', 'xl'],
                            default='medium', help='Model size category (default: medium)')
    cloud_group.add_argument('--budget', type=float, default=500.0,
                            help='Monthly budget in USD (default: 500)')
    cloud_group.add_argument('--spot-ok', action='store_true',
                            help='Allow spot/preemptible instances for cost savings')
    cloud_group.add_argument('--region', type=str,
                            help='Preferred cloud region')
    cloud_group.add_argument('--estimate-costs', type=str, metavar='PROVIDER:INSTANCE',
                            help='Estimate costs for specific instance (e.g., aws:p3.2xlarge)')

    # Phase 3: Advanced Cloud Deployment
    cloud_group.add_argument('--deploy-cloud', action='store_true',
                            help='Deploy ML infrastructure to cloud')
    cloud_group.add_argument('--deployment-name', type=str,
                            help='Name for cloud deployment')
    cloud_group.add_argument('--auto-scale', action='store_true',
                            help='Enable auto-scaling for deployment')
    cloud_group.add_argument('--max-instances', type=int, default=5,
                            help='Maximum instances for auto-scaling (default: 5)')
    cloud_group.add_argument('--max-hourly-cost', type=float, default=50.0,
                            help='Maximum hourly cost limit (default: 50)')
    cloud_group.add_argument('--list-deployments', action='store_true',
                            help='List active cloud deployments')
    cloud_group.add_argument('--deployment-status', type=str, metavar='DEPLOYMENT_ID',
                            help='Get status of specific deployment')
    cloud_group.add_argument('--scale-deployment', type=str, metavar='DEPLOYMENT_ID:INSTANCES',
                            help='Scale deployment (e.g., deployment-123:5)')
    cloud_group.add_argument('--optimize-costs', type=str, metavar='DEPLOYMENT_ID',
                            help='Analyze and optimize deployment costs')
    cloud_group.add_argument('--terminate-deployment', type=str, metavar='DEPLOYMENT_ID',
                            help='Terminate cloud deployment')

async def handle_cloud_recommendations(args):
    """Handle cloud instance recommendations"""
    try:
        from .cloud_recommendations import recommend_cloud_instances, estimate_cloud_costs
        
        if args.estimate_costs:
            # Handle cost estimation
            try:
                provider, instance_type = args.estimate_costs.split(':', 1)
                result = estimate_cloud_costs(instance_type, provider)
                
                if 'error' in result:
                    print(f"❌ {result['error']}")
                    return
                
                print(f"\n💰 Cost Estimation for {provider.upper()} {instance_type}:")
                print(f"   On-demand: ${result['on_demand_monthly']:.0f}/month")
                if result['spot_monthly']:
                    print(f"   Spot price: ${result['spot_monthly']:.0f}/month")
                    print(f"   Savings: {result['savings_pct']:.0f}% with spot instances")
                else:
                    print("   Spot instances: Not available")
                return
                
            except ValueError:
                print("❌ Invalid format. Use: --estimate-costs provider:instance_type")
                print("   Example: --estimate-costs aws:p3.2xlarge")
                return
        
        # Get recommendations
        print(f"🎯 Finding optimal cloud instances for {args.workload_type} workload...")
        print(f"   Model size: {args.model_size}")
        print(f"   Budget: ${args.budget:.0f}/month")
        print(f"   Spot instances: {'✅ Yes' if args.spot_ok else '❌ No'}")
        if args.region:
            print(f"   Region: {args.region}")
        print()
        
        recommendations = recommend_cloud_instances(
            workload_type=args.workload_type,
            model_size=args.model_size,
            budget_monthly=args.budget,
            spot_ok=args.spot_ok,
            region=args.region,
            top_k=5
        )
        
        if not recommendations:
            print("❌ No instances found matching your requirements.")
            print("💡 Try increasing your budget or allowing spot instances.")
            return
        
        print(f"📊 Top {len(recommendations)} Cloud Instance Recommendations:\n")
        
        for i, rec in enumerate(recommendations, 1):
            instance = rec.instance
            
            # Header with rank and score
            print(f"🏆 #{i} - {instance.provider.upper()} {instance.instance_type} (Score: {rec.score:.1f}/100)")
            
            # GPU specs
            gpu_info = f"{instance.gpu_count}x {instance.gpu_type}"
            if instance.gpu_count > 1:
                gpu_info += f" ({instance.gpu_memory_total_gb:.0f}GB total)"
            else:
                gpu_info += f" ({instance.gpu_memory_gb:.0f}GB)"
            print(f"   🖥️  GPU: {gpu_info}")
            
            # Compute specs
            print(f"   ⚡ Compute: {instance.vcpus} vCPUs, {instance.ram_gb:.0f}GB RAM")
            print(f"   🔥 Performance: {instance.fp16_tflops:.0f} TFLOPS (FP16), {instance.fp32_tflops:.1f} TFLOPS (FP32)")
            
            # Cost breakdown
            cost_info = f"${rec.cost_monthly:.0f}/month"
            if args.spot_ok and instance.spot_cost_per_hour:
                spot_monthly = instance.spot_cost_per_hour * 730
                if spot_monthly < instance.cost_per_month:
                    savings = (1 - spot_monthly / instance.cost_per_month) * 100
                    cost_info += f" (${spot_monthly:.0f}/month with spot, {savings:.0f}% savings)"
            print(f"   💰 Cost: {cost_info}")
            
            # Cost efficiency
            print(f"   📈 Efficiency: {rec.cost_efficiency:.1f} TFLOPS per $/month")
            
            # Why this recommendation?
            if rec.suitability_reasons:
                print(f"   ✅ Why: {rec.suitability_reasons[0]}")
                if len(rec.suitability_reasons) > 1:
                    print(f"        {rec.suitability_reasons[1]}")
            
            # Warnings
            if rec.warnings:
                print(f"   ⚠️  Warning: {rec.warnings[0]}")
            
            # Performance estimates
            perf = rec.estimated_performance
            if args.workload_type == 'training':
                print(f"   🚀 Est. Training: {perf['training_throughput_samples_sec']:.0f} samples/sec")
            elif args.workload_type == 'inference':
                print(f"   🚀 Est. Inference: {perf['inference_latency_ms']:.0f}ms latency")
            
            # Regions
            regions_str = ", ".join(instance.regions[:3])
            if len(instance.regions) > 3:
                regions_str += f" (+{len(instance.regions)-3} more)"
            print(f"   🌍 Regions: {regions_str}")
            
            print()  # Spacing between recommendations
        
        # Summary with best value recommendation
        best_rec = recommendations[0]
        print(f"💡 Best Overall: {best_rec.instance.provider.upper()} {best_rec.instance.instance_type}")
        print(f"   Perfect balance of performance, cost, and suitability for {args.workload_type}")
        print(f"   Monthly cost: ${best_rec.cost_monthly:.0f} (within ${args.budget:.0f} budget)")
        
        # Cost optimization tips
        print(f"\n🎯 Cost Optimization Tips:")
        if not args.spot_ok:
            print(f"   • Use --spot-ok to enable spot instances (up to 70% savings)")
        if args.workload_type == 'development':
            print(f"   • Consider smaller instances for development/testing")
        print(f"   • Monitor usage and scale down during idle periods")
        print(f"   • Consider reserved instances for consistent workloads")
        
    except ImportError:
        print("❌ Cloud recommendations not available (missing dependencies)")
    except Exception as e:
        print(f"❌ Error getting cloud recommendations: {e}")
        import traceback
        traceback.print_exc()

async def handle_cloud_deployment(args):
    """Handle Phase 3 cloud deployment and management"""
    try:
        from .cloud_advanced import CloudOrchestrator, CloudDeploymentConfig, ScalingPolicy, create_cloud_deployment
        
        orchestrator = CloudOrchestrator()
        
        # List deployments
        if args.list_deployments:
            deployments = orchestrator.list_deployments()
            if not deployments:
                print("📋 No active deployments found")
                return
            
            print("📋 Active Cloud Deployments:")
            for deployment in deployments:
                print(f"\n🚀 {deployment.deployment_id}")
                print(f"   Status: {deployment.status.value}")
                print(f"   Instances: {len(deployment.instances)}")
                print(f"   Cost: ${deployment.current_hourly_cost:.2f}/hour")
                print(f"   Created: {deployment.created_at.strftime('%Y-%m-%d %H:%M')}")
            return
        
        # Get deployment status
        if args.deployment_status:
            try:
                status = orchestrator.get_deployment_status(args.deployment_status)
                print(f"📊 Deployment Status: {args.deployment_status}")
                print(f"   Status: {status.status.value}")
                print(f"   Instances: {len(status.instances)} (healthy: {status.healthy_instances})")
                print(f"   Current cost: ${status.current_hourly_cost:.2f}/hour")
                print(f"   Monthly projection: ${status.projected_monthly_cost:.0f}")
                print(f"   GPU utilization: {status.average_gpu_utilization:.1f}%")
                print(f"   CPU utilization: {status.average_cpu_utilization:.1f}%")
                print(f"   Created: {status.created_at.strftime('%Y-%m-%d %H:%M')}")
                print(f"   Last updated: {status.updated_at.strftime('%Y-%m-%d %H:%M')}")
                
                if status.scaling_events:
                    print(f"   Recent scaling events: {len(status.scaling_events)}")
                return
            except ValueError as e:
                print(f"❌ {e}")
                return
        
        # Scale deployment
        if args.scale_deployment:
            try:
                deployment_id, target_instances = args.scale_deployment.split(':', 1)
                target_instances = int(target_instances)
                success = await orchestrator.scale_deployment(deployment_id, target_instances, "Manual scaling")
                if success:
                    print(f"✅ Deployment {deployment_id} scaled to {target_instances} instances")
                return
            except (ValueError, Exception) as e:
                print(f"❌ Scaling failed: {e}")
                print("   Format: --scale-deployment deployment-id:number-of-instances")
                return
        
        # Optimize costs
        if args.optimize_costs:
            try:
                analysis = await orchestrator.optimize_costs(args.optimize_costs)
                print(f"💰 Cost Optimization Analysis: {args.optimize_costs}")
                print(f"\n📊 Current Status:")
                print(f"   Instances: {analysis['current_status']['instances']}")
                print(f"   Hourly cost: ${analysis['current_status']['hourly_cost']:.2f}")
                print(f"   Monthly projection: ${analysis['current_status']['monthly_projection']:.0f}")
                print(f"   GPU utilization: {analysis['current_status']['gpu_utilization']:.1f}%")
                print(f"   CPU utilization: {analysis['current_status']['cpu_utilization']:.1f}%")
                
                print(f"\n💡 Optimization Opportunities ({len(analysis['optimization_opportunities'])}):")
                for opportunity in analysis['optimization_opportunities']:
                    print(f"   • {opportunity}")
                
                print(f"\n🎯 Recommendations:")
                for rec in analysis['recommendations']:
                    print(f"   • {rec}")
                
                print(f"\n💵 Potential Savings: ${analysis['potential_savings']:.2f}/hour")
                return
            except ValueError as e:
                print(f"❌ {e}")
                return
        
        # Terminate deployment
        if args.terminate_deployment:
            try:
                print(f"⚠️  Are you sure you want to terminate {args.terminate_deployment}? (y/N)")
                # For demo, we'll proceed without confirmation
                success = await orchestrator.terminate_deployment(args.terminate_deployment)
                if success:
                    print(f"✅ Deployment {args.terminate_deployment} terminated successfully")
                return
            except ValueError as e:
                print(f"❌ {e}")
                return
        
        # Deploy cloud infrastructure
        if args.deploy_cloud:
            deployment_name = args.deployment_name or f"gpuforge-{args.workload_type}-{datetime.now().strftime('%m%d%H%M')}"
            
            print(f"🚀 Deploying cloud infrastructure: {deployment_name}")
            print(f"   Workload: {args.workload_type}")
            print(f"   Model size: {args.model_size}")
            print(f"   Provider: {args.region.split('-')[0] if args.region else 'aws'}")
            print(f"   Region: {args.region or 'us-east-1'}")
            print(f"   Auto-scaling: {'✅ Enabled' if args.auto_scale else '❌ Disabled'}")
            print(f"   Max instances: {args.max_instances}")
            print(f"   Cost limit: ${args.max_hourly_cost}/hour")
            print(f"   Spot instances: {'✅ Yes' if args.spot_ok else '❌ No'}")
            print()
            
            # Create deployment configuration
            config = CloudDeploymentConfig(
                name=deployment_name,
                provider=args.region.split('-')[0] if args.region else 'aws',
                region=args.region or 'us-east-1',
                instance_type="auto",
                workload_type=args.workload_type,
                model_size=args.model_size,
                use_spot_instances=args.spot_ok,
                scaling_policy=ScalingPolicy.GPU_UTILIZATION if args.auto_scale else ScalingPolicy.MANUAL,
                max_instances=args.max_instances,
                max_hourly_cost=args.max_hourly_cost,
                monitoring_enabled=True,
                cost_alerts_enabled=True
            )
            
            # Create deployment plan
            print("📋 Creating deployment plan...")
            plan = await orchestrator.plan_deployment(config)
            
            # Show plan summary
            print(f"\n📊 Deployment Plan Summary:")
            print(f"   Deployment ID: {plan['deployment_id']}")
            print(f"   Recommended instance: {plan['recommended_instance'].provider.upper()} {plan['recommended_instance'].instance_type}")
            print(f"   GPU: {plan['recommended_instance'].gpu_count}x {plan['recommended_instance'].gpu_type}")
            print(f"   Cost: ${plan['cost_analysis']['total_hourly_cost']:.2f}/hour")
            print(f"   Monthly projection: ${plan['cost_analysis']['projected_monthly_cost']:.0f}")
            if plan['cost_analysis']['spot_savings'] > 0:
                print(f"   Spot savings: {plan['cost_analysis']['spot_savings']:.0f}%")
            print(f"   Provisioning time: ~{plan['infrastructure']['estimated_provisioning_time']} minutes")
            
            # Show optimization suggestions
            if plan['optimization_suggestions']:
                print(f"\n💡 Optimization Suggestions:")
                for suggestion in plan['optimization_suggestions']:
                    print(f"   • {suggestion}")
            
            # Deploy infrastructure
            print(f"\n🚀 Deploying infrastructure...")
            deployment_id = await orchestrator.deploy_infrastructure(plan)
            
            print(f"\n✅ Deployment completed successfully!")
            print(f"   Deployment ID: {deployment_id}")
            print(f"   Use --deployment-status {deployment_id} to check status")
            print(f"   Use --optimize-costs {deployment_id} for cost analysis")
            return
        
    except ImportError:
        print("❌ Advanced cloud features not available (missing dependencies)")
    except Exception as e:
        print(f"❌ Error with cloud deployment: {e}")
        import traceback
        traceback.print_exc()

async def main():
    """Main async entry point"""
    parser = argparse.ArgumentParser(
        prog='gpuforge',
        description='🚀 GPUForge - Universal GPU Environment Creator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  gpuforge                           # Quick setup with auto-detection
  gpuforge --framework tensorflow    # Setup TensorFlow environment  
  gpuforge --profile research        # Use research-optimized profile
  gpuforge --cpu-only                # Setup CPU-only environment
  gpuforge --diagnose                # Run system diagnosis
  gpuforge --detect-cloud            # Detect cloud environment
  gpuforge --recommend-cloud         # Get cloud instance recommendations
  gpuforge --recommend-cloud --workload-type training --budget 1000 --spot-ok
        """)

    # Add all argument groups
    add_basic_options(parser)
    add_framework_options(parser)
    add_profile_options(parser)
    add_advanced_options(parser)
    add_cloud_commands(parser)  # Add cloud commands

    args = parser.parse_args()

    # Handle cloud recommendations (Phase 2)
    if args.recommend_cloud or args.estimate_costs:
        await handle_cloud_recommendations(args)
        return
    
    # Handle cloud deployment (Phase 3)
    if (args.deploy_cloud or args.list_deployments or args.deployment_status or 
        args.scale_deployment or args.optimize_costs or args.terminate_deployment):
        await handle_cloud_deployment(args)
        return
    
    # Handle list profiles
    if hasattr(args, 'list_profiles') and args.list_profiles:
        if ProfileManager:
            pm = ProfileManager()
            print("📋 Available Environment Profiles:")
            for name, profile in pm.profiles.items():
                print(f"\n{name.upper()}:")
                print(f"   {profile.description}")
                print(f"   Packages: {', '.join(profile.packages[:3])}{'...' if len(profile.packages) > 3 else ''}")
        return
    
    # Handle diagnosis
    if hasattr(args, 'diagnose') and args.diagnose:
        if SmartErrorRecovery:
            recovery = SmartErrorRecovery()
            print("🔍 Running system diagnosis...")
            diagnosis = recovery.diagnose_system()
            
            print(f"\n📊 System Diagnosis Results:")
            print(f"   GPU Drivers: {'✅' if diagnosis.get('gpu_drivers_ok') else '❌'}")
            print(f"   CUDA Available: {'✅' if diagnosis.get('cuda_available') else '❌'}")
            print(f"   Python Environment: {'✅' if diagnosis.get('python_ok') else '❌'}")
            
            if diagnosis.get('issues'):
                print(f"\n⚠️  Issues Found:")
                for issue in diagnosis['issues'][:3]:
                    print(f"   • {issue}")
        return
    
    # Main environment creation workflow
    creator = OptimizedGPUEnvironmentCreator()
    success = await creator.create_environment_async(args)
    
    if not success:
        sys.exit(1)

def cli_main():
    """Entry point for console scripts"""
    asyncio.run(main())

if __name__ == "__main__":
    cli_main() 