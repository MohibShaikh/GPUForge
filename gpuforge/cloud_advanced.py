"""
GPUForge Cloud Advanced - Phase 3: Advanced Cloud Features & Deployment Automation
Enterprise-grade cloud orchestration, auto-scaling, and deployment automation
"""

import json
import asyncio
import logging
import yaml
from typing import Dict, List, Optional, Tuple, NamedTuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path

from .cloud_support import CloudInstance, CloudDetector
from .cloud_recommendations import CloudGPUInstance, CloudRecommendationEngine, WorkloadRequirements

logger = logging.getLogger(__name__)

class DeploymentState(Enum):
    """Deployment status states"""
    PENDING = "pending"
    PROVISIONING = "provisioning"
    RUNNING = "running"
    SCALING = "scaling"
    ERROR = "error"
    STOPPED = "stopped"
    TERMINATED = "terminated"

class ScalingPolicy(Enum):
    """Auto-scaling policies"""
    MANUAL = "manual"
    CPU_BASED = "cpu_based"
    GPU_UTILIZATION = "gpu_utilization"
    QUEUE_LENGTH = "queue_length"
    COST_OPTIMIZED = "cost_optimized"
    TIME_BASED = "time_based"

@dataclass
class CloudDeploymentConfig:
    """Complete cloud deployment configuration"""
    # Basic deployment info
    name: str
    provider: str
    region: str
    instance_type: str
    
    # ML workload configuration
    workload_type: str
    framework: str = "pytorch"
    model_size: str = "medium"
    environment_profile: str = "research"
    
    # Scaling configuration
    min_instances: int = 1
    max_instances: int = 10
    scaling_policy: ScalingPolicy = ScalingPolicy.MANUAL
    target_gpu_utilization: float = 70.0
    target_cpu_utilization: float = 80.0
    
    # Cost management
    max_hourly_cost: float = 100.0
    use_spot_instances: bool = False
    cost_alerts_enabled: bool = True
    
    # Storage configuration
    storage_type: str = "ssd"
    storage_size_gb: int = 100
    backup_enabled: bool = True
    
    # Security configuration
    ssh_key_name: Optional[str] = None
    security_group_rules: List[Dict] = field(default_factory=list)
    encryption_enabled: bool = True
    
    # Monitoring configuration
    monitoring_enabled: bool = True
    log_retention_days: int = 30
    alerts_email: Optional[str] = None
    
    # Advanced features
    preemptible_fallback: bool = True
    multi_zone_deployment: bool = False
    load_balancer_enabled: bool = False
    
    def __post_init__(self):
        if not self.security_group_rules:
            self.security_group_rules = [
                {"protocol": "tcp", "port": 22, "source": "0.0.0.0/0", "description": "SSH"},
                {"protocol": "tcp", "port": 8888, "source": "0.0.0.0/0", "description": "Jupyter"},
                {"protocol": "tcp", "port": 6006, "source": "0.0.0.0/0", "description": "TensorBoard"}
            ]

@dataclass
class DeploymentStatus:
    """Deployment status and metrics"""
    deployment_id: str
    status: DeploymentState
    instances: List[Dict]
    created_at: datetime
    updated_at: datetime
    
    # Cost tracking
    current_hourly_cost: float = 0.0
    total_cost_to_date: float = 0.0
    projected_monthly_cost: float = 0.0
    
    # Performance metrics
    average_gpu_utilization: float = 0.0
    average_cpu_utilization: float = 0.0
    total_compute_hours: float = 0.0
    
    # Scaling metrics
    scaling_events: List[Dict] = field(default_factory=list)
    last_scale_action: Optional[datetime] = None
    
    # Health metrics
    healthy_instances: int = 0
    unhealthy_instances: int = 0
    pending_instances: int = 0

@dataclass
class CloudTemplate:
    """Infrastructure as Code templates"""
    name: str
    provider: str
    template_type: str  # terraform, cloudformation, gcp_deployment
    template_content: str
    variables: Dict = field(default_factory=dict)
    description: str = ""

class CloudOrchestrator:
    """Advanced cloud orchestration and deployment automation"""
    
    def __init__(self):
        self.recommendation_engine = CloudRecommendationEngine()
        self.active_deployments = {}
        self.deployment_history = []
        self.templates = self._load_templates()
    
    def _load_templates(self) -> Dict[str, CloudTemplate]:
        """Load Infrastructure as Code templates"""
        return {
            "aws_single_gpu": CloudTemplate(
                name="aws_single_gpu",
                provider="aws",
                template_type="terraform",
                template_content=self._get_aws_single_gpu_template(),
                description="Single GPU instance on AWS with auto-scaling"
            ),
            "aws_multi_gpu_cluster": CloudTemplate(
                name="aws_multi_gpu_cluster",
                provider="aws", 
                template_type="terraform",
                template_content=self._get_aws_multi_gpu_template(),
                description="Multi-GPU cluster with distributed training setup"
            ),
            "gcp_preemptible_training": CloudTemplate(
                name="gcp_preemptible_training",
                provider="gcp",
                template_type="gcp_deployment",
                template_content=self._get_gcp_preemptible_template(),
                description="Cost-optimized preemptible training cluster"
            ),
            "azure_spot_inference": CloudTemplate(
                name="azure_spot_inference",
                provider="azure",
                template_type="arm_template",
                template_content=self._get_azure_spot_template(),
                description="Azure spot instances for inference workloads"
            )
        }
    
    async def plan_deployment(self, config: CloudDeploymentConfig) -> Dict:
        """Create deployment plan with cost estimation and optimization recommendations"""
        
        # Get instance recommendations
        from .cloud_recommendations import WorkloadType, ModelSize
        
        requirements = WorkloadRequirements(
            workload_type=WorkloadType(config.workload_type),
            model_size=ModelSize(config.model_size),
            framework=config.framework,
            budget_monthly=config.max_hourly_cost * 730,
            spot_instances_ok=config.use_spot_instances,
            region_preference=config.region
        )
        
        recommendations = self.recommendation_engine.recommend_instances(requirements, top_k=3)
        
        if not recommendations:
            raise ValueError(f"No suitable instances found for deployment in {config.region}")
        
        best_instance = recommendations[0].instance
        
        # Calculate deployment costs
        hourly_cost = best_instance.spot_cost_per_hour if config.use_spot_instances and best_instance.spot_cost_per_hour else best_instance.cost_per_hour
        storage_cost_per_hour = self._calculate_storage_cost(config.storage_size_gb, config.provider)
        
        total_hourly_cost = (hourly_cost + storage_cost_per_hour) * config.min_instances
        projected_monthly_cost = total_hourly_cost * 730
        
        # Generate deployment plan
        deployment_plan = {
            "deployment_id": f"gpuforge-{config.name}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "config": config,
            "recommended_instance": best_instance,
            "cost_analysis": {
                "hourly_cost_per_instance": hourly_cost,
                "storage_cost_per_hour": storage_cost_per_hour,
                "total_hourly_cost": total_hourly_cost,
                "projected_monthly_cost": projected_monthly_cost,
                "max_monthly_cost": total_hourly_cost * config.max_instances * 730,
                "spot_savings": ((best_instance.cost_per_hour - hourly_cost) / best_instance.cost_per_hour * 100) if config.use_spot_instances else 0
            },
            "infrastructure": {
                "template": self._select_template(config),
                "estimated_provisioning_time": self._estimate_provisioning_time(config),
                "scaling_configuration": self._generate_scaling_config(config)
            },
            "optimization_suggestions": self._generate_optimization_suggestions(config, best_instance),
            "security_considerations": self._generate_security_recommendations(config),
            "monitoring_setup": self._generate_monitoring_config(config)
        }
        
        return deployment_plan
    
    async def deploy_infrastructure(self, deployment_plan: Dict) -> str:
        """Deploy cloud infrastructure based on plan"""
        
        deployment_id = deployment_plan["deployment_id"]
        config = deployment_plan["config"]
        
        print(f"🚀 Starting deployment: {deployment_id}")
        
        # Create deployment status
        deployment_status = DeploymentStatus(
            deployment_id=deployment_id,
            status=DeploymentState.PROVISIONING,
            instances=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.active_deployments[deployment_id] = deployment_status
        
        try:
            # Phase 1: Infrastructure provisioning
            print("📦 Phase 1: Provisioning infrastructure...")
            await self._provision_infrastructure(deployment_plan)
            
            # Phase 2: Environment setup
            print("🔧 Phase 2: Setting up ML environment...")
            await self._setup_ml_environment(deployment_plan)
            
            # Phase 3: Configure monitoring
            print("📊 Phase 3: Configuring monitoring and logging...")
            await self._setup_monitoring(deployment_plan)
            
            # Phase 4: Enable auto-scaling
            if config.scaling_policy != ScalingPolicy.MANUAL:
                print("📈 Phase 4: Enabling auto-scaling...")
                await self._setup_autoscaling(deployment_plan)
            
            # Phase 5: Security hardening
            print("🔒 Phase 5: Applying security configurations...")
            await self._apply_security_configs(deployment_plan)
            
            deployment_status.status = DeploymentState.RUNNING
            deployment_status.updated_at = datetime.now()
            
            # Simulate running instances
            deployment_status.instances = [
                {
                    "instance_id": f"i-{deployment_id[-8:]}-001",
                    "status": "running",
                    "launched_at": datetime.now(),
                    "instance_type": deployment_plan["recommended_instance"].instance_type
                }
            ]
            deployment_status.healthy_instances = 1
            deployment_status.current_hourly_cost = deployment_plan["cost_analysis"]["total_hourly_cost"]
            deployment_status.projected_monthly_cost = deployment_plan["cost_analysis"]["projected_monthly_cost"]
            
            print(f"✅ Deployment {deployment_id} completed successfully!")
            
            return deployment_id
            
        except Exception as e:
            deployment_status.status = DeploymentState.ERROR
            deployment_status.updated_at = datetime.now()
            print(f"❌ Deployment failed: {e}")
            raise
    
    async def scale_deployment(self, deployment_id: str, target_instances: int, reason: str = "manual") -> bool:
        """Scale deployment up or down"""
        
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self.active_deployments[deployment_id]
        current_instances = len(deployment.instances)
        
        print(f"📈 Scaling deployment {deployment_id}: {current_instances} → {target_instances} instances")
        print(f"   Reason: {reason}")
        
        # Simulate scaling operation
        await asyncio.sleep(2)  # Simulate scaling time
        
        # Update deployment status
        deployment.status = DeploymentState.SCALING
        deployment.scaling_events.append({
            "timestamp": datetime.now(),
            "from_instances": current_instances,
            "to_instances": target_instances,
            "reason": reason
        })
        deployment.last_scale_action = datetime.now()
        
        # Update instances list
        deployment.instances = []
        for i in range(target_instances):
            deployment.instances.append({
                "instance_id": f"i-{deployment_id[-8:]}-{i+1:03d}",
                "status": "running",
                "launched_at": datetime.now(),
                "instance_type": "placeholder"
            })
        
        deployment.healthy_instances = target_instances
        deployment.status = DeploymentState.RUNNING
        deployment.updated_at = datetime.now()
        
        # Update costs
        deployment.current_hourly_cost *= (target_instances / current_instances) if current_instances > 0 else target_instances
        deployment.projected_monthly_cost = deployment.current_hourly_cost * 730
        
        print(f"✅ Scaling completed: {target_instances} instances running")
        return True
    
    async def optimize_costs(self, deployment_id: str) -> Dict:
        """Analyze and optimize deployment costs"""
        
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self.active_deployments[deployment_id]
        
        # Simulate some usage data
        deployment.average_gpu_utilization = 45.0  # 45% utilization
        deployment.average_cpu_utilization = 25.0  # 25% utilization
        deployment.total_compute_hours = 24.5     # 24.5 hours of compute
        
        # Analyze current costs and usage
        analysis = {
            "current_status": {
                "instances": len(deployment.instances),
                "hourly_cost": deployment.current_hourly_cost,
                "monthly_projection": deployment.projected_monthly_cost,
                "gpu_utilization": deployment.average_gpu_utilization,
                "cpu_utilization": deployment.average_cpu_utilization
            },
            "optimization_opportunities": [],
            "potential_savings": 0.0,
            "recommendations": []
        }
        
        # Check for underutilized resources
        if deployment.average_gpu_utilization < 50:
            analysis["optimization_opportunities"].append("Low GPU utilization detected")
            analysis["recommendations"].append("Consider scaling down or using smaller instances")
            analysis["potential_savings"] += deployment.current_hourly_cost * 0.3
        
        if deployment.average_cpu_utilization < 30:
            analysis["optimization_opportunities"].append("Low CPU utilization detected")
            analysis["recommendations"].append("Switch to GPU-optimized instances with fewer CPUs")
            analysis["potential_savings"] += deployment.current_hourly_cost * 0.2
        
        # Check spot instance opportunities
        if not any("spot" in str(inst) for inst in deployment.instances):
            analysis["optimization_opportunities"].append("Not using spot instances")
            analysis["recommendations"].append("Migrate to spot instances for 60-70% cost savings")
            analysis["potential_savings"] += deployment.current_hourly_cost * 0.65
        
        # Check for idle instances
        idle_hours = 8  # Simulate 8 hours of idle time detected
        if idle_hours > 4:
            analysis["optimization_opportunities"].append(f"Instances idle for {idle_hours} hours")
            analysis["recommendations"].append("Implement auto-shutdown for idle instances")
            analysis["potential_savings"] += deployment.current_hourly_cost * (idle_hours / 24)
        
        return analysis
    
    def get_deployment_status(self, deployment_id: str) -> DeploymentStatus:
        """Get current deployment status"""
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        return self.active_deployments[deployment_id]
    
    def list_deployments(self) -> List[DeploymentStatus]:
        """List all active deployments"""
        return list(self.active_deployments.values())
    
    async def terminate_deployment(self, deployment_id: str, backup_data: bool = True) -> bool:
        """Safely terminate deployment with optional data backup"""
        
        if deployment_id not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment = self.active_deployments[deployment_id]
        
        print(f"🛑 Terminating deployment {deployment_id}...")
        
        if backup_data:
            print("💾 Creating data backup...")
            await asyncio.sleep(2)  # Simulate backup time
            print("✅ Data backup completed")
        
        # Simulate termination
        await asyncio.sleep(1)
        
        deployment.status = DeploymentState.TERMINATED
        deployment.updated_at = datetime.now()
        
        # Move to history
        self.deployment_history.append(deployment)
        del self.active_deployments[deployment_id]
        
        print(f"✅ Deployment {deployment_id} terminated successfully")
        return True
    
    # Helper methods
    def _calculate_storage_cost(self, size_gb: int, provider: str) -> float:
        """Calculate storage costs per hour"""
        # Rough estimates per GB per hour
        storage_rates = {
            "aws": 0.10 / 730,    # $0.10/GB/month
            "gcp": 0.08 / 730,    # $0.08/GB/month  
            "azure": 0.12 / 730   # $0.12/GB/month
        }
        return size_gb * storage_rates.get(provider, 0.10 / 730)
    
    def _select_template(self, config: CloudDeploymentConfig) -> str:
        """Select appropriate infrastructure template"""
        if config.max_instances > 1:
            return f"{config.provider}_multi_gpu_cluster"
        elif config.use_spot_instances:
            return f"{config.provider}_spot_inference"
        else:
            return f"{config.provider}_single_gpu"
    
    def _estimate_provisioning_time(self, config: CloudDeploymentConfig) -> int:
        """Estimate provisioning time in minutes"""
        base_time = 5  # 5 minutes base
        if config.max_instances > 3:
            base_time += 10  # Additional time for clusters
        if config.monitoring_enabled:
            base_time += 3   # Additional time for monitoring setup
        return base_time
    
    def _generate_scaling_config(self, config: CloudDeploymentConfig) -> Dict:
        """Generate auto-scaling configuration"""
        return {
            "policy": config.scaling_policy.value,
            "min_instances": config.min_instances,
            "max_instances": config.max_instances,
            "target_gpu_utilization": config.target_gpu_utilization,
            "target_cpu_utilization": config.target_cpu_utilization,
            "scale_up_cooldown": 300,    # 5 minutes
            "scale_down_cooldown": 600,  # 10 minutes
            "metrics_evaluation_period": 60  # 1 minute
        }
    
    def _generate_optimization_suggestions(self, config: CloudDeploymentConfig, instance: CloudGPUInstance) -> List[str]:
        """Generate cost and performance optimization suggestions"""
        suggestions = []
        
        if not config.use_spot_instances and instance.spot_cost_per_hour:
            savings = (1 - instance.spot_cost_per_hour / instance.cost_per_hour) * 100
            suggestions.append(f"Enable spot instances for {savings:.0f}% cost savings")
        
        if config.scaling_policy == ScalingPolicy.MANUAL:
            suggestions.append("Enable auto-scaling to optimize costs and performance")
        
        if not config.backup_enabled:
            suggestions.append("Enable automated backups for data protection")
        
        if config.storage_type == "ssd" and config.workload_type == "training":
            suggestions.append("Consider high-performance NVMe storage for training workloads")
        
        return suggestions
    
    def _generate_security_recommendations(self, config: CloudDeploymentConfig) -> List[str]:
        """Generate security recommendations"""
        recommendations = []
        
        if not config.encryption_enabled:
            recommendations.append("Enable encryption at rest and in transit")
        
        if not config.ssh_key_name:
            recommendations.append("Configure SSH key authentication")
        
        # Check for overly permissive security groups
        for rule in config.security_group_rules:
            if rule.get("source") == "0.0.0.0/0" and rule.get("port") != 22:
                recommendations.append(f"Restrict access to port {rule['port']} - currently open to all")
        
        recommendations.append("Implement network segmentation and VPC isolation")
        recommendations.append("Enable audit logging and monitoring")
        
        return recommendations
    
    def _generate_monitoring_config(self, config: CloudDeploymentConfig) -> Dict:
        """Generate monitoring configuration"""
        return {
            "metrics_enabled": config.monitoring_enabled,
            "log_retention_days": config.log_retention_days,
            "alerts_email": config.alerts_email,
            "dashboards": [
                "GPU Utilization",
                "Cost Tracking", 
                "Performance Metrics",
                "System Health"
            ],
            "alerts": [
                {"metric": "gpu_utilization", "threshold": 90, "duration": "5m"},
                {"metric": "cost_per_hour", "threshold": config.max_hourly_cost, "duration": "1m"},
                {"metric": "instance_health", "threshold": 1, "duration": "2m"}
            ]
        }
    
    # Infrastructure provisioning simulation methods
    async def _provision_infrastructure(self, plan: Dict):
        """Simulate infrastructure provisioning"""
        await asyncio.sleep(3)  # Simulate provisioning time
    
    async def _setup_ml_environment(self, plan: Dict):
        """Simulate ML environment setup"""
        await asyncio.sleep(2)
    
    async def _setup_monitoring(self, plan: Dict):
        """Simulate monitoring setup"""
        await asyncio.sleep(1)
    
    async def _setup_autoscaling(self, plan: Dict):
        """Simulate auto-scaling setup"""
        await asyncio.sleep(1)
    
    async def _apply_security_configs(self, plan: Dict):
        """Simulate security configuration"""
        await asyncio.sleep(1)
    
    # Template generation methods (simplified for demo)
    def _get_aws_single_gpu_template(self) -> str:
        return """
resource "aws_instance" "gpu_instance" {
  ami           = var.gpu_ami_id
  instance_type = var.instance_type
  key_name      = var.ssh_key_name
  
  vpc_security_group_ids = [aws_security_group.gpu_sg.id]
  
  user_data = templatefile("setup_ml_env.sh", {
    framework = var.framework
    profile   = var.environment_profile
  })
  
  tags = {
    Name = "GPUForge-${var.deployment_name}"
    Environment = var.environment_profile
    ManagedBy = "GPUForge"
  }
}
"""
    
    def _get_aws_multi_gpu_template(self) -> str:
        return """
resource "aws_launch_template" "gpu_cluster" {
  name_prefix   = "gpuforge-cluster-"
  image_id      = var.gpu_ami_id
  instance_type = var.instance_type
  key_name      = var.ssh_key_name
  
  vpc_security_group_ids = [aws_security_group.gpu_cluster_sg.id]
  
  user_data = base64encode(templatefile("setup_distributed_training.sh", {
    framework = var.framework
    cluster_size = var.max_instances
  }))
}

resource "aws_autoscaling_group" "gpu_cluster_asg" {
  name                = "gpuforge-cluster-${var.deployment_name}"
  vpc_zone_identifier = var.subnet_ids
  target_group_arns   = [aws_lb_target_group.gpu_cluster_tg.arn]
  
  min_size         = var.min_instances
  max_size         = var.max_instances
  desired_capacity = var.min_instances
  
  launch_template {
    id      = aws_launch_template.gpu_cluster.id
    version = "$Latest"
  }
}
"""
    
    def _get_gcp_preemptible_template(self) -> str:
        return """
resources:
- name: gpu-training-cluster
  type: compute.v1.instanceGroupManager
  properties:
    zone: {{ properties['zone'] }}
    targetSize: {{ properties['min_instances'] }}
    baseInstanceName: gpuforge-training
    instanceTemplate: $(ref.gpu-training-template.selfLink)
    
- name: gpu-training-template
  type: compute.v1.instanceTemplate
  properties:
    properties:
      machineType: {{ properties['machine_type'] }}
      scheduling:
        preemptible: true
      disks:
      - deviceName: boot
        type: PERSISTENT
        boot: true
        autoDelete: true
        initializeParams:
          sourceImage: {{ properties['source_image'] }}
      metadata:
        items:
        - key: startup-script
          value: |
            #!/bin/bash
            # Install ML environment
            curl -sSL https://gpuforge.dev/setup.sh | bash
"""
    
    def _get_azure_spot_template(self) -> str:
        return """
{
  "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
  "contentVersion": "1.0.0.0",
  "parameters": {
    "vmSize": {
      "type": "string",
      "defaultValue": "Standard_NC6s_v3"
    }
  },
  "resources": [
    {
      "type": "Microsoft.Compute/virtualMachineScaleSets",
      "apiVersion": "2019-12-01",
      "name": "gpuforge-spot-inference",
      "location": "[resourceGroup().location]",
      "properties": {
        "upgradePolicy": {
          "mode": "Manual"
        },
        "virtualMachineProfile": {
          "priority": "Spot",
          "evictionPolicy": "Delete",
          "billingProfile": {
            "maxPrice": -1
          },
          "osProfile": {
            "computerNamePrefix": "gpuforge",
            "adminUsername": "azureuser",
            "customData": "[base64('#!/bin/bash\\ncurl -sSL https://gpuforge.dev/setup.sh | bash')]"
          },
          "storageProfile": {
            "imageReference": {
              "publisher": "microsoft-dsvm",
              "offer": "ubuntu-1804",
              "sku": "1804-gen2",
              "version": "latest"
            }
          },
          "hardwareProfile": {
            "vmSize": "[parameters('vmSize')]"
          }
        }
      }
    }
  ]
}
"""

# High-level API functions
async def create_cloud_deployment(
    name: str,
    workload_type: str,
    provider: str = "aws",
    region: str = "us-east-1",
    instance_type: Optional[str] = None,
    use_spot: bool = False,
    auto_scale: bool = True,
    max_hourly_cost: float = 50.0
) -> str:
    """High-level function to create a cloud deployment"""
    
    orchestrator = CloudOrchestrator()
    
    config = CloudDeploymentConfig(
        name=name,
        provider=provider,
        region=region,
        instance_type=instance_type or "auto",  # Will be selected automatically
        workload_type=workload_type,
        model_size="medium",  # Default model size
        use_spot_instances=use_spot,
        scaling_policy=ScalingPolicy.GPU_UTILIZATION if auto_scale else ScalingPolicy.MANUAL,
        max_hourly_cost=max_hourly_cost,
        monitoring_enabled=True,
        cost_alerts_enabled=True
    )
    
    # Create deployment plan
    plan = await orchestrator.plan_deployment(config)
    
    # Deploy infrastructure
    deployment_id = await orchestrator.deploy_infrastructure(plan)
    
    return deployment_id

async def optimize_deployment_costs(deployment_id: str) -> Dict:
    """Optimize costs for existing deployment"""
    orchestrator = CloudOrchestrator()
    return await orchestrator.optimize_costs(deployment_id)

if __name__ == "__main__":
    # Test Phase 3 functionality
    async def test_phase3():
        print("🚀 GPUForge Phase 3: Advanced Cloud Features")
        print("=" * 60)
        
        # Test deployment planning
        orchestrator = CloudOrchestrator()
        
        config = CloudDeploymentConfig(
            name="test-training-cluster",
            provider="aws",
            region="us-east-1", 
            instance_type="auto",
            workload_type="training",
            model_size="large",
            use_spot_instances=True,
            scaling_policy=ScalingPolicy.GPU_UTILIZATION,
            max_instances=5,
            max_hourly_cost=100.0
        )
        
        print("📋 Creating deployment plan...")
        plan = await orchestrator.plan_deployment(config)
        
        print(f"✅ Plan created for: {plan['deployment_id']}")
        print(f"   Instance: {plan['recommended_instance'].provider} {plan['recommended_instance'].instance_type}")
        print(f"   Cost: ${plan['cost_analysis']['total_hourly_cost']:.2f}/hour")
        print(f"   Spot savings: {plan['cost_analysis']['spot_savings']:.0f}%")
        
        print("\n🚀 Deploying infrastructure...")
        deployment_id = await orchestrator.deploy_infrastructure(plan)
        
        print(f"\n📊 Deployment status:")
        status = orchestrator.get_deployment_status(deployment_id)
        print(f"   Status: {status.status}")
        print(f"   Instances: {len(status.instances)}")
        
        print(f"\n📈 Testing auto-scaling...")
        await orchestrator.scale_deployment(deployment_id, 3, "Load spike detected")
        
        print(f"\n💰 Cost optimization analysis...")
        optimization = await orchestrator.optimize_costs(deployment_id)
        print(f"   Potential savings: ${optimization['potential_savings']:.2f}/hour")
        print(f"   Opportunities: {len(optimization['optimization_opportunities'])}")
        
        print(f"\n🛑 Terminating deployment...")
        await orchestrator.terminate_deployment(deployment_id)
        
        print(f"\n✅ Phase 3 testing complete!")
    
    asyncio.run(test_phase3()) 