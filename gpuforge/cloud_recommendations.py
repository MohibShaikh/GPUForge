"""
GPUForge Cloud Recommendations - Phase 2: Instance Recommendations & Cost Optimization
Smart workload-to-instance matching with real-time cost analysis
"""

import json
import asyncio
import logging
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

from .cloud_support import CloudInstance, CloudDetector

logger = logging.getLogger(__name__)

class WorkloadType(Enum):
    """ML Workload types"""
    TRAINING = "training"
    INFERENCE = "inference"
    RESEARCH = "research"
    DEVELOPMENT = "development"
    PRODUCTION = "production"
    DISTRIBUTED = "distributed"

class ModelSize(Enum):
    """Model size categories"""
    SMALL = "small"      # <1B parameters
    MEDIUM = "medium"    # 1-7B parameters  
    LARGE = "large"      # 7-70B parameters
    XL = "xl"           # 70B+ parameters

@dataclass
class WorkloadRequirements:
    """ML workload specification"""
    workload_type: WorkloadType
    model_size: ModelSize
    framework: str = "pytorch"
    batch_size: int = 32
    concurrent_users: int = 1
    budget_monthly: float = 1000.0
    region_preference: Optional[str] = None
    spot_instances_ok: bool = False
    min_gpu_memory_gb: float = 4.0
    max_latency_ms: int = 1000

@dataclass
class CloudGPUInstance:
    """Enhanced cloud instance with pricing and performance data"""
    provider: str
    instance_type: str
    gpu_type: str
    gpu_count: int
    gpu_memory_gb: float
    vcpus: int
    ram_gb: float
    storage_gb: int
    
    # Pricing
    cost_per_hour: float
    spot_cost_per_hour: Optional[float] = None
    
    # Computed fields (set in __post_init__)
    gpu_memory_total_gb: float = 0.0  # gpu_memory_gb * gpu_count
    cost_per_month: float = 0.0  # Assuming 730 hours/month
    
    # Performance metrics
    compute_capability: float = 7.0
    memory_bandwidth_gbps: float = 500.0
    fp16_tflops: float = 100.0
    fp32_tflops: float = 50.0
    
    # Availability
    regions: List[str] = None
    availability_zones: List[str] = None
    
    # ML-specific scoring
    training_score: float = 0.0
    inference_score: float = 0.0
    cost_efficiency_score: float = 0.0
    
    def __post_init__(self):
        if self.regions is None:
            self.regions = []
        if self.availability_zones is None:
            self.availability_zones = []
        self.gpu_memory_total_gb = self.gpu_memory_gb * self.gpu_count
        self.cost_per_month = self.cost_per_hour * 730  # 730 hours/month average

@dataclass
class InstanceRecommendation:
    """Instance recommendation with scoring and reasoning"""
    instance: CloudGPUInstance
    score: float  # 0-100
    cost_monthly: float
    cost_efficiency: float  # Performance per dollar
    suitability_reasons: List[str]
    warnings: List[str]
    estimated_performance: Dict[str, float]

class CloudInstanceDatabase:
    """Comprehensive cloud instance database with real pricing"""
    
    def __init__(self):
        self.instances = self._build_instance_database()
    
    def _build_instance_database(self) -> List[CloudGPUInstance]:
        """Build comprehensive instance database"""
        instances = []
        
        # AWS Instances
        instances.extend(self._get_aws_instances())
        
        # GCP Instances  
        instances.extend(self._get_gcp_instances())
        
        # Azure Instances
        instances.extend(self._get_azure_instances())
        
        return instances
    
    def _get_aws_instances(self) -> List[CloudGPUInstance]:
        """AWS GPU instances with current pricing"""
        return [
            # P3 Instances (Tesla V100)
            CloudGPUInstance(
                provider="aws", instance_type="p3.2xlarge",
                gpu_type="Tesla V100", gpu_count=1, gpu_memory_gb=16.0,
                vcpus=8, ram_gb=61.0, storage_gb=0,
                cost_per_hour=3.06, spot_cost_per_hour=0.918,
                compute_capability=7.0, memory_bandwidth_gbps=900,
                fp16_tflops=125, fp32_tflops=15.7,
                regions=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]
            ),
            CloudGPUInstance(
                provider="aws", instance_type="p3.8xlarge", 
                gpu_type="Tesla V100", gpu_count=4, gpu_memory_gb=16.0,
                vcpus=32, ram_gb=244.0, storage_gb=0,
                cost_per_hour=12.24, spot_cost_per_hour=3.672,
                compute_capability=7.0, memory_bandwidth_gbps=900,
                fp16_tflops=500, fp32_tflops=62.8,
                regions=["us-east-1", "us-west-2", "eu-west-1"]
            ),
            
            # P4 Instances (A100)
            CloudGPUInstance(
                provider="aws", instance_type="p4d.24xlarge",
                gpu_type="A100", gpu_count=8, gpu_memory_gb=40.0,
                vcpus=96, ram_gb=1152.0, storage_gb=8000,
                cost_per_hour=32.77, spot_cost_per_hour=9.831,
                compute_capability=8.0, memory_bandwidth_gbps=1555,
                fp16_tflops=1248, fp32_tflops=312,
                regions=["us-east-1", "us-west-2", "eu-west-1"]
            ),
            
            # G4 Instances (Tesla T4)
            CloudGPUInstance(
                provider="aws", instance_type="g4dn.xlarge",
                gpu_type="Tesla T4", gpu_count=1, gpu_memory_gb=16.0,
                vcpus=4, ram_gb=16.0, storage_gb=125,
                cost_per_hour=0.526, spot_cost_per_hour=0.158,
                compute_capability=7.5, memory_bandwidth_gbps=320,
                fp16_tflops=65, fp32_tflops=8.1,
                regions=["us-east-1", "us-west-2", "eu-west-1", "ap-southeast-1"]
            ),
            CloudGPUInstance(
                provider="aws", instance_type="g4dn.12xlarge",
                gpu_type="Tesla T4", gpu_count=4, gpu_memory_gb=16.0,
                vcpus=48, ram_gb=192.0, storage_gb=900,
                cost_per_hour=3.912, spot_cost_per_hour=1.174,
                compute_capability=7.5, memory_bandwidth_gbps=320,
                fp16_tflops=260, fp32_tflops=32.4,
                regions=["us-east-1", "us-west-2", "eu-west-1"]
            ),
            
            # G5 Instances (A10G)
            CloudGPUInstance(
                provider="aws", instance_type="g5.xlarge",
                gpu_type="A10G", gpu_count=1, gpu_memory_gb=24.0,
                vcpus=4, ram_gb=16.0, storage_gb=250,
                cost_per_hour=1.006, spot_cost_per_hour=0.302,
                compute_capability=8.6, memory_bandwidth_gbps=600,
                fp16_tflops=70, fp32_tflops=35,
                regions=["us-east-1", "us-west-2", "eu-west-1"]
            ),
        ]
    
    def _get_gcp_instances(self) -> List[CloudGPUInstance]:
        """GCP GPU instances"""
        return [
            # N1 with Tesla T4
            CloudGPUInstance(
                provider="gcp", instance_type="n1-standard-4-t4",
                gpu_type="Tesla T4", gpu_count=1, gpu_memory_gb=16.0,
                vcpus=4, ram_gb=15.0, storage_gb=100,
                cost_per_hour=0.74, spot_cost_per_hour=0.22,
                compute_capability=7.5, memory_bandwidth_gbps=320,
                fp16_tflops=65, fp32_tflops=8.1,
                regions=["us-central1", "us-west1", "europe-west1"]
            ),
            
            # N1 with Tesla V100
            CloudGPUInstance(
                provider="gcp", instance_type="n1-standard-8-v100",
                gpu_type="Tesla V100", gpu_count=1, gpu_memory_gb=16.0,
                vcpus=8, ram_gb=30.0, storage_gb=100,
                cost_per_hour=2.48, spot_cost_per_hour=0.74,
                compute_capability=7.0, memory_bandwidth_gbps=900,
                fp16_tflops=125, fp32_tflops=15.7,
                regions=["us-central1", "us-west1", "europe-west1"]
            ),
            
            # A2 with A100
            CloudGPUInstance(
                provider="gcp", instance_type="a2-highgpu-1g",
                gpu_type="A100", gpu_count=1, gpu_memory_gb=40.0,
                vcpus=12, ram_gb=85.0, storage_gb=100,
                cost_per_hour=3.67, spot_cost_per_hour=1.10,
                compute_capability=8.0, memory_bandwidth_gbps=1555,
                fp16_tflops=156, fp32_tflops=39,
                regions=["us-central1", "us-west1", "europe-west1"]
            ),
        ]
    
    def _get_azure_instances(self) -> List[CloudGPUInstance]:
        """Azure GPU instances"""
        return [
            # NC v3 Series (Tesla V100)
            CloudGPUInstance(
                provider="azure", instance_type="Standard_NC6s_v3",
                gpu_type="Tesla V100", gpu_count=1, gpu_memory_gb=16.0,
                vcpus=6, ram_gb=112.0, storage_gb=736,
                cost_per_hour=3.168, spot_cost_per_hour=0.95,
                compute_capability=7.0, memory_bandwidth_gbps=900,
                fp16_tflops=125, fp32_tflops=15.7,
                regions=["eastus", "westus2", "westeurope"]
            ),
            
            # ND A100 v4 Series
            CloudGPUInstance(
                provider="azure", instance_type="Standard_ND96asr_v4",
                gpu_type="A100", gpu_count=8, gpu_memory_gb=40.0,
                vcpus=96, ram_gb=900.0, storage_gb=6000,
                cost_per_hour=33.27, spot_cost_per_hour=9.98,
                compute_capability=8.0, memory_bandwidth_gbps=1555,
                fp16_tflops=1248, fp32_tflops=312,
                regions=["eastus", "westus2", "westeurope"]
            ),
        ]
    
    def get_instances_by_provider(self, provider: str) -> List[CloudGPUInstance]:
        """Get instances for specific provider"""
        return [i for i in self.instances if i.provider == provider]
    
    def get_instances_by_gpu_type(self, gpu_type: str) -> List[CloudGPUInstance]:
        """Get instances with specific GPU type"""
        return [i for i in self.instances if i.gpu_type == gpu_type]
    
    def get_instances_in_budget(self, max_monthly_cost: float, spot_ok: bool = False) -> List[CloudGPUInstance]:
        """Get instances within budget"""
        instances = []
        for instance in self.instances:
            cost = instance.cost_per_month
            if spot_ok and instance.spot_cost_per_hour:
                cost = min(cost, instance.spot_cost_per_hour * 730)
            if cost <= max_monthly_cost:
                instances.append(instance)
        return instances

class CloudRecommendationEngine:
    """Advanced ML workload to cloud instance recommendation engine"""
    
    def __init__(self):
        self.db = CloudInstanceDatabase()
        self.workload_requirements = {
            # GPU memory requirements by model size
            ModelSize.SMALL: {"min_gpu_memory": 4.0, "recommended_gpu_memory": 8.0},
            ModelSize.MEDIUM: {"min_gpu_memory": 8.0, "recommended_gpu_memory": 16.0},
            ModelSize.LARGE: {"min_gpu_memory": 16.0, "recommended_gpu_memory": 32.0},
            ModelSize.XL: {"min_gpu_memory": 32.0, "recommended_gpu_memory": 80.0},
        }
    
    def recommend_instances(self, requirements: WorkloadRequirements, top_k: int = 5) -> List[InstanceRecommendation]:
        """Get top instance recommendations for workload"""
        
        # Filter instances by hard requirements
        candidates = self._filter_candidates(requirements)
        
        if not candidates:
            return []
        
        # Score each candidate
        recommendations = []
        for instance in candidates:
            score = self._calculate_workload_score(instance, requirements)
            rec = self._create_recommendation(instance, requirements, score)
            recommendations.append(rec)
        
        # Sort by score and return top K
        recommendations.sort(key=lambda x: x.score, reverse=True)
        return recommendations[:top_k]
    
    def _filter_candidates(self, req: WorkloadRequirements) -> List[CloudGPUInstance]:
        """Filter instances by hard requirements"""
        candidates = []
        
        for instance in self.db.instances:
            # Budget check
            monthly_cost = instance.cost_per_month
            if req.spot_instances_ok and instance.spot_cost_per_hour:
                monthly_cost = min(monthly_cost, instance.spot_cost_per_hour * 730)
            
            if monthly_cost > req.budget_monthly:
                continue
            
            # GPU memory check
            if instance.gpu_memory_total_gb < req.min_gpu_memory_gb:
                continue
            
            # Region check
            if req.region_preference and req.region_preference not in instance.regions:
                continue
            
            candidates.append(instance)
        
        return candidates
    
    def _calculate_workload_score(self, instance: CloudGPUInstance, req: WorkloadRequirements) -> float:
        """Calculate how well instance fits workload (0-100)"""
        score = 0.0
        
        # Base performance score (40 points)
        perf_score = self._calculate_performance_score(instance, req)
        score += perf_score * 0.4
        
        # Cost efficiency score (30 points)
        cost_score = self._calculate_cost_efficiency_score(instance, req)
        score += cost_score * 0.3
        
        # Workload fit score (20 points)
        fit_score = self._calculate_workload_fit_score(instance, req)
        score += fit_score * 0.2
        
        # Availability/reliability score (10 points)
        avail_score = self._calculate_availability_score(instance, req)
        score += avail_score * 0.1
        
        return min(100.0, score)
    
    def _calculate_performance_score(self, instance: CloudGPUInstance, req: WorkloadRequirements) -> float:
        """Score based on raw performance"""
        score = 0.0
        
        # GPU memory score (most important)
        memory_req = self.workload_requirements[req.model_size]["recommended_gpu_memory"]
        memory_ratio = instance.gpu_memory_total_gb / memory_req
        
        if memory_ratio >= 1.0:
            score += 40.0  # Meets requirements
            if memory_ratio <= 2.0:
                score += 10.0  # Not overpowered
        else:
            score += 20.0 * memory_ratio  # Partial credit
        
        # Compute performance
        if req.workload_type in [WorkloadType.TRAINING, WorkloadType.RESEARCH]:
            # Training workloads prefer FP16 performance
            flops_score = min(50.0, instance.fp16_tflops / 100.0 * 50.0)
        else:
            # Inference workloads balance FP16 and FP32
            flops_score = min(50.0, (instance.fp16_tflops + instance.fp32_tflops) / 150.0 * 50.0)
        
        score += flops_score
        
        return min(100.0, score)
    
    def _calculate_cost_efficiency_score(self, instance: CloudGPUInstance, req: WorkloadRequirements) -> float:
        """Score based on cost efficiency"""
        # Cost per TFLOP (lower is better)
        cost_per_hour = instance.cost_per_hour
        if req.spot_instances_ok and instance.spot_cost_per_hour:
            cost_per_hour = instance.spot_cost_per_hour
        
        cost_per_tflop = cost_per_hour / max(instance.fp16_tflops, 1.0)
        
        # Normalize to 0-100 scale (lower cost_per_tflop = higher score)
        # Assuming $0.01-0.10 per TFLOP range
        normalized_score = max(0, min(100, (0.10 - cost_per_tflop) / 0.09 * 100))
        
        return normalized_score
    
    def _calculate_workload_fit_score(self, instance: CloudGPUInstance, req: WorkloadRequirements) -> float:
        """Score based on workload-specific fit"""
        score = 50.0  # Base score
        
        # Training workloads prefer more GPUs for distributed training
        if req.workload_type in [WorkloadType.TRAINING, WorkloadType.DISTRIBUTED]:
            if instance.gpu_count >= 4:
                score += 30.0
            elif instance.gpu_count >= 2:
                score += 15.0
        
        # Inference workloads prefer single GPU efficiency
        elif req.workload_type == WorkloadType.INFERENCE:
            if instance.gpu_count == 1:
                score += 20.0
            score += min(30.0, instance.fp32_tflops / 50.0 * 30.0)  # Good FP32 performance
        
        # Development workloads prefer cost-effective options
        elif req.workload_type == WorkloadType.DEVELOPMENT:
            if instance.cost_per_hour < 1.0:
                score += 30.0
            elif instance.cost_per_hour < 2.0:
                score += 15.0
        
        return min(100.0, score)
    
    def _calculate_availability_score(self, instance: CloudGPUInstance, req: WorkloadRequirements) -> float:
        """Score based on availability and reliability"""
        score = 50.0  # Base score
        
        # More regions = better availability
        score += min(30.0, len(instance.regions) * 5.0)
        
        # Popular instance types are more reliable
        if instance.gpu_type in ["Tesla V100", "A100", "Tesla T4"]:
            score += 20.0
        
        return min(100.0, score)
    
    def _create_recommendation(self, instance: CloudGPUInstance, req: WorkloadRequirements, score: float) -> InstanceRecommendation:
        """Create detailed recommendation"""
        
        # Calculate costs
        monthly_cost = instance.cost_per_month
        if req.spot_instances_ok and instance.spot_cost_per_hour:
            monthly_cost = min(monthly_cost, instance.spot_cost_per_hour * 730)
        
        # Cost efficiency (TFLOPS per dollar per month)
        cost_efficiency = instance.fp16_tflops / monthly_cost
        
        # Generate reasons
        reasons = []
        warnings = []
        
        # Memory fit
        memory_req = self.workload_requirements[req.model_size]["recommended_gpu_memory"]
        if instance.gpu_memory_total_gb >= memory_req:
            reasons.append(f"Sufficient GPU memory: {instance.gpu_memory_total_gb}GB >= {memory_req}GB required")
        else:
            warnings.append(f"Limited GPU memory: {instance.gpu_memory_total_gb}GB < {memory_req}GB recommended")
        
        # Cost analysis
        if monthly_cost <= req.budget_monthly * 0.7:
            reasons.append(f"Well within budget: ${monthly_cost:.0f}/month vs ${req.budget_monthly:.0f} budget")
        elif monthly_cost <= req.budget_monthly:
            reasons.append(f"Fits budget: ${monthly_cost:.0f}/month vs ${req.budget_monthly:.0f} budget")
        
        # Performance
        if instance.gpu_count > 1:
            reasons.append(f"Multi-GPU setup: {instance.gpu_count}x {instance.gpu_type} for distributed training")
        
        if req.spot_instances_ok and instance.spot_cost_per_hour:
            savings = (1 - instance.spot_cost_per_hour / instance.cost_per_hour) * 100
            reasons.append(f"Spot instance savings: {savings:.0f}% cost reduction")
        
        # Estimated performance
        estimated_perf = {
            "training_throughput_samples_sec": instance.fp16_tflops * 10,  # Rough estimate
            "inference_latency_ms": 1000 / (instance.fp32_tflops / 10),  # Rough estimate
            "memory_utilization_pct": min(100, memory_req / instance.gpu_memory_total_gb * 80)
        }
        
        return InstanceRecommendation(
            instance=instance,
            score=score,
            cost_monthly=monthly_cost,
            cost_efficiency=cost_efficiency,
            suitability_reasons=reasons,
            warnings=warnings,
            estimated_performance=estimated_perf
        )

# High-level API functions
def recommend_cloud_instances(
    workload_type: str,
    model_size: str,
    budget_monthly: float,
    framework: str = "pytorch",
    spot_ok: bool = False,
    region: Optional[str] = None,
    top_k: int = 5
) -> List[InstanceRecommendation]:
    """High-level function to get cloud instance recommendations"""
    
    engine = CloudRecommendationEngine()
    
    requirements = WorkloadRequirements(
        workload_type=WorkloadType(workload_type),
        model_size=ModelSize(model_size),
        framework=framework,
        budget_monthly=budget_monthly,
        spot_instances_ok=spot_ok,
        region_preference=region
    )
    
    return engine.recommend_instances(requirements, top_k)

def estimate_cloud_costs(instance_type: str, provider: str, hours_per_month: int = 730) -> Dict:
    """Estimate costs for specific instance"""
    db = CloudInstanceDatabase()
    
    for instance in db.instances:
        if instance.instance_type == instance_type and instance.provider == provider:
            on_demand_cost = instance.cost_per_hour * hours_per_month
            spot_cost = None
            if instance.spot_cost_per_hour:
                spot_cost = instance.spot_cost_per_hour * hours_per_month
            
            return {
                "instance_type": instance_type,
                "provider": provider,
                "on_demand_monthly": on_demand_cost,
                "spot_monthly": spot_cost,
                "savings_pct": ((on_demand_cost - spot_cost) / on_demand_cost * 100) if spot_cost else 0
            }
    
    return {"error": f"Instance {provider}:{instance_type} not found"}

if __name__ == "__main__":
    # Test recommendations
    recs = recommend_cloud_instances(
        workload_type="training",
        model_size="medium", 
        budget_monthly=500,
        spot_ok=True
    )
    
    print(f"🎯 Top recommendations for medium model training (budget: $500/month):")
    for i, rec in enumerate(recs, 1):
        print(f"\n{i}. {rec.instance.provider.upper()} {rec.instance.instance_type}")
        print(f"   Score: {rec.score:.1f}/100")
        print(f"   Cost: ${rec.cost_monthly:.0f}/month")
        print(f"   GPU: {rec.instance.gpu_count}x {rec.instance.gpu_type}")
        print(f"   Reasons: {', '.join(rec.suitability_reasons[:2])}") 