"""
GPUForge - The Smart GPU Environment Creator
Forge perfect GPU environments in seconds, not hours.
"""

__version__ = "2.0.0"
__author__ = "GPUForge Contributors"
__description__ = "Intelligent GPU environment creator for machine learning"

# Main components - import safely
try:
    from .gpu_detector_optimized import UniversalGPUDetector
except ImportError:
    UniversalGPUDetector = None

try:
    from .environment_profiles import ProfileManager, EnvironmentProfile
except ImportError:
    ProfileManager = None
    EnvironmentProfile = None

try:
    from .error_recovery import SmartErrorRecovery, ConfigurationManager
except ImportError:
    SmartErrorRecovery = None
    ConfigurationManager = None

try:
    from .compatibility_finder import CompatibilityFinder
except ImportError:
    CompatibilityFinder = None

try:
    from .env_generator import EnvironmentGenerator
except ImportError:
    EnvironmentGenerator = None

# Fallback components
try:
    from .gpu_detector import GPUDetector
except ImportError:
    GPUDetector = None

try:
    from .gpu_env_creator import GPUEnvironmentCreator
except ImportError:
    GPUEnvironmentCreator = None

# Cloud Support (Phase 1, 2 & 3)
try:
    from .cloud_support import detect_cloud, get_cloud_environment_info, is_cloud_gpu_instance
    from .cloud_recommendations import (
        recommend_cloud_instances, 
        estimate_cloud_costs,
        CloudRecommendationEngine,
        WorkloadRequirements,
        WorkloadType,
        ModelSize
    )
    from .cloud_advanced import (
        CloudOrchestrator,
        CloudDeploymentConfig,
        ScalingPolicy,
        create_cloud_deployment,
        optimize_deployment_costs
    )
    CLOUD_SUPPORT_AVAILABLE = True
    CLOUD_RECOMMENDATIONS_AVAILABLE = True
    CLOUD_ADVANCED_AVAILABLE = True
except ImportError:
    CLOUD_SUPPORT_AVAILABLE = False
    CLOUD_RECOMMENDATIONS_AVAILABLE = False
    CLOUD_ADVANCED_AVAILABLE = False
    detect_cloud = None
    get_cloud_environment_info = None
    is_cloud_gpu_instance = None
    recommend_cloud_instances = None
    estimate_cloud_costs = None
    CloudOrchestrator = None
    CloudDeploymentConfig = None
    ScalingPolicy = None
    create_cloud_deployment = None
    optimize_deployment_costs = None

# Build __all__ list dynamically based on what's available
__all__ = []
if UniversalGPUDetector:
    __all__.append('UniversalGPUDetector')
if ProfileManager:
    __all__.extend(['ProfileManager', 'EnvironmentProfile'])
if SmartErrorRecovery:
    __all__.extend(['SmartErrorRecovery', 'ConfigurationManager'])
if CompatibilityFinder:
    __all__.append('CompatibilityFinder')
if EnvironmentGenerator:
    __all__.append('EnvironmentGenerator')
if GPUDetector:
    __all__.append('GPUDetector')
if GPUEnvironmentCreator:
    __all__.append('GPUEnvironmentCreator')
if CLOUD_SUPPORT_AVAILABLE:
    __all__.extend(['detect_cloud', 'get_cloud_environment_info', 'is_cloud_gpu_instance', 'CLOUD_SUPPORT_AVAILABLE'])
if CLOUD_RECOMMENDATIONS_AVAILABLE:
    __all__.extend(['recommend_cloud_instances', 'estimate_cloud_costs', 'CloudRecommendationEngine', 'WorkloadRequirements', 'WorkloadType', 'ModelSize', 'CLOUD_RECOMMENDATIONS_AVAILABLE'])
if CLOUD_ADVANCED_AVAILABLE:
    __all__.extend(['CloudOrchestrator', 'CloudDeploymentConfig', 'ScalingPolicy', 'create_cloud_deployment', 'optimize_deployment_costs', 'CLOUD_ADVANCED_AVAILABLE'])

def cli_main():
    """Entry point for the GPUForge CLI"""
    from .cli import cli_main as _cli_main
    _cli_main() 