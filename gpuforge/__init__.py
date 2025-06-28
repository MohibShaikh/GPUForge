"""
GPUForge - The Smart GPU Environment Creator
Forge perfect GPU environments in seconds, not hours.
"""

__version__ = "1.0.0"
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

def cli_main():
    """Entry point for the GPUForge CLI"""
    from .cli import cli_main as _cli_main
    _cli_main() 