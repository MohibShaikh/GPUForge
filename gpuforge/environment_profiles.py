"""
Smart Environment Profiles
Use-case specific optimizations for different ML workflows
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
import json

@dataclass
class EnvironmentProfile:
    name: str
    description: str
    packages: Dict[str, List[str]]  # conda vs pip packages
    memory_efficient: bool = False
    pinned_versions: bool = False
    specialized_for: Optional[str] = None
    min_memory_gb: float = 0
    recommended_memory_gb: float = 8
    target_use_case: str = "general"

class ProfileManager:
    def __init__(self):
        self.profiles = self._load_builtin_profiles()
    
    def _load_builtin_profiles(self) -> Dict[str, EnvironmentProfile]:
        """Load built-in environment profiles"""
        
        profiles = {
            'learning': EnvironmentProfile(
                name='learning',
                description='Perfect for ML learning, tutorials, and educational use',
                packages={
                    'conda': [
                        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn',
                        'jupyter', 'jupyterlab', 'ipykernel', 'notebook'
                    ],
                    'pip': [
                        'plotly', 'ipywidgets'
                    ]
                },
                memory_efficient=True,
                min_memory_gb=2,
                recommended_memory_gb=4,
                target_use_case='education'
            ),
            
            'research': EnvironmentProfile(
                name='research',
                description='Full-featured environment for research and experimentation',
                packages={
                    'conda': [
                        'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn', 'plotly',
                        'scikit-learn', 'jupyter', 'jupyterlab', 'ipykernel', 'notebook',
                        'ipywidgets', 'tqdm', 'pillow'
                    ],
                    'pip': [
                        'transformers', 'datasets', 'accelerate', 'wandb', 'tensorboard',
                        'mlflow', 'optuna', 'ray[tune]', 'hydra-core'
                    ]
                },
                memory_efficient=False,
                min_memory_gb=6,
                recommended_memory_gb=12,
                target_use_case='research'
            ),
            
            'computer_vision': EnvironmentProfile(
                name='computer_vision',
                description='Optimized for computer vision and image processing tasks',
                packages={
                    'conda': [
                        'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn',
                        'pillow', 'opencv', 'scikit-image', 'jupyter', 'tqdm'
                    ],
                    'pip': [
                        'albumentations', 'timm', 'detectron2', 'ultralytics',
                        'torchvision', 'transformers', 'datasets', 'wandb'
                    ]
                },
                specialized_for='cv',
                min_memory_gb=6,
                recommended_memory_gb=16,
                target_use_case='computer_vision'
            ),
            
            'nlp': EnvironmentProfile(
                name='nlp',
                description='Natural language processing and text analysis',
                packages={
                    'conda': [
                        'numpy', 'pandas', 'matplotlib', 'seaborn',
                        'jupyter', 'tqdm', 'nltk', 'spacy'
                    ],
                    'pip': [
                        'transformers', 'tokenizers', 'datasets', 'accelerate',
                        'sentence-transformers', 'sacrebleu', 'rouge-score',
                        'wandb', 'tensorboard'
                    ]
                },
                specialized_for='nlp',
                min_memory_gb=4,
                recommended_memory_gb=8,
                target_use_case='nlp'
            ),
            
            'production': EnvironmentProfile(
                name='production',
                description='Stable, pinned versions for production deployment',
                packages={
                    'conda': [
                        'numpy=1.24.*', 'pandas=2.0.*', 'scikit-learn=1.3.*'
                    ],
                    'pip': [
                        'fastapi==0.104.*', 'uvicorn==0.24.*', 'pydantic==2.5.*',
                        'prometheus-client==0.19.*', 'gunicorn==21.2.*'
                    ]
                },
                pinned_versions=True,
                memory_efficient=True,
                min_memory_gb=2,
                recommended_memory_gb=4,
                target_use_case='production'
            ),
            
            'deep_learning': EnvironmentProfile(
                name='deep_learning',
                description='Heavy-duty deep learning with large models',
                packages={
                    'conda': [
                        'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn',
                        'jupyter', 'tqdm', 'pillow'
                    ],
                    'pip': [
                        'transformers', 'datasets', 'accelerate', 'deepspeed',
                        'bitsandbytes', 'peft', 'trl', 'wandb', 'tensorboard',
                        'flash-attn', 'xformers'
                    ]
                },
                specialized_for='deep_learning',
                min_memory_gb=12,
                recommended_memory_gb=24,
                target_use_case='deep_learning'
            ),
            
            'lightweight': EnvironmentProfile(
                name='lightweight',
                description='Minimal setup for resource-constrained environments',
                packages={
                    'conda': [
                        'numpy', 'pandas', 'matplotlib', 'scikit-learn'
                    ],
                    'pip': [
                        'jupyter'
                    ]
                },
                memory_efficient=True,
                min_memory_gb=1,
                recommended_memory_gb=2,
                target_use_case='minimal'
            ),
            
            'reinforcement_learning': EnvironmentProfile(
                name='reinforcement_learning',
                description='Reinforcement learning and game AI development',
                packages={
                    'conda': [
                        'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn',
                        'jupyter', 'tqdm'
                    ],
                    'pip': [
                        'gymnasium', 'stable-baselines3', 'ray[rllib]',
                        'tensorboard', 'wandb', 'ale-py', 'shimmy[atari]'
                    ]
                },
                specialized_for='rl',
                min_memory_gb=4,
                recommended_memory_gb=8,
                target_use_case='reinforcement_learning'
            )
        }
        
        return profiles
    
    def recommend_profile(self, gpu_info: Dict, use_case: Optional[str] = None) -> str:
        """Recommend profile based on GPU specs and use case"""
        
        memory_gb = gpu_info.get('memory_mb', 0) / 1024
        vendor = gpu_info.get('vendor', 'Unknown')
        
        # If use case is specified, prefer that
        if use_case and use_case in self.profiles:
            profile = self.profiles[use_case]
            if memory_gb >= profile.min_memory_gb:
                return use_case
        
        # Memory-based recommendations
        if memory_gb < 4:
            return 'lightweight'
        elif memory_gb < 6:
            return 'learning'
        elif memory_gb < 12:
            return 'research'
        else:
            return 'deep_learning'
    
    def get_profile(self, profile_name: str) -> Optional[EnvironmentProfile]:
        """Get profile by name"""
        return self.profiles.get(profile_name)
    
    def list_profiles(self) -> List[str]:
        """List all available profile names"""
        return list(self.profiles.keys())
    
    def get_profile_info(self, profile_name: str) -> Dict:
        """Get detailed profile information"""
        profile = self.profiles.get(profile_name)
        if not profile:
            return {}
        
        return {
            'name': profile.name,
            'description': profile.description,
            'packages': profile.packages,
            'memory_requirements': {
                'minimum_gb': profile.min_memory_gb,
                'recommended_gb': profile.recommended_memory_gb
            },
            'optimizations': {
                'memory_efficient': profile.memory_efficient,
                'pinned_versions': profile.pinned_versions,
                'specialized_for': profile.specialized_for
            },
            'target_use_case': profile.target_use_case,
            'total_packages': len(profile.packages.get('conda', [])) + len(profile.packages.get('pip', []))
        }
    
    def validate_profile_for_gpu(self, profile_name: str, gpu_info: Dict) -> Dict:
        """Validate if profile is suitable for given GPU"""
        profile = self.profiles.get(profile_name)
        if not profile:
            return {'valid': False, 'reason': 'Profile not found'}
        
        memory_gb = gpu_info.get('memory_mb', 0) / 1024
        vendor = gpu_info.get('vendor', 'Unknown')
        
        issues = []
        warnings = []
        
        # Memory validation
        if memory_gb < profile.min_memory_gb:
            issues.append(f"Insufficient GPU memory: {memory_gb:.1f} GB < {profile.min_memory_gb} GB required")
        elif memory_gb < profile.recommended_memory_gb:
            warnings.append(f"GPU memory below recommended: {memory_gb:.1f} GB < {profile.recommended_memory_gb} GB")
        
        # Vendor-specific warnings
        if vendor != 'NVIDIA':
            if profile.specialized_for in ['deep_learning', 'computer_vision']:
                warnings.append(f"Profile optimized for NVIDIA GPUs, {vendor} support may be limited")
        
        # Framework compatibility for specialized profiles
        if profile.specialized_for == 'deep_learning' and memory_gb < 8:
            warnings.append("Deep learning profile works best with 8+ GB GPU memory")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'warnings': warnings,
            'suitability_score': self._calculate_suitability_score(profile, gpu_info)
        }
    
    def _calculate_suitability_score(self, profile: EnvironmentProfile, gpu_info: Dict) -> float:
        """Calculate how suitable a profile is for given GPU (0-100)"""
        score = 50  # Base score
        
        memory_gb = gpu_info.get('memory_mb', 0) / 1024
        vendor = gpu_info.get('vendor', 'Unknown')
        
        # Memory scoring
        if memory_gb >= profile.recommended_memory_gb:
            score += 30
        elif memory_gb >= profile.min_memory_gb:
            score += 20
        else:
            score -= 30
        
        # Vendor scoring
        if vendor == 'NVIDIA':
            score += 20
        elif vendor == 'AMD':
            score += 10
        
        # Special adjustments
        if profile.memory_efficient and memory_gb < 6:
            score += 10
        
        if profile.specialized_for == 'deep_learning' and memory_gb >= 12:
            score += 15
        
        return max(0, min(100, score))
    
    def generate_custom_profile(self, requirements: Dict) -> EnvironmentProfile:
        """Generate custom profile based on requirements"""
        
        name = requirements.get('name', 'custom')
        description = requirements.get('description', 'Custom environment profile')
        
        # Base packages
        conda_packages = ['numpy', 'pandas', 'matplotlib', 'jupyter']
        pip_packages = []
        
        # Add packages based on requirements
        frameworks = requirements.get('frameworks', [])
        if 'pytorch' in frameworks:
            conda_packages.extend(['pytorch', 'torchvision', 'torchaudio'])
        if 'tensorflow' in frameworks:
            pip_packages.append('tensorflow')
        
        use_cases = requirements.get('use_cases', [])
        if 'computer_vision' in use_cases:
            conda_packages.extend(['pillow', 'opencv'])
            pip_packages.extend(['albumentations', 'timm'])
        
        if 'nlp' in use_cases:
            pip_packages.extend(['transformers', 'datasets'])
        
        if 'research' in use_cases:
            pip_packages.extend(['wandb', 'tensorboard', 'optuna'])
        
        # Additional packages
        additional_packages = requirements.get('additional_packages', {})
        conda_packages.extend(additional_packages.get('conda', []))
        pip_packages.extend(additional_packages.get('pip', []))
        
        return EnvironmentProfile(
            name=name,
            description=description,
            packages={
                'conda': list(set(conda_packages)),  # Remove duplicates
                'pip': list(set(pip_packages))
            },
            memory_efficient=requirements.get('memory_efficient', False),
            pinned_versions=requirements.get('pinned_versions', False),
            min_memory_gb=requirements.get('min_memory_gb', 2),
            recommended_memory_gb=requirements.get('recommended_memory_gb', 8),
            target_use_case=requirements.get('target_use_case', 'custom')
        )
    
    def optimize_profile_for_gpu(self, profile_name: str, gpu_info: Dict) -> EnvironmentProfile:
        """Optimize existing profile for specific GPU"""
        
        base_profile = self.profiles.get(profile_name)
        if not base_profile:
            raise ValueError(f"Profile {profile_name} not found")
        
        memory_gb = gpu_info.get('memory_mb', 0) / 1024
        vendor = gpu_info.get('vendor', 'Unknown')
        
        # Create optimized copy
        optimized_packages = {
            'conda': base_profile.packages.get('conda', []).copy(),
            'pip': base_profile.packages.get('pip', []).copy()
        }
        
        # Memory optimizations
        if memory_gb < 6:
            # Remove memory-intensive packages for low-memory GPUs
            memory_intensive = ['deepspeed', 'flash-attn', 'xformers']
            optimized_packages['pip'] = [
                pkg for pkg in optimized_packages['pip'] 
                if not any(intensive in pkg for intensive in memory_intensive)
            ]
            
            # Add memory-efficient alternatives
            if 'transformers' in optimized_packages['pip']:
                optimized_packages['pip'].append('bitsandbytes')
        
        # Vendor-specific optimizations
        if vendor == 'AMD':
            # Replace CUDA-specific packages with ROCm alternatives
            if 'pytorch-cuda' in optimized_packages['conda']:
                optimized_packages['conda'].remove('pytorch-cuda')
                optimized_packages['conda'].append('pytorch-rocm')
        
        elif vendor == 'Intel':
            # Add Intel optimizations
            optimized_packages['pip'].append('intel-extension-for-pytorch')
        
        # Create optimized profile
        return EnvironmentProfile(
            name=f"{base_profile.name}_optimized",
            description=f"{base_profile.description} (optimized for {vendor})",
            packages=optimized_packages,
            memory_efficient=True if memory_gb < 6 else base_profile.memory_efficient,
            pinned_versions=base_profile.pinned_versions,
            specialized_for=base_profile.specialized_for,
            min_memory_gb=base_profile.min_memory_gb,
            recommended_memory_gb=base_profile.recommended_memory_gb,
            target_use_case=base_profile.target_use_case
        ) 