"""
Smart Error Recovery and Configuration Management
Automatic problem detection and intelligent recovery suggestions
"""

import subprocess
import platform
import time
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

class ErrorSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class RecoveryAction:
    description: str
    command: Optional[str] = None
    manual_steps: Optional[List[str]] = None
    estimated_time: str = "< 1 minute"
    requires_admin: bool = False
    platform_specific: Optional[str] = None

class SmartErrorRecovery:
    def __init__(self):
        self.recovery_history = []
        self.config_cache = {}
        self.known_issues = self._load_known_issues()
    
    def _load_known_issues(self) -> Dict:
        """Load database of known issues and solutions"""
        return {
            'conda_not_found': {
                'patterns': ['conda: command not found', 'conda is not recognized'],
                'severity': ErrorSeverity.CRITICAL,
                'recovery': RecoveryAction(
                    description="Install Miniconda or Anaconda",
                    manual_steps=[
                        "Download Miniconda from https://docs.conda.io/en/latest/miniconda.html",
                        "Run the installer and follow instructions",
                        "Restart your terminal/command prompt",
                        "Verify with: conda --version"
                    ],
                    estimated_time="5-10 minutes"
                )
            },
            
            'nvidia_driver_missing': {
                'patterns': ['nvidia-smi: command not found', 'NVIDIA-SMI has failed'],
                'severity': ErrorSeverity.ERROR,
                'recovery': RecoveryAction(
                    description="Install NVIDIA GPU drivers",
                    manual_steps=[
                        "Visit https://www.nvidia.com/drivers",
                        "Download latest driver for your GPU",
                        "Run installer as administrator",
                        "Restart computer after installation",
                        "Verify with: nvidia-smi"
                    ],
                    estimated_time="10-15 minutes",
                    requires_admin=True
                )
            },
            
            'cuda_version_mismatch': {
                'patterns': ['CUDA version mismatch', 'incompatible CUDA version'],
                'severity': ErrorSeverity.ERROR,
                'recovery': RecoveryAction(
                    description="Fix CUDA version compatibility",
                    manual_steps=[
                        "Check your NVIDIA driver version with: nvidia-smi",
                        "Install compatible CUDA version for your driver",
                        "Update PyTorch/TensorFlow to match CUDA version",
                        "Recreate conda environment with correct versions"
                    ],
                    estimated_time="15-30 minutes"
                )
            },
            
            'out_of_memory': {
                'patterns': ['CUDA out of memory', 'RuntimeError: out of memory'],
                'severity': ErrorSeverity.WARNING,
                'recovery': RecoveryAction(
                    description="Optimize memory usage",
                    manual_steps=[
                        "Reduce batch size in your training script",
                        "Enable gradient checkpointing",
                        "Use mixed precision training (fp16)",
                        "Clear GPU cache: torch.cuda.empty_cache()",
                        "Consider using CPU for inference"
                    ],
                    estimated_time="5 minutes"
                )
            },
            
            'environment_corruption': {
                'patterns': ['environment is corrupt', 'package conflicts', 'ImportError'],
                'severity': ErrorSeverity.WARNING,
                'recovery': RecoveryAction(
                    description="Recreate conda environment",
                    command="conda env remove -n {env_name} && conda env create -f {env_file}",
                    manual_steps=[
                        "Remove corrupted environment",
                        "Recreate from environment file",
                        "Test import of key packages"
                    ],
                    estimated_time="10-20 minutes"
                )
            },
            
            'package_conflict': {
                'patterns': ['PackageConflictError', 'UnsatisfiableError'],
                'severity': ErrorSeverity.WARNING,
                'recovery': RecoveryAction(
                    description="Resolve package conflicts",
                    manual_steps=[
                        "Check for conflicting package versions",
                        "Use conda-forge channel: conda install -c conda-forge",
                        "Create fresh environment with minimal packages",
                        "Install packages one by one to identify conflicts"
                    ],
                    estimated_time="10-15 minutes"
                )
            },
            
            'slow_download': {
                'patterns': ['download timeout', 'connection timeout', 'slow download'],
                'severity': ErrorSeverity.INFO,
                'recovery': RecoveryAction(
                    description="Optimize conda downloads",
                    manual_steps=[
                        "Add faster conda channels",
                        "Set conda to use libmamba solver: conda install -n base conda-libmamba-solver",
                        "Configure conda to use multiple channels",
                        "Use mamba instead of conda for faster installs"
                    ],
                    estimated_time="5 minutes"
                )
            }
        }
    
    def diagnose_system(self) -> Dict:
        """Comprehensive system diagnosis"""
        print("🔍 Running system diagnosis...")
        
        diagnosis = {
            'timestamp': time.time(),
            'platform': platform.platform(),
            'checks': {}
        }
        
        # Check conda
        diagnosis['checks']['conda'] = self._check_conda()
        
        # Check NVIDIA drivers
        diagnosis['checks']['nvidia'] = self._check_nvidia_drivers()
        
        # Check Python environment
        diagnosis['checks']['python'] = self._check_python_environment()
        
        # Check disk space
        diagnosis['checks']['disk_space'] = self._check_disk_space()
        
        # Check network connectivity
        diagnosis['checks']['network'] = self._check_network()
        
        # Check GPU accessibility
        diagnosis['checks']['gpu_access'] = self._check_gpu_accessibility()
        
        return diagnosis
    
    def _check_conda(self) -> Dict:
        """Check conda installation and configuration"""
        try:
            result = subprocess.run(['conda', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                version = result.stdout.strip()
                
                # Check conda configuration
                config_result = subprocess.run(['conda', 'config', '--show'], 
                                             capture_output=True, text=True, timeout=10)
                
                channels = []
                if 'channels:' in config_result.stdout:
                    lines = config_result.stdout.split('\n')
                    in_channels = False
                    for line in lines:
                        if 'channels:' in line:
                            in_channels = True
                        elif in_channels and line.startswith('  - '):
                            channels.append(line.strip('  - '))
                        elif in_channels and not line.startswith('  '):
                            break
                
                return {
                    'status': 'ok',
                    'version': version,
                    'channels': channels,
                    'message': f"Conda {version} installed successfully"
                }
            else:
                return {
                    'status': 'error',
                    'message': 'Conda command failed',
                    'error': result.stderr
                }
                
        except FileNotFoundError:
            return {
                'status': 'missing',
                'message': 'Conda not found - please install Miniconda or Anaconda',
                'recovery': 'conda_not_found'
            }
        except subprocess.TimeoutExpired:
            return {
                'status': 'timeout',
                'message': 'Conda command timed out',
                'recovery': 'environment_corruption'
            }
    
    def _check_nvidia_drivers(self) -> Dict:
        """Check NVIDIA driver installation"""
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # Parse nvidia-smi output for driver version
                lines = result.stdout.split('\n')
                driver_line = next((line for line in lines if 'Driver Version:' in line), None)
                
                if driver_line:
                    driver_version = driver_line.split('Driver Version:')[1].split()[0]
                    return {
                        'status': 'ok',
                        'driver_version': driver_version,
                        'message': f"NVIDIA driver {driver_version} installed"
                    }
                else:
                    return {
                        'status': 'partial',
                        'message': 'NVIDIA driver detected but version unclear'
                    }
            else:
                return {
                    'status': 'error',
                    'message': 'nvidia-smi failed',
                    'error': result.stderr,
                    'recovery': 'nvidia_driver_missing'
                }
                
        except FileNotFoundError:
            return {
                'status': 'missing',
                'message': 'NVIDIA drivers not installed',
                'recovery': 'nvidia_driver_missing'
            }
    
    def _check_python_environment(self) -> Dict:
        """Check Python environment health"""
        try:
            import sys
            
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            
            # Check if we're in a conda environment
            in_conda = 'CONDA_DEFAULT_ENV' in os.environ if 'os' in globals() else False
            
            # Check key ML packages
            key_packages = ['numpy', 'pandas', 'matplotlib']
            available_packages = []
            missing_packages = []
            
            for pkg in key_packages:
                try:
                    __import__(pkg)
                    available_packages.append(pkg)
                except ImportError:
                    missing_packages.append(pkg)
            
            return {
                'status': 'ok' if not missing_packages else 'partial',
                'python_version': python_version,
                'in_conda_env': in_conda,
                'available_packages': available_packages,
                'missing_packages': missing_packages,
                'message': f"Python {python_version} environment {'healthy' if not missing_packages else 'needs packages'}"
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Python environment check failed: {str(e)}"
            }
    
    def _check_disk_space(self) -> Dict:
        """Check available disk space"""
        try:
            import shutil
            
            # Check space in current directory
            current_dir = Path.cwd()
            total, used, free = shutil.disk_usage(current_dir)
            
            free_gb = free / (1024**3)
            total_gb = total / (1024**3)
            
            status = 'ok' if free_gb > 5 else 'warning' if free_gb > 2 else 'critical'
            
            return {
                'status': status,
                'free_gb': round(free_gb, 1),
                'total_gb': round(total_gb, 1),
                'message': f"{free_gb:.1f} GB free space available"
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Disk space check failed: {str(e)}"
            }
    
    def _check_network(self) -> Dict:
        """Check network connectivity to package repositories"""
        try:
            import urllib.request
            
            test_urls = [
                ('conda-forge', 'https://conda-forge.org'),
                ('PyPI', 'https://pypi.org'),
                ('NVIDIA', 'https://developer.nvidia.com')
            ]
            
            connectivity = {}
            
            for name, url in test_urls:
                try:
                    response = urllib.request.urlopen(url, timeout=5)
                    connectivity[name] = response.status == 200
                except:
                    connectivity[name] = False
            
            all_connected = all(connectivity.values())
            
            return {
                'status': 'ok' if all_connected else 'warning',
                'connectivity': connectivity,
                'message': 'Network connectivity ' + ('good' if all_connected else 'limited')
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f"Network check failed: {str(e)}"
            }
    
    def _check_gpu_accessibility(self) -> Dict:
        """Check if GPU is accessible to Python"""
        gpu_status = {
            'pytorch': False,
            'tensorflow': False,
            'cuda_available': False
        }
        
        # Check PyTorch GPU access
        try:
            import torch
            gpu_status['pytorch'] = torch.cuda.is_available()
            gpu_status['cuda_available'] = torch.cuda.is_available()
        except ImportError:
            pass
        
        # Check TensorFlow GPU access
        try:
            import tensorflow as tf
            gpu_status['tensorflow'] = len(tf.config.list_physical_devices('GPU')) > 0
        except ImportError:
            pass
        
        any_gpu_access = any(gpu_status.values())
        
        return {
            'status': 'ok' if any_gpu_access else 'warning',
            'gpu_access': gpu_status,
            'message': 'GPU ' + ('accessible' if any_gpu_access else 'not accessible') + ' to ML frameworks'
        }
    
    def analyze_error(self, error_message: str, context: Optional[Dict] = None) -> Dict:
        """Analyze error message and suggest recovery actions"""
        
        error_message_lower = error_message.lower()
        
        # Find matching known issues
        matches = []
        for issue_key, issue_data in self.known_issues.items():
            for pattern in issue_data['patterns']:
                if pattern.lower() in error_message_lower:
                    matches.append({
                        'issue': issue_key,
                        'data': issue_data,
                        'confidence': len(pattern) / len(error_message_lower)  # Simple confidence metric
                    })
        
        # Sort by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        if matches:
            best_match = matches[0]
            return {
                'identified': True,
                'issue': best_match['issue'],
                'severity': best_match['data']['severity'].value,
                'recovery': best_match['data']['recovery'],
                'confidence': best_match['confidence'],
                'alternatives': [m['issue'] for m in matches[1:3]]  # Top 2 alternatives
            }
        else:
            # Generic recovery suggestions
            return {
                'identified': False,
                'issue': 'unknown',
                'severity': ErrorSeverity.WARNING.value,
                'recovery': RecoveryAction(
                    description="General troubleshooting steps",
                    manual_steps=[
                        "Check error message for specific package names",
                        "Search for the error message online",
                        "Try recreating the conda environment",
                        "Check system requirements and compatibility",
                        "Consult documentation for the failing package"
                    ],
                    estimated_time="varies"
                ),
                'confidence': 0.0,
                'suggestion': "Run system diagnosis for more specific help"
            }
    
    def suggest_recovery(self, issue_key: str, context: Optional[Dict] = None) -> List[str]:
        """Generate step-by-step recovery instructions"""
        
        if issue_key not in self.known_issues:
            return ["Unknown issue - please run system diagnosis"]
        
        issue_data = self.known_issues[issue_key]
        recovery = issue_data['recovery']
        
        steps = []
        
        # Add context-aware information
        if context:
            steps.append(f"🔍 Context: {context.get('operation', 'Unknown operation')} failed")
        
        steps.append(f"📋 Issue: {recovery.description}")
        steps.append(f"⏱️  Estimated time: {recovery.estimated_time}")
        
        if recovery.requires_admin:
            steps.append("⚠️  Administrator privileges required")
        
        # Add platform-specific note
        if recovery.platform_specific:
            current_platform = platform.system()
            if recovery.platform_specific.lower() != current_platform.lower():
                steps.append(f"ℹ️  Note: These steps are for {recovery.platform_specific}")
        
        # Add manual steps
        if recovery.manual_steps:
            steps.append("\n📝 Steps to fix:")
            for i, step in enumerate(recovery.manual_steps, 1):
                steps.append(f"   {i}. {step}")
        
        # Add command if available
        if recovery.command:
            steps.append(f"\n💻 Command to run:")
            steps.append(f"   {recovery.command}")
        
        return steps
    
    def auto_fix_attempt(self, issue_key: str, env_name: Optional[str] = None) -> Dict:
        """Attempt automatic fix for known issues"""
        
        if issue_key not in self.known_issues:
            return {'success': False, 'message': 'Unknown issue'}
        
        recovery = self.known_issues[issue_key]['recovery']
        
        if not recovery.command:
            return {
                'success': False, 
                'message': 'No automatic fix available',
                'manual_required': True
            }
        
        try:
            # Substitute variables in command
            command = recovery.command
            if env_name and '{env_name}' in command:
                command = command.replace('{env_name}', env_name)
            if '{env_file}' in command:
                command = command.replace('{env_file}', f"{env_name}.yml")
            
            # Execute command
            result = subprocess.run(command.split(), 
                                  capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'message': 'Automatic fix completed successfully',
                    'output': result.stdout
                }
            else:
                return {
                    'success': False,
                    'message': 'Automatic fix failed',
                    'error': result.stderr,
                    'manual_required': True
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'Auto-fix attempt failed: {str(e)}',
                'manual_required': True
            }
    
    def generate_recovery_report(self, diagnosis: Dict) -> str:
        """Generate comprehensive recovery report"""
        
        report = []
        report.append("🏥 GPU Environment Creator - System Health Report")
        report.append("=" * 50)
        report.append(f"Generated: {time.ctime(diagnosis['timestamp'])}")
        report.append(f"Platform: {diagnosis['platform']}")
        report.append("")
        
        # Analyze each check
        issues_found = []
        warnings_found = []
        
        for check_name, check_result in diagnosis['checks'].items():
            status = check_result.get('status', 'unknown')
            message = check_result.get('message', 'No details')
            
            if status == 'ok':
                report.append(f"✅ {check_name.replace('_', ' ').title()}: {message}")
            elif status == 'warning':
                report.append(f"⚠️  {check_name.replace('_', ' ').title()}: {message}")
                warnings_found.append(check_name)
            elif status in ['error', 'missing', 'critical']:
                report.append(f"❌ {check_name.replace('_', ' ').title()}: {message}")
                issues_found.append(check_name)
                
                # Add recovery suggestion if available
                if 'recovery' in check_result:
                    recovery_key = check_result['recovery']
                    recovery_steps = self.suggest_recovery(recovery_key)
                    report.append("   🔧 Recovery steps:")
                    for step in recovery_steps:
                        report.append(f"      {step}")
            else:
                report.append(f"❓ {check_name.replace('_', ' ').title()}: {message}")
        
        # Summary
        report.append("")
        report.append("📊 Summary:")
        
        if not issues_found and not warnings_found:
            report.append("🎉 System appears healthy! No major issues detected.")
        else:
            if issues_found:
                report.append(f"🚨 Critical issues found: {len(issues_found)}")
                report.append("   Priority: Fix these issues first")
            
            if warnings_found:
                report.append(f"⚠️  Warnings: {len(warnings_found)}")
                report.append("   These may affect performance but aren't blocking")
        
        return "\n".join(report)

class ConfigurationManager:
    """Manage GPU environment creator configuration"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".gpu_env_creator"
        self.config_dir.mkdir(exist_ok=True)
        self.config_file = self.config_dir / "config.json"
        self.load_config()
    
    def load_config(self) -> Dict:
        """Load configuration from file"""
        default_config = {
            'cache_enabled': True,
            'cache_ttl': 3600,
            'auto_update_check': True,
            'preferred_channels': ['conda-forge', 'pytorch', 'nvidia'],
            'default_profile': 'research',
            'error_reporting': True,
            'verbose_output': False,
            'parallel_installs': True,
            'backup_envs': True
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
            except:
                pass
        
        self.config = default_config
        return self.config
    
    def save_config(self):
        """Save configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save config: {e}")
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def set(self, key: str, value):
        """Set configuration value"""
        self.config[key] = value
        self.save_config()
    
    def reset_to_defaults(self):
        """Reset configuration to defaults"""
        self.config_file.unlink(missing_ok=True)
        self.load_config()
        print("Configuration reset to defaults") 