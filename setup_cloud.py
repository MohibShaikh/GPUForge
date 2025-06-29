#!/usr/bin/env python3
"""
GPUForge Cloud Setup - Configure real cloud deployment
"""

import os
import sys
import subprocess
import tempfile
from pathlib import Path

def check_python_version():
    """Check Python version compatibility"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required for cloud deployment")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_cloud_dependencies():
    """Install cloud deployment dependencies"""
    print("📦 Installing cloud deployment dependencies...")
    
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r", "requirements-cloud.txt"
        ], check=True, capture_output=True)
        print("✅ Cloud dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False
    except FileNotFoundError:
        print("❌ requirements-cloud.txt not found")
        return False

def check_terraform():
    """Check if Terraform is installed"""
    try:
        result = subprocess.run(
            ["terraform", "--version"], 
            capture_output=True, 
            text=True, 
            check=True
        )
        version = result.stdout.split('\n')[0]
        print(f"✅ {version}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Terraform not found")
        print("   Download from: https://www.terraform.io/downloads")
        print("   Add to PATH after installation")
        return False

def setup_aws_credentials():
    """Help setup AWS credentials"""
    print("\n🔑 AWS Credentials Setup")
    
    # Check existing credentials
    aws_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret = os.environ.get('AWS_SECRET_ACCESS_KEY')
    
    if aws_key and aws_secret:
        print("✅ AWS credentials found in environment variables")
        return True
    
    # Check AWS CLI config
    aws_config = Path.home() / ".aws" / "credentials"
    if aws_config.exists():
        print("✅ AWS credentials found in ~/.aws/credentials")
        return True
    
    print("⚠️ No AWS credentials found")
    print("\nSetup options:")
    print("1. Environment variables:")
    print("   export AWS_ACCESS_KEY_ID=your_access_key")
    print("   export AWS_SECRET_ACCESS_KEY=your_secret_key")
    print("\n2. AWS CLI: aws configure")
    print("\n3. IAM roles (for EC2 instances)")
    
    return False

def create_sample_deployment():
    """Create sample deployment configuration"""
    config_dir = Path.home() / ".gpuforge"
    config_dir.mkdir(exist_ok=True)
    
    sample_config = config_dir / "sample_deployment.py"
    
    sample_content = '''"""
Sample GPUForge Real Cloud Deployment
Run with: python sample_deployment.py
"""

from gpuforge.cloud_deployment import RealDeploymentConfig, RealCloudDeployer

def create_sample_deployment():
    """Create a sample ML training deployment"""
    
    config = RealDeploymentConfig(
        name="ml-training-sample",
        provider="aws",
        region="us-east-1",
        instance_type="g4dn.xlarge",  # ~$0.526/hour
        
        # Cost controls
        use_spot_instances=True,      # 70% cost savings
        max_hourly_cost=2.0,          # Budget protection
        auto_shutdown_hours=6,        # Auto-shutdown after 6 hours
        
        # ML configuration
        environment_profile="research",
        storage_size_gb=100,
        gpu_count=1
    )
    
    deployer = RealCloudDeployer()
    
    print("🚀 Creating real AWS deployment...")
    print(f"   Instance: {config.instance_type}")
    print(f"   Region: {config.region}")
    print(f"   Cost limit: ${config.max_hourly_cost}/hour")
    print(f"   Auto-shutdown: {config.auto_shutdown_hours} hours")
    
    deployment_id = deployer.deploy_aws_instance(config)
    
    if deployment_id:
        print(f"✅ Deployment created: {deployment_id}")
        
        # Monitor deployment
        print("\\n📊 Monitoring deployment...")
        status = deployer.get_deployment_status(deployment_id)
        if status:
            print(f"   Status: {status.get('real_status', 'unknown')}")
            if 'outputs' in status:
                print(f"   Public IP: {status['outputs'].get('public_ip', 'N/A')}")
        
        # Show cleanup command
        print(f"\\n🛑 To terminate when done:")
        print(f"   python -c \\"from gpuforge.cloud_deployment import RealCloudDeployer; RealCloudDeployer().terminate_deployment('{deployment_id}')\\"")
    
    return deployment_id

if __name__ == "__main__":
    create_sample_deployment()
'''
    
    with open(sample_config, 'w') as f:
        f.write(sample_content)
    
    print(f"✅ Sample deployment created: {sample_config}")
    print(f"   Run with: python {sample_config}")

def main():
    """Main setup workflow"""
    print("🔥 GPUForge Cloud Deployment Setup")
    print("=" * 50)
    
    # Check requirements
    if not check_python_version():
        return False
    
    print("\\n📦 Checking dependencies...")
    deps_ok = install_cloud_dependencies()
    terraform_ok = check_terraform()
    
    print("\\n🔑 Checking cloud credentials...")
    aws_ok = setup_aws_credentials()
    
    print("\\n📋 Creating sample configuration...")
    create_sample_deployment()
    
    print("\\n" + "=" * 50)
    print("🎯 Setup Summary:")
    print(f"   Dependencies: {'✅' if deps_ok else '❌'}")
    print(f"   Terraform: {'✅' if terraform_ok else '❌'}")
    print(f"   AWS Credentials: {'✅' if aws_ok else '⚠️'}")
    
    if deps_ok and terraform_ok and aws_ok:
        print("\\n🚀 Ready for real cloud deployment!")
        print("\\nNext steps:")
        print("1. Test with: python ~/.gpuforge/sample_deployment.py")
        print("2. Or use CLI: python -m gpuforge --deploy-cloud-real --deployment-name test")
    else:
        print("\\n⚠️ Complete setup requirements above before deploying")
    
    return deps_ok and terraform_ok and aws_ok

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 