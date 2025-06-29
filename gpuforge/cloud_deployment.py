"""
GPUForge Real Cloud Deployment - Production Infrastructure Automation
Uses actual Terraform, CloudFormation, and cloud provider APIs for real deployments
"""

import json
import os
import subprocess
import tempfile
import uuid
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class RealDeploymentConfig:
    """Real cloud deployment configuration"""
    name: str
    provider: str  # aws, gcp, azure
    region: str
    instance_type: str
    
    # Authentication
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    gcp_project_id: Optional[str] = None
    gcp_credentials_path: Optional[str] = None
    azure_subscription_id: Optional[str] = None
    
    # Infrastructure settings
    ami_id: Optional[str] = None  # Auto-selected if None
    key_pair_name: str = "gpuforge-key"
    security_group_name: str = "gpuforge-ml-sg"
    
    # ML Configuration
    gpu_count: int = 1
    storage_size_gb: int = 100
    environment_profile: str = "research"
    
    # Cost controls
    use_spot_instances: bool = False
    max_hourly_cost: float = 10.0
    auto_shutdown_hours: int = 24

class RealCloudDeployer:
    """Production cloud deployment using real infrastructure tools"""
    
    def __init__(self):
        self.deployments_dir = Path.home() / ".gpuforge" / "deployments"
        self.deployments_dir.mkdir(parents=True, exist_ok=True)
        self.active_deployments = self._load_deployments()
    
    def _load_deployments(self) -> Dict:
        """Load active deployments from local state"""
        state_file = self.deployments_dir / "active_deployments.json"
        if state_file.exists():
            with open(state_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_deployments(self):
        """Save active deployments to local state"""
        state_file = self.deployments_dir / "active_deployments.json"
        with open(state_file, 'w') as f:
            json.dump(self.active_deployments, f, indent=2, default=str)
    
    def deploy_aws_instance(self, config: RealDeploymentConfig) -> str:
        """Deploy real AWS EC2 instance using Terraform with detailed logging"""
        
        print("🚀 Starting Real AWS Cloud Deployment")
        print("="*60)
        print(f"   📦 Deployment Name: {config.name}")
        print(f"   🌍 AWS Region: {config.region}")
        print(f"   💻 Instance Type: {config.instance_type}")
        print(f"   💰 Max Hourly Cost: ${config.max_hourly_cost}")
        print(f"   📈 Environment Profile: {config.environment_profile}")
        print(f"   💾 Storage Size: {config.storage_size_gb}GB")
        print(f"   ⏰ Auto-shutdown: {config.auto_shutdown_hours} hours")
        
        print("\n🔐 Step 1: Validating AWS credentials...")
        if not self._validate_aws_credentials(config):
            print("❌ AWS credentials validation failed")
            print("   ℹ️ Please configure AWS credentials using one of:")
            print("   - AWS CLI: aws configure")
            print("   - Environment variables: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY")
            print("   - IAM roles (for EC2 instances)")
            print("   - Config parameters: aws_access_key_id, aws_secret_access_key")
            raise ValueError("Invalid AWS credentials. Please configure AWS credentials.")
        print("✅ AWS credentials validated successfully")
        
        deployment_id = f"gpuforge-{config.name}-{uuid.uuid4().hex[:8]}"
        deployment_dir = self.deployments_dir / deployment_id
        deployment_dir.mkdir(exist_ok=True)
        
        print(f"\n📁 Step 2: Setting up deployment workspace...")
        print(f"   📂 Directory: {deployment_dir}")
        print(f"   🏷️ Deployment ID: {deployment_id}")
        print(f"   🔧 Provider: AWS")
        
        try:
            print("\n📝 Step 3: Generating Terraform infrastructure code...")
            # Generate Terraform configuration
            terraform_config = self._generate_aws_terraform(config, deployment_id)
            terraform_file = deployment_dir / "main.tf"
            
            with open(terraform_file, 'w') as f:
                f.write(terraform_config)
            print(f"   ✅ Generated main.tf ({len(terraform_config)} characters)")
            
            # Generate variables file
            variables = self._generate_terraform_variables(config)
            vars_file = deployment_dir / "terraform.tfvars"
            with open(vars_file, 'w') as f:
                for key, value in variables.items():
                    if isinstance(value, str):
                        f.write(f'{key} = "{value}"\n')
                    else:
                        f.write(f'{key} = {value}\n')
            print(f"   ✅ Generated terraform.tfvars ({len(variables)} variables)")
            
            # Create user data script
            user_data_script = self._generate_user_data_script(config, deployment_id)
            user_data_file = deployment_dir / "user_data.sh"
            with open(user_data_file, 'w') as f:
                f.write(user_data_script)
            print(f"   ✅ Generated user_data.sh ({len(user_data_script)} characters)")
            
            print("\n🔧 Step 4: Initializing Terraform...")
            print(f"   📂 Working directory: {deployment_dir}")
            result = subprocess.run(
                ["terraform", "init"],
                cwd=deployment_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"❌ Terraform initialization failed:")
                print(f"   Error: {result.stderr}")
                raise RuntimeError(f"Terraform init failed: {result.stderr}")
            print("   ✅ Terraform initialized successfully")
            
            print("\n📋 Step 5: Planning infrastructure changes...")
            print("   🔍 Analyzing required AWS resources...")
            result = subprocess.run(
                ["terraform", "plan", "-var-file=terraform.tfvars"],
                cwd=deployment_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"❌ Terraform planning failed:")
                print(f"   Error: {result.stderr}")
                raise RuntimeError(f"Terraform plan failed: {result.stderr}")
            
            print("   ✅ Infrastructure plan created successfully")
            print("\n📊 Terraform Plan Summary:")
            print("-" * 40)
            print(result.stdout[-1000:])  # Show last 1000 chars of plan
            print("-" * 40)
            
            print("\n⚠️ COST WARNING: This will create REAL AWS resources that incur charges!")
            print(f"   💰 Estimated cost: ~${self._estimate_hourly_cost(config):.2f}/hour")
            print(f"   📅 Auto-shutdown: {config.auto_shutdown_hours} hours")
            confirm = input(f"\n🚀 Deploy real AWS infrastructure for '{config.name}'? Type 'yes' to proceed: ")
            if confirm.lower() != 'yes':
                print("❌ Deployment cancelled by user - no charges incurred")
                return ""
            
            print("\n🚀 Step 6: Deploying AWS infrastructure...")
            print("   ⏱️ This may take several minutes...")
            print("   📡 Creating EC2 instance, security groups, and storage...")
            result = subprocess.run(
                ["terraform", "apply", "-auto-approve", "-var-file=terraform.tfvars"],
                cwd=deployment_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"❌ Terraform deployment failed:")
                print(f"   Error: {result.stderr}")
                raise RuntimeError(f"Terraform apply failed: {result.stderr}")
            print("   ✅ AWS infrastructure deployed successfully!")
            
            print("\n📤 Step 7: Retrieving deployment outputs...")
            # Parse Terraform outputs
            outputs = self._parse_terraform_outputs(deployment_dir)
            print(f"   ✅ Retrieved {len(outputs)} output values")
            
            print("\n💾 Step 8: Saving deployment information...")
            # Save deployment info
            deployment_info = {
                "deployment_id": deployment_id,
                "config": config.__dict__,
                "status": "running",
                "created_at": datetime.now().isoformat(),
                "terraform_dir": str(deployment_dir),
                "outputs": outputs,
                "provider": "aws"
            }
            
            self.active_deployments[deployment_id] = deployment_info
            self._save_deployments()
            print("   ✅ Deployment information saved")
            
            print("\n🎉 AWS Deployment Completed Successfully!")
            print("="*60)
            print(f"   🏷️ Deployment ID: {deployment_id}")
            print(f"   🆔 Instance ID: {outputs.get('instance_id', 'N/A')}")
            print(f"   🌐 Public IP: {outputs.get('public_ip', 'N/A')}")
            print(f"   🔑 SSH Command: ssh -i ~/.ssh/{config.key_pair_name}.pem ubuntu@{outputs.get('public_ip', 'N/A')}")
            if outputs.get('public_ip'):
                print(f"   📓 Jupyter URL: http://{outputs.get('public_ip')}:8888")
                print(f"   📊 TensorBoard: http://{outputs.get('public_ip')}:6006")
            print(f"   💰 Remember: Instance is accruing charges at ~${self._estimate_hourly_cost(config):.2f}/hour")
            print("   ⚠️ Don't forget to terminate when done to avoid charges!")
            
            return deployment_id
            
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            # Cleanup on failure
            self._cleanup_failed_deployment(deployment_dir)
            raise
    
    def _validate_aws_credentials(self, config: RealDeploymentConfig) -> bool:
        """Validate AWS credentials"""
        try:
            # Check for credentials in environment or config
            aws_access_key = config.aws_access_key_id or os.environ.get('AWS_ACCESS_KEY_ID')
            aws_secret_key = config.aws_secret_access_key or os.environ.get('AWS_SECRET_ACCESS_KEY')
            
            if not aws_access_key or not aws_secret_key:
                return False
            
            # Try a simple AWS API call to validate
            import boto3
            session = boto3.Session(
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key,
                region_name=config.region
            )
            
            # Test credentials by listing regions
            ec2 = session.client('ec2')
            ec2.describe_regions()
            return True
            
        except ImportError:
            print("⚠️ boto3 not installed. Install with: pip install boto3")
            return False
        except Exception as e:
            logger.error(f"AWS credential validation failed: {e}")
            return False
    
    def _generate_aws_terraform(self, config: RealDeploymentConfig, deployment_id: str) -> str:
        """Generate real Terraform configuration for AWS"""
        
        # Auto-select GPU-optimized AMI if not provided
        ami_id = config.ami_id or self._get_gpu_optimized_ami(config.region)
        
        terraform_config = f'''
terraform {{
  required_providers {{
    aws = {{
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }}
  }}
}}

provider "aws" {{
  region = var.aws_region
}}

# Variables
variable "aws_region" {{
  description = "AWS region"
  type        = string
  default     = "{config.region}"
}}

variable "instance_type" {{
  description = "EC2 instance type"
  type        = string
  default     = "{config.instance_type}"
}}

variable "key_pair_name" {{
  description = "AWS key pair name"
  type        = string
  default     = "{config.key_pair_name}"
}}

variable "deployment_name" {{
  description = "Deployment name"
  type        = string
  default     = "{config.name}"
}}

# Data sources
data "aws_availability_zones" "available" {{
  state = "available"
}}

# Security Group
resource "aws_security_group" "gpuforge_sg" {{
  name_prefix = "gpuforge-{deployment_id}-"
  description = "Security group for GPUForge ML instance"
  
  # SSH access
  ingress {{
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "SSH"
  }}
  
  # Jupyter notebook
  ingress {{
    from_port   = 8888
    to_port     = 8888
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "Jupyter"
  }}
  
  # TensorBoard
  ingress {{
    from_port   = 6006
    to_port     = 6006
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
    description = "TensorBoard"
  }}
  
  # All outbound traffic
  egress {{
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }}
  
  tags = {{
    Name = "gpuforge-{deployment_id}"
    Project = "GPUForge"
    Environment = "ml-training"
  }}
}}

# EBS volume for data storage
resource "aws_ebs_volume" "ml_data" {{
  availability_zone = data.aws_availability_zones.available.names[0]
  size              = {config.storage_size_gb}
  type              = "gp3"
  
  tags = {{
    Name = "gpuforge-{deployment_id}-data"
    Project = "GPUForge"
  }}
}}

# EC2 Instance
resource "aws_instance" "gpu_instance" {{
  ami                    = "{ami_id}"
  instance_type          = var.instance_type
  key_name              = var.key_pair_name
  vpc_security_group_ids = [aws_security_group.gpuforge_sg.id]
  availability_zone     = data.aws_availability_zones.available.names[0]
  
  # Instance store optimization for NVMe
  ebs_optimized = true
  
  # User data script for initial setup
  user_data = file("${{path.module}}/user_data.sh")
  
  # Root volume
  root_block_device {{
    volume_type           = "gp3"
    volume_size           = 50
    delete_on_termination = true
    encrypted             = true
  }}
  
  tags = {{
    Name = "gpuforge-{deployment_id}"
    Project = "GPUForge"
    Environment = "ml-training"
    AutoShutdown = "enabled"
    MaxHours = "{config.auto_shutdown_hours}"
  }}
  
  {f'instance_market_options {{ market_type = "spot" }}' if config.use_spot_instances else ''}
}}

# Attach EBS volume
resource "aws_volume_attachment" "ml_data_attachment" {{
  device_name = "/dev/sdf"
  volume_id   = aws_ebs_volume.ml_data.id
  instance_id = aws_instance.gpu_instance.id
}}

# Outputs
output "instance_id" {{
  description = "ID of the EC2 instance"
  value       = aws_instance.gpu_instance.id
}}

output "public_ip" {{
  description = "Public IP address of the instance"
  value       = aws_instance.gpu_instance.public_ip
}}

output "public_dns" {{
  description = "Public DNS name of the instance"
  value       = aws_instance.gpu_instance.public_dns
}}

output "security_group_id" {{
  description = "ID of the security group"
  value       = aws_security_group.gpuforge_sg.id
}}

output "deployment_id" {{
  description = "GPUForge deployment ID"
  value       = "{deployment_id}"
}}
'''
        
        return terraform_config
    
    def _generate_terraform_variables(self, config: RealDeploymentConfig) -> Dict:
        """Generate Terraform variables"""
        return {
            "aws_region": config.region,
            "instance_type": config.instance_type,
            "key_pair_name": config.key_pair_name,
            "deployment_name": config.name
        }
    
    def _generate_user_data_script(self, config: RealDeploymentConfig, deployment_id: str) -> str:
        """Generate user data script for instance initialization"""
        return f'''#!/bin/bash
set -e

# Update system
apt-get update -y
apt-get upgrade -y

# Install basic tools
apt-get install -y curl wget git htop nvtop

# Install NVIDIA drivers (if not already present)
if ! command -v nvidia-smi &> /dev/null; then
    apt-get install -y nvidia-driver-535
fi

# Install Miniconda
if [ ! -d "/opt/miniconda3" ]; then
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/miniconda3
    chown -R ubuntu:ubuntu /opt/miniconda3
fi

# Add conda to PATH
echo 'export PATH="/opt/miniconda3/bin:$PATH"' >> /home/ubuntu/.bashrc

# Install GPUForge
sudo -u ubuntu bash -c 'cd /home/ubuntu && git clone https://github.com/MohibShaikh/GPUForge.git'
sudo -u ubuntu bash -c 'cd /home/ubuntu/GPUForge && /opt/miniconda3/bin/pip install -r requirements.txt'

# Create ML environment
sudo -u ubuntu bash -c 'cd /home/ubuntu/GPUForge && /opt/miniconda3/bin/python gpuforge.py ml-env --profile {config.environment_profile}'

# Setup auto-shutdown script
cat > /home/ubuntu/auto_shutdown.py << 'EOF'
import time
import subprocess
import datetime

max_hours = {config.auto_shutdown_hours}
start_time = datetime.datetime.now()

while True:
    time.sleep(300)  # Check every 5 minutes
    elapsed = (datetime.datetime.now() - start_time).total_seconds() / 3600
    
    if elapsed > max_hours:
        print(f"Auto-shutdown: {config.auto_shutdown_hours} hours elapsed")
        subprocess.run(["sudo", "shutdown", "-h", "now"])
        break
EOF

# Start auto-shutdown script in background
sudo -u ubuntu nohup /opt/miniconda3/bin/python /home/ubuntu/auto_shutdown.py > /home/ubuntu/auto_shutdown.log 2>&1 &

# Mount additional EBS volume
mkdir -p /mnt/ml-data
echo '/dev/xvdf /mnt/ml-data ext4 defaults,nofail 0 2' >> /etc/fstab

# Format and mount the EBS volume
sleep 30  # Wait for volume attachment
if [ -b /dev/xvdf ]; then
    mkfs.ext4 /dev/xvdf
    mount /mnt/ml-data
    chown ubuntu:ubuntu /mnt/ml-data
fi

# Create setup completion marker
touch /home/ubuntu/gpuforge_setup_complete
chown ubuntu:ubuntu /home/ubuntu/gpuforge_setup_complete

echo "GPUForge instance setup completed at $(date)" > /home/ubuntu/setup.log
'''
    
    def _get_gpu_optimized_ami(self, region: str) -> str:
        """Get latest GPU-optimized AMI for region"""
        # AWS Deep Learning AMI GPU PyTorch (Ubuntu 20.04)
        ami_map = {
            "us-east-1": "ami-0c7217cdde317cfec",
            "us-west-2": "ami-0318c7e74b9f8e6dc", 
            "eu-west-1": "ami-0bf84c42e04519c85",
            "us-east-2": "ami-0e312c9bd0d0a6f88",
            "ap-southeast-1": "ami-0c6b8b0e3b6f8a5c5"
        }
        
        return ami_map.get(region, ami_map["us-east-1"])  # Default to us-east-1
    
    def _estimate_hourly_cost(self, config: RealDeploymentConfig) -> float:
        """Estimate hourly cost for the instance configuration"""
        # AWS pricing (approximate, varies by region - US East 1 pricing)
        pricing_map = {
            # General purpose
            't3.micro': 0.0104, 't3.small': 0.0208, 't3.medium': 0.0416,
            't3.large': 0.0832, 't3.xlarge': 0.1664, 't3.2xlarge': 0.3328,
            
            # Compute optimized
            'c5.large': 0.085, 'c5.xlarge': 0.17, 'c5.2xlarge': 0.34,
            'c5.4xlarge': 0.68, 'c5.9xlarge': 1.53, 'c5.18xlarge': 3.06,
            
            # GPU instances (most common for ML)
            'g4dn.xlarge': 0.526, 'g4dn.2xlarge': 0.752, 'g4dn.4xlarge': 1.204,
            'g4dn.8xlarge': 2.176, 'g4dn.12xlarge': 3.912, 'g4dn.16xlarge': 4.352,
            
            # High-end GPU (for serious ML workloads)
            'p3.2xlarge': 3.06, 'p3.8xlarge': 12.24, 'p3.16xlarge': 24.48,
            'p4d.24xlarge': 32.77, 'p5.48xlarge': 98.32,
            
            # Memory optimized
            'r5.large': 0.126, 'r5.xlarge': 0.252, 'r5.2xlarge': 0.504,
            'r5.4xlarge': 1.008, 'r5.8xlarge': 2.016, 'r5.12xlarge': 3.024,
        }
        
        base_cost = pricing_map.get(config.instance_type, 1.0)  # Default $1/hour if unknown
        
        # Add EBS storage cost (gp3 pricing ~$0.08 per GB per month)
        storage_cost_per_hour = config.storage_size_gb * 0.08 / (24 * 30)  # Monthly to hourly
        
        # Apply spot instance discount
        if config.use_spot_instances:
            base_cost *= 0.3  # Spot instances are typically 70% cheaper
        
        total_cost = base_cost + storage_cost_per_hour
        return total_cost
    
    def _parse_terraform_outputs(self, deployment_dir: Path) -> Dict:
        """Parse Terraform outputs"""
        try:
            result = subprocess.run(
                ["terraform", "output", "-json"],
                cwd=deployment_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                outputs = json.loads(result.stdout)
                # Extract values from Terraform output format
                return {key: value["value"] for key, value in outputs.items()}
            else:
                logger.warning(f"Could not parse Terraform outputs: {result.stderr}")
                return {}
                
        except Exception as e:
            logger.error(f"Error parsing Terraform outputs: {e}")
            return {}
    
    def _cleanup_failed_deployment(self, deployment_dir: Path):
        """Cleanup failed deployment"""
        try:
            subprocess.run(
                ["terraform", "destroy", "-auto-approve"],
                cwd=deployment_dir,
                capture_output=True
            )
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def terminate_deployment(self, deployment_id: str) -> bool:
        """Terminate real cloud deployment"""
        if deployment_id not in self.active_deployments:
            print(f"❌ Deployment {deployment_id} not found")
            return False
        
        deployment = self.active_deployments[deployment_id]
        terraform_dir = Path(deployment["terraform_dir"])
        
        if not terraform_dir.exists():
            print(f"❌ Terraform directory not found: {terraform_dir}")
            return False
        
        try:
            print(f"🛑 Terminating deployment {deployment_id}...")
            
            # Terraform destroy
            result = subprocess.run(
                ["terraform", "destroy", "-auto-approve"],
                cwd=terraform_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                print(f"⚠️ Terraform destroy had issues: {result.stderr}")
                print("Please check AWS console for remaining resources")
            
            # Remove from active deployments
            del self.active_deployments[deployment_id]
            self._save_deployments()
            
            print(f"✅ Deployment {deployment_id} terminated")
            return True
            
        except Exception as e:
            print(f"❌ Error terminating deployment: {e}")
            return False
    
    def list_deployments(self) -> List[Dict]:
        """List all active deployments"""
        return list(self.active_deployments.values())
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict]:
        """Get real deployment status from AWS"""
        if deployment_id not in self.active_deployments:
            return None
        
        deployment = self.active_deployments[deployment_id]
        
        # Get real status from AWS
        try:
            if deployment["provider"] == "aws":
                import boto3
                region = deployment["config"]["region"]
                session = boto3.Session(region_name=region)
                ec2 = session.client('ec2')
                
                if "outputs" in deployment and "instance_id" in deployment["outputs"]:
                    instance_id = deployment["outputs"]["instance_id"]
                    response = ec2.describe_instances(InstanceIds=[instance_id])
                    
                    if response["Reservations"]:
                        instance = response["Reservations"][0]["Instances"][0]
                        deployment["real_status"] = instance["State"]["Name"]
                        deployment["last_checked"] = datetime.now().isoformat()
                        self._save_deployments()
                        
        except Exception as e:
            logger.error(f"Error checking deployment status: {e}")
            deployment["real_status"] = "unknown"
        
        return deployment

# CLI integration function
def deploy_real_cloud(
    name: str,
    provider: str = "aws",
    region: str = "us-east-1", 
    instance_type: str = "g4dn.xlarge",
    use_spot: bool = False,
    max_cost: float = 10.0
) -> str:
    """Deploy real cloud infrastructure"""
    
    config = RealDeploymentConfig(
        name=name,
        provider=provider,
        region=region,
        instance_type=instance_type,
        use_spot_instances=use_spot,
        max_hourly_cost=max_cost
    )
    
    deployer = RealCloudDeployer()
    
    if provider == "aws":
        return deployer.deploy_aws_instance(config)
    else:
        raise NotImplementedError(f"Provider {provider} not yet implemented") 