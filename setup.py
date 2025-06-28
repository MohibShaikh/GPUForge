from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="gpuforge",
    version="2.0.0",  # Complete cloud support (Phases 1-3)
    author="GPUForge Contributors",
    author_email="developers@gpuforge.dev",
    description="Universal GPU Environment Creator with Advanced Cloud Support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MohibShaikh/GPUForge",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: System :: Hardware",
        "Topic :: System :: Systems Administration",
    ],
    python_requires=">=3.8",
    install_requires=[
        "psutil>=5.8.0",
        "requests>=2.25.0",
        "pyyaml>=5.4.0",
        "colorama>=0.4.4",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800",
        ],
        "cloud": [
            "boto3>=1.17.0",         # AWS SDK
            "google-cloud>=0.34.0",  # Google Cloud SDK
            "azure-mgmt>=4.0.0",     # Azure SDK
            "kubernetes>=18.0.0",    # Kubernetes support
        ],
        "monitoring": [
            "prometheus-client>=0.8.0",
            "grafana-api>=1.0.3",
            "datadog>=0.41.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "gpuforge=gpuforge:cli_main",
        ],
    },
    keywords=[
        "gpu", "machine learning", "deep learning", "pytorch", "tensorflow", 
        "cuda", "environment", "automation", "cloud", "aws", "gcp", "azure",
        "orchestration", "scaling", "cost optimization", "deployment"
    ],
    project_urls={
        "Documentation": "https://github.com/MohibShaikh/GPUForge#readme",
        "Source": "https://github.com/MohibShaikh/GPUForge",
        "Tracker": "https://github.com/MohibShaikh/GPUForge/issues",
    },
) 