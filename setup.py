from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gpuforge",
    version="1.0.0",
    author="GPUForge Contributors",
    author_email="",
    description="The Smart GPU Environment Creator - Forge perfect GPU environments in seconds",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/gpuforge",
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
        "Topic :: System :: Systems Administration",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "gpuforge=gpuforge:cli_main",
        ],
    },
    keywords=[
        "gpu", "cuda", "pytorch", "tensorflow", "ml", "ai", 
        "environment", "conda", "nvidia", "amd", "intel",
        "machine-learning", "deep-learning", "automation"
    ],
    project_urls={
        "Bug Reports": "https://github.com/your-username/gpuforge/issues",
        "Source": "https://github.com/your-username/gpuforge",
        "Documentation": "https://github.com/your-username/gpuforge#readme",
    },
) 