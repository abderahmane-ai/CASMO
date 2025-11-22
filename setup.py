"""
CASMO: Confident Adaptive Selective Momentum Optimizer

A production-ready PyTorch optimizer that extends Adam with confidence-based 
learning rate scaling using AGAR (Adaptive Gradient Alignment Ratio).

Key Features:
- Automatic noise robustness via AGAR metric
- Faster than AdamW on large models
- Drop-in replacement for Adam/AdamW
- Comprehensive testing and documentation

Example:
    >>> from casmo import CASMO
    >>> optimizer = CASMO(model.parameters(), lr=1e-3, weight_decay=0.01)
"""

from setuptools import setup, find_packages
import os

# Read long description from README if it exists
long_description = ""
readme_path = os.path.join(os.path.dirname(__file__), "README.md")
if os.path.exists(readme_path):
    with open(readme_path, "r", encoding="utf-8") as f:
        long_description = f.read()
else:
    long_description = __doc__

setup(
    name="casmo-optimizer",
    version="0.1.0",
    author="Abderahmane Ainouche",
    author_email="abderahmane.ainouche.ai@gmail.com",
    description="Noise-robust PyTorch optimizer with automatic confidence-based learning rate adaptation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/abderahmane-ai/CASMO",
    py_modules=["casmo"],
    packages=find_packages(exclude=["tests", "benchmarks", "data"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.20.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "benchmarks": [
            "torchvision>=0.11.0",
            "matplotlib>=3.5.0",
            "transformers>=4.20.0",
            "datasets>=2.0.0",
        ],
    },
    keywords="pytorch optimizer adam noise-robust machine-learning deep-learning",
    project_urls={
        "Bug Reports": "https://github.com/abderahmane-ai/CASMO/issues",
        "Source": "https://github.com/abderahmane-ai/CASMO",
    },
)
