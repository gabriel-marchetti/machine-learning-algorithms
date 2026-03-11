from setuptools import setup, find_packages

setup(
    name='Temis',
    version='0.1',
    description='Machine Learning algorithms with fairness metrics',
    author='Gabriel',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.21.0',
        'pandas>=1.3.0',
        'scikit-learn>=1.0.0',
        'jax>=0.3.0',
        'jaxlib>=0.3.0',
        'matplotlib>=3.4.0',
    ],
    python_requires='>=3.8',
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
)