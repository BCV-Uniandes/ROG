from setuptools import setup, find_packages

setup(
    name='ROG_3dsegm',
    packages=find_packages(exclude=['test']),
    package_dir={'rog': 'rog'},
    version='1.0',
    description='Towards Robust General Medical Image Segmentation',
    author='Laura Daza',
    install_requires=[
        "setuptools>=18.0",
        "torch>=1.6.0",
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "nibabel",
        "batchgenerators"
    ],
)
