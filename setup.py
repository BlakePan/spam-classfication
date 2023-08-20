from setuptools import setup, find_packages

setup(
    name="spam-classfication",
    version="0.1.0",
    description="spam-classfication",
    author="Blake",
    author_email="",
    url="https://github.com/BlakePan/spam-classfication",
    packages=find_packages(),
    install_requires=[
        # List your project's dependencies here
        "numpy",
        "pandas",
        "torch",
        "torchvision",
        "scikit-learn",
        "transformers",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
