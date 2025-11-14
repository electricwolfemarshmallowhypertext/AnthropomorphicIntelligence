from setuptools import setup, find_packages

setup(
    name="learnarena",
    version="1.0.0",
    description="LearnArena: A Benchmark Suite for Evaluating General Learning Ability of Language Models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(include=["agents", "agents.*", "envs", "envs.*", "wrappers", "wrappers.*", "learnarena_utils", "learnarena_utils.*"]),
    py_modules=["core"],
    include_package_data=True,
    package_data={
        "": ["*.json", "*.jsonl"],  # Include all JSON/JSONL files
        "envs": ["**/*.json", "**/data/*"],
        "learnarena_utils": ["data/*"],
    },
    install_requires=[
        "requests",
        "openai",
        "rich",
        "nltk",
        "chess",
        "networkx",
        "python-dotenv",
        "opencv-python",
        "websockets",
        "vllm",
        "numpy",
        "pandas",
        "torch",
        "transformers",
    ],
    python_requires='>=3.10',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
