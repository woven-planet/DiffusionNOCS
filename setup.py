from setuptools import find_packages, setup

setup(
    name="diffusion_nocs",
    packages=find_packages(),
    package_data={
        "diffusion_nocs": [
            "py.typed",
            "resources/*.pt",
            "resources/*.pkl",
            "resources/**/*.safetensors",
            "resources/**/*.json",
        ]
    },
)
