import os

from setuptools import find_packages, setup


def readme():
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    return content


if __name__ == "__main__":
    setup(
        name="spine-detection",
        version="1.0.2",
        description="Dendritic Spine Detection module",
        long_description=readme(),
        long_description_content_type="text/markdown",
        author="Sercan Alipek, Fabian W. Vogel",
        author_email="alipek@physik.uni-frankfurt.de, fabian@vogel-nest.de",
        keywords="computer vision, object detection, dendritic spine",
        url="https://github.com/SaILaIDiN/Spine-Detection-with-CNNs",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        package_data={"spine_detection": ["configs/*"]},
        include_package_data=True,
        license="Apache License 2.0",
        ext_modules=[],
        zip_safe=False,
    )
