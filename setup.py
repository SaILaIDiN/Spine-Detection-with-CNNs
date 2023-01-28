import os

from setuptools import find_packages, setup


def readme():
    with open("README.md", encoding="utf-8") as f:
        content = f.read()
    return content


if __name__ == "__main__":
    setup(
        name="spine-detection",
        version="0.0.1",
        description="Dendritic Spine Detection module",
        long_description=readme(),
        long_description_content_type="text/markdown",
        author="OpenMMLab",
        author_email="openmmlab@gmail.com",
        keywords="computer vision, object detection, dendritic spine",
        url="https://github.com/SaILaIDiN/Spine-Detection-with-CNNs",
        packages=find_packages(where="src"),
        package_dir={"": "src"},
        package_data={"spine_detection": ["configs/*"]},
        include_package_data=True,
        license="Apache License 2.0",
        # setup_requires=parse_requirements("requirements/build.txt"),
        # tests_require=parse_requirements("requirements/tests.txt"),
        # install_requires=parse_requirements("requirements/runtime.txt"),
        # extras_require={
        #     "all": parse_requirements("requirements.txt"),
        #     "tests": parse_requirements("requirements/tests.txt"),
        #     "build": parse_requirements("requirements/build.txt"),
        #     "optional": parse_requirements("requirements/optional.txt"),
        # },
        ext_modules=[],
        zip_safe=False,
    )
