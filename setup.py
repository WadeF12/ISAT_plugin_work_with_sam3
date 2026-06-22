# -*- coding: utf-8 -*-

import codecs
from setuptools import setup, find_packages


def get_version():
    try:
        from ISAT_plugin_sam3_text_prompt.__init__ import __version__
        return __version__
    except FileExistsError:
        FileExistsError("__init__.py not exists.")


setup(
    name="isat-plugin-sam3-text-prompt",
    version=get_version(),
    author="your-name",
    author_email="your-email@example.com",
    description="ISAT plugin: SAM3 multi-category text-prompt batch auto annotation.",
    long_description=(codecs.open("README.md", encoding="utf-8").read()),
    long_description_content_type="text/markdown",

    url="https://github.com/yourname/ISAT_plugin_sam3_text_prompt",

    keywords=["isat-sam", "isat plugin", "sam3", "text prompt", "auto annotation"],
    license="Apache2.0",

    packages=find_packages(),
    include_package_data=True,

    python_requires=">=3.8",
    install_requires=[
        "isat-sam>=1.4.0",
        "numpy",
        "opencv-python-headless",  # headless 版本避免与 PyQt5 的 Qt 插件冲突
    ],

    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta",
        "Natural Language :: Chinese (Simplified)",
        "Natural Language :: English",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],

    entry_points={
        "isat.plugins": [
            "sam3_text_prompt_plugin = ISAT_plugin_sam3_text_prompt.main:SAM3TextPromptPlugin",
        ]
    },
)
