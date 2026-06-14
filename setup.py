from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).parent
README = ROOT.joinpath("README.md").read_text(encoding="utf-8")


setup(
    name="chat-spx",
    version="0.1.0",
    description="Close prices memorized in a transformer checkpoint",
    long_description=README,
    long_description_content_type="text/markdown",
    author="Stephen Lihn",
    license="GPL-3.0-or-later",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    package_data={"chat_spx": ["data/*.pt", "py.typed"]},
    python_requires=">=3.9",
    install_requires=["torch>=2.0"],
    extras_require={"test": ["pytest>=7"]},
    entry_points={"console_scripts": ["chat-spx=chat_spx.__main__:main"]},
    project_urls={
        "Homepage": "https://github.com/slihn/chat-spx",
        "Repository": "https://github.com/slihn/chat-spx",
        "Issues": "https://github.com/slihn/chat-spx/issues",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
