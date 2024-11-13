from setuptools import find_packages, Extension, setup
import numpy as np
from Cython.Build import cythonize
from setuptools.command.build_ext import build_ext

import re

VERSIONFILE = "adapted/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

ext_modules = [
    Extension(
        name=str("adapted.detect._c_llr"),
        sources=[str("adapted/detect/_c_llr.pyx")],
        include_dirs=[np.get_include()],
        language="c++",
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

for e in ext_modules:
    e.cython_directives = {"embedsignature": True}


setup(
    name="ADAPTed",
    version=verstr,
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "tqdm",
        "bottleneck",
        "scipy",
        "torch==2.4.1",
    ],
    author="Wiep van der Toorn",
    author_email="w.vandertoorn@fu-berlin.de",
    description="Adapter detection in direct RNA sequencing reads.",
    license="CC BY-SA 4.0",
    keywords="dRNA-seq nanopore adapter detection",
    url="https://github.com/wvandertoorn/ADAPTed",
    entry_points={"console_scripts": ["adapted = adapted.main:main"]},
    include_package_data=True,
    package_data={
        "adapted.config": ["config_files/*.toml"],
        "adapted.models": ["*.pth"],
    },
    ext_modules=cythonize(ext_modules, language_level="3"),
    cmdclass={"build_ext": build_ext},
    test_suite="pytest",
    tests_require=["pytest"],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython,"
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Typing :: Typed",
    ],
)
