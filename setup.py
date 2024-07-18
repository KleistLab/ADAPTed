import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext

VERSION = "0.2.0"

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
    version=VERSION,
    packages=["adapted"],
    install_requires=[
        "numpy",
        "pandas",
        "tqdm",
        "bottleneck",
        "scipy",
    ],
    author="Wiep van der Toorn",
    author_email="w.vandertoorn@fu-berlin.de",
    description="Adapter detection in direct RNA sequencing reads.",
    license="CC BY-SA 4.0",
    keywords="dRNA-seq nanopore adapter detection",
    url="https://github.com/wvandertoorn/ADAPTed",
    entry_points={"console_scripts": ["adapted = adapted.main:main"]},
    include_package_data=True,
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
