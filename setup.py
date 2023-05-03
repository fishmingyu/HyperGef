from setuptools import setup
import glob
import os
import os.path as osp
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# python setup.py build
# python setup.py install

src = "HyperGsys/source/"
include = "HyperGsys/include/"

def get_extensions():
    target = ['hgnnaggr', 'unignnaggr']
    extensions = []
    for tag in target:
        extensions_dir = osp.join(src)
        libraries = []
        extra_compile_args = {"cxx": ["-O2"]}
        extra_link_args = ["-lcusparse"]
        nvcc_flags = os.getenv("NVCC_FLAGS", "")
        nvcc_flags = [] if nvcc_flags == "" else nvcc_flags.split(" ")
        nvcc_flags += ["-O2"]
        extra_compile_args["nvcc"] = nvcc_flags
        main_file = osp.join(extensions_dir, tag, f"{tag}.cc")
        cuda_file = osp.join(extensions_dir, tag, f"{tag}_cuda.cu")
        sources = [main_file, cuda_file]
        extension = CUDAExtension(
                    f"{tag}",
                    sources,
                    include_dirs=[include],
                    extra_compile_args=extra_compile_args,
                    extra_link_args=extra_link_args,
                    libraries=libraries,
                )
        
        extensions += [extension]
    return extensions

install_requires = [
    "scipy",
    "sklearn",
    "torch"
]

test_requires = [
    "pytest",
]


setup(
    name='HyperGsys',
    version="0.1",
    author="yzm ys henry genghan",
    description="HyperGsys",
    long_description=long_description,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    # package_dir={"": "dgNN"},
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    extras_require={
        "test": test_requires,
    },
    python_requires=">=3.7",
    ext_modules=get_extensions(),
    cmdclass={
        'build_ext': BuildExtension
    },
)
