import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="torchelper",
    version="0.1.3",
    author="huachao",
    author_email="huachao1001@qq.com",
    description="A helper library for pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/huachao1001/torch_helper",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)