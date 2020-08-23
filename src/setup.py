import setuptools

with open("README", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="janos",  # How you named your package folder (janos)
    packages=['janos'],   # Chose the same as "name"
    version="0.0.9",
    author="David Bergman",
    author_email="david.bergman@uconn.edu",
    description="JANOS",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="http://janos.opt-operations.com",
	install_requires = [],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
