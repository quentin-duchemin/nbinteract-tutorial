import setuptools

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


with open("README.rst", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="wakapy",
    version="0.0.1",
    author="surister",
    author_email="surister98@gmail.com",
    description="Wakatime data manipulation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/quentin-duchemin/nbinteract-tutorial",
    include_package_data=True,
    python_requires='>3.6.0',
    install_requires=[
        'matplotlib'
    ],
    packages=setuptools.find_packages(),
)