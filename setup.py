from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='segmentation-plot',
    version='0.2',
    author='Onofrey Lab',
    author_email='john.onofrey@yale.edu',
    description='Visualize image segmentations as contours using matplotlib.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/OnofreyLab/segmentation-plot',
    license='MIT',
    packages=find_packages(),
    package_data={"segmentation_plot": ["py.typed"]},
)
