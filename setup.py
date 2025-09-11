from setuptools import setup, Extension


def readme():
    with open('README.md') as f:
        return f.read()
        
        
setup(
    name='plm_entropy',
    version='0',
    description='',
    long_description= readme(),
    long_description_content_type='text/markdown',
    url='',
    download_url = '',
    classifiers=[
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Programming Language :: Python",
    ],
    keywords='',
    project_urls={
    'Documentation': '',
    },
    author='Spyros Lytras',
    author_email='',
    license='MIT',
    packages=['plm_entropy'],
    install_requires=['biopython'],
    python_requires="~=3.12")