from setuptools import setup, Extension


def readme():
    with open('README.md') as f:
        return f.read()
        
        
setup(
    name='plm_entropy',
    version='0.1',
    description='The pLM entropy Python package',
    long_description= readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/spyros-lytras/plm_entropy',
    download_url = 'https://github.com/spyros-lytras/plm_entropy/dist/plm_entropy-0.1.tar',
    classifiers=[
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Programming Language :: Python",
    ],
    keywords='pLM LLM viruses conservation protein',
    project_urls={
    'Documentation': 'https://github.com/spyros-lytras/plm_entropy',
    },
    author='Spyros Lytras',
    author_email='spyros@ims.u-tokyo.ac.jp',
    license='MIT',
    packages=['plm_entropy'],
    install_requires=['biopython'],
    python_requires="~=3.12")