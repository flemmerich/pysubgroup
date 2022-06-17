from setuptools import setup

setup(
    name='pysubgroup',
    version='0.7.3',
    packages=['pysubgroup', 'pysubgroup.tests'],
    package_data={'pysubgroup': ['data/credit-g.arff', 'data/titanic.csv']},
    url='http://florian.lemmerich.net/pysubgroup',
    license='',
    author='Florian Lemmerich, Felix Stamm, Martin Becker',
    author_email='florian@lemmerich.net',
    description='pysubgroup is a Python library for the data analysis task of subgroup discovery.',
    install_requires=[
        'pandas>=0.24.0',
        'scipy',
        'numpy',
        'matplotlib'
    ],
    tests_require=['pytest'],
    python_requires='>=3.6',
    classifiers=['Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'Programming Language :: Python',
                 'Topic :: Software Development',
                 'Topic :: Scientific/Engineering',
                 'Operating System :: Microsoft :: Windows',
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.5',
                 'Programming Language :: Python :: 3.6',
                 'Programming Language :: Python :: 3.7',
                 ('Programming Language :: Python :: '
                  'Implementation :: CPython')
                 ],
)
