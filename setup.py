from setuptools import setup



setup(
        name="handson_dash",
        version="0.0.1",
        install_requires=[
            'dash==0.34.0', 
            'dash-html-components==0.13.4', 
            'dash-core-components==0.42.0',
            'dash-table==3.1.11',
            'scikit-learn',
            'numpy',
            'jupyter'],
)
