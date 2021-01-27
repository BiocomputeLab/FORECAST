from distutils.core import setup
setup(
  name = 'flair',
  packages = setuptools.find_packages(),
  version = '1.0',
  description = 'Genetic design visualization',
  author = 'Pierre-Aurelien Gilliot',
  author_email = 'pa.gilliot@orange.fr',
  url = 'https://github.com/BiocomputeLab/FLAIR',
  download_url = 'https://github.com/VoigtLab/dnaplotlib/archive/1.0.tar.gz',
  keywords = ['MPRA','simulation','inference','Flow-Seq'],
  classifiers = [],
  scripts=['apps/quick.py', 'apps/library_plot.py']
)

# http://peterdowns.com/posts/first-time-with-pypi.html
# http://stackoverflow.com/questions/14863785/pip-install-virtualenv-where-are-the-scripts
