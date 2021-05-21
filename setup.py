from setuptools import setup

__version_info__ = (0, 1, 0)
__version__ = '.'.join(map(str, __version_info__))


setup(name='DrivenCorticalSheet',
      version=__version__,
      description='AMATH 575 project on weakly coupled phase synchonous oscillators\
                   with the addition of a driving oscillator',
      url='https://github.com/TomekFraczek/DrivenCorticalSheet',
      author=['Tomek Fraczek',
              'Michael Willy',
              'Group Contributors:',
              'Yundong Lin',
              'Blake Fletcher'],
      author_email=['michael  willy at gmail dot com','add yours here'],
      license='MIT',
      packages=[''],
      zip_safe=False
      )
