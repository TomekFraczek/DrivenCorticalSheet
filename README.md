Driven Cortical Sheet

AMATH 575 group project on

["Generative models of cortical oscillations: neurobiological implications of the Kuramoto model” Breakspear, Heitmann, & Daffertshofer](https://www.frontiersin.org/articles/10.3389/fnhum.2010.00190/full)

A python implementation inspired by the above paper, with the addition of a driving oscillator


## Install
Should run with just a git clone and conda install

## Usage
Main functions for using the model all live in main.py, and plots are output to the `plots` folder in that
same directory.

#### main()
This function provides a command line interface. Arguments:
  - `--set` Name of the model spec to select from the json
  - `--path` Path to the json specification file. By default this is the current directory
  - `--out` Output path to save raw data and plots. By default this is `plots/` in the current directory
  - `--plot` Flag. If present will perform plotting right away

#### run()
Same functionality as main, but intended to be used from within python. Runs a simulation with `model` using
the desired parameter set from the config file. Optionally, will plot the results right away

#### model()
Run the model of a cortical sheet using the passed parameters. Usually used through run/main but can also
be run directly if passed a ready dictionary of parameters

#### plot()
Plot an animation (gif) of the evolution of phases over time. It can function in one of two modes:
  - *1:* Pass a string path to `data_folder`. This will load the oscillator data in the folder and prepare the animation
  - *2:* Pass the data directly to the remaining variables, this will plot and save w/o re-reading from disk
