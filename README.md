

<div align="center">
    <header><h1>System for bandwidth extension of narrow-band speech</h1></header>
    <a href="#">
        <img src="https://img.shields.io/badge/Python-3.10-efefef">
    </a>
    <a href="#">
        <img src="https://img.shields.io/badge/License-MIT-efefef">
    </a>
    <br>
    <a href="https://patents.google.com/patent/US7216074"><strong>Explore Patent in Google Patents »</strong></a>
</div>
<br>


Artificial Bandwidth Expansion for Narrow-band Speech (patent by David Malah) implemented in Python, and refactored for our usages.
This work is part of my thesis in the Department of Computer Science, University of Crete, supervised by Professor [Yannis Stylianou](https://www.csd.uoc.gr/CSD/index.jsp?custom=yannis_stylianou&lang=en) 
and Dr. [George Kafentzis](https://www.csd.uoc.gr/~kafentz/). Make sure to also see the results [project report](Artificial_Bandwidth_Expansion.pdf).

The experiments were carried out and produced by the jupyter notebook in [notebooks](notebooks/spectrograms.ipynb). You can listen
to the system's output speech files in [output_speechfiles](output_speechfiles).


## Getting Started

The project uses Poetry (version >=1.2.0) for package management and Miniconda3 for virtual environment. Make sure both are installed on either your Linux/MacOS or Windows.
The project's settings have been tested on Windows 10 and MacOS.


Create and activate a new conda environment using:

```sh
$ conda create -n sbe python=3.10 -y && conda activate sbe
```

Then, to install all required packages on your new environment use:

```sh
$ poetry install
```

Note: WSL uses a different conda than Windows. This is not a problem in general but when using an IDE, you need to somehow index the IDE to that conda environment.

Example, in your project in Pycharm:

* Choose File > Setting > Project > Python Interpreter > Add
* Choose WSL on the left. Linux = your Ubuntu
* Python interpreter path = `home/<your_name>/miniconda3/envs/<your_env>/bin/python3` -- this is the environment you have created in Ubuntu with Conda.

## Contributing
To add another library in the project, make sure to use `poetry add <libname>` to keep poetry files up-to-date with the required interpreter settings.

The project also uses `black` and `isort` code formatters for styling purposes. Prior to pushing, use

```sh
$ black .
```

to reformat all python files and

```sh
$ isort .
```

to sort imports alphabetically.