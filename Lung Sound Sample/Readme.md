# Project description
Coming soon

# Project to-do list:
https://docs.google.com/document/d/1ufPW4vJU0QmhRtp4pt_VTMG5bGchrA2ma9VUqxs1CVY/edit?usp=sharing

# Environment Set-up:
Overview: http://conda.pydata.org/docs/using/envs.html
Anaconda virtual environments are not compatible across platforms. So, I will create environment files for each OS that I use. Here are also the list of commands required to set up an environment on each platform. Sometimes, Theano and Lasange will require updated versions of Numpy and Scipy. It is best to update those separately using conda update in the root directory before cloning.
Steps:
Install Anaconda
conda create -n pulmonary-diagnostics-lancet-win64 python=3.4 anaconda
activate pulmonary-diagnostics-lancet-win64
conda update scipy numpy
pip install sklearn-pandas
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

### Linux
Install Anaconda
conda create -n pulmonary-diagnostics-lancet-linux python=3.4 anaconda
source activate pulmonary-diagnostics-lancet-linux
conda update scipy numpy
pip install sklearn-pandas
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip


### Future updates?
https://pythonhosted.org/nolearn/


## Commands
### Linux
conda env create -f environment-linux.yml
source activate pulmonary-diagnostics-lancet

### Windows 64
conda env create -f environment-win64.yml
activate pulmonary-diagnostics-lancet

