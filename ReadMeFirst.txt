# install anaconda distribution of python, at this point version is not too important 
# because we will make virtual environment for the project 
https://www.anaconda.com/download/

# cd into the app directory

# make a new environment with conda:
conda create -n env_test python=3.6.1

# activate environment
(source) activate env_test

# install packages under current environment
pip install -r requirements.txt

# run app from terminal
python  heat_GS_app.py

# use app in browser 127.0.0.xxxx