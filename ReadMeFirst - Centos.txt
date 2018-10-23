# install anaconda distribution of python, at this point version is not too important 
# because we will make virtual environment for the project 
https://www.anaconda.com/download/


# on Centos make a new environment with venv
python3.6 -m venv /media/data/Environments/Shuyi/dash_test/

#activate environment on Centos
source /media/data/Environments/Shuyi/dash_test/bin/activate

# install packages under current environment
pip install -r requirements.txt

# run app from terminal
python  heat_GS_app.py

# use app in browser 127.0.0.xxxx
