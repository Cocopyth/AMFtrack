# AMF segmentation and analysis
AMF segmentation


# Setup
## Setup with conda
*For the script*

```bash
conda install -c open3d-admin open3d==0.9.0
conda install -c anaconda scipy
conda install -c anaconda pandas
conda install -c anaconda networkx
conda install -c conda-forge matplotlib
pip install pymatreader
conda install -c anaconda numpy
conda install -c conda-forge opencv
pip install imageio #use pip here to avoid conflict
conda install -c conda-forge jupyterlab
pip install pycpd
pip install cython
git clone https://github.com/gattia/cycpd
cd cycpd
sudo python setup.py install
pip install bresenham
conda install scikit-image
conda install -c conda-forge scikit-learn 
pip install Shapely
pip install tqdm
pip install dropbox
```
<!-- - conda install -c anaconda ipykernel -->

*For nice display*
```bash
conda install -c conda-forge ipympl
conda install -c conda-forge nodejs
conda install -c conda-forge/label/gcc7 nodejs
conda install -c conda-forge/label/cf201901 nodejs
conda install -c conda-forge/label/cf202003 nodejs
jupyter labextension install @jupyter-widgets/jupyterlab-manager
jupyter lab build
```

useful jupyterlab extensions:
https://github.com/jpmorganchase/jupyter-fs

## Setup for Linux

### Setting up environment

From base folder:
`virtualenv --python=python3 venv`
(or replace `python3` by the path to the python version you want to clone.)

Launching environment:
`source venv/bin/activate`

### Install requiered packages

Activate the environnement before launching

`pip3 install -r requirements.txt`

Additionnal packages to install:

```bash
git clone https://github.com/gattia/cycpd
cd cycpd
sudo python setup.py install
```

Install Fiji:

Chose a location on the computer and download:
https://imagej.net/software/fiji/downloads

Install anisotropic filtering:

Chose a location on the computer and download:
http://forge.cbp.ens-lyon.fr/redmine/projects/anifilters

### Install the package in editable mode

For better display:

`jupyter labextension install @jupyter-widgets/jupyterlab-manager`
`jupyter lab build`

### Install the package in editable mode
Remove the *pyproject.toml* file (for poetry)

To run from the base folder:
(will run the setup.py script)
`pip install -e .`

### Local.env file

Create a text file named `local.env` in the base folder
(for example: `touch local.env`)

And fill the file with the following lines and adapt them to your situation:

```
DATA_PATH=/home/cbisot/pycode/data_info.json
FIJI_PATH=/home/cbisot/Fiji.app/ImageJ-linux64
TEMP_PATH=/scratch-shared/amftrack/temp #careful no backslash here
STORAGE_PATH=/scratch-shared/amftrack/temp #careful no backslash here
PASTIS_PATH=/home/cbisot/anis_filter/anifilters/bin/ani2D 
#the path to the executable of anisotropic filtering
SLURM_PATH=/scratch-shared/amftrack/slurm #this is for parallelizez job on snellius
API_KEY=
```

To have access to a path: 
Always import from the util.sys
### Formattage

Le formatage du code est fait avec `black`