# AMBIA

AMBIA (**A**utomated **M**ouse **B**rain **I**mage **A**nalysis) is a tool to fascilitate the analysis of 2D histological Mouse brain images. It has a modular design and consists of the following modules:
- Cell detection Module
- Localization Module
- Atlas generator module
- Registration Module

## Requirements

AMBIA is intentionally designed with few requirements. You can easily run Ambia on your local computer. Ofcourse computers with higher ram and stronger CPUs make Ambia run faster. Following a the requirements befor you install Ambia:
- Python > 3.7


## Installation

1. Install python, ideally version 3.7.9, on your system. 
2. Download Ambia package files from this Git repository
or you can use
`Git clone https://github.com/mrymsadeghi/AMBIA.git`
3. Use the requirements.txt file provided in the package to install the necessary python packages with optimal versions
`pip install -r requirements.txt`
4. Run the Main.py file to run Ambia

## Structure of files and folders
The files and folders in the Ambia Git repository consists of the following folers:
- Gui Atlases
- mb_gui
- models
- Processed
Keep this folder structure in your system for Ambia to work correctly.
Here is an explanation of the content and functionality of each folder
#### Gui Atlases folder
This folder contains the color coded coronal atlases of adult mouse brain or P56 mouse brain. In the file `Static_Switches.py` if you choose `atlas_type = Adult`, Ambia will use Adult atlases in this folder. Also it is noteworthy that the color codes of each atlas is contained in the folder `mb_gui/src/atlas_codes`.
#### mb_gui folder
This folder contains all the python code for the Cell detection, Localization, Atlas generator and Registration Module, as well as the GUI. The file `mb_gui/src/Switches_Static.py` contatins parameters that are editable by the users. Parameters such as for choosing the atlas type (`atlas_type`), turning auto registration on or off (`auto_registration`) and etc. 
#### models folder
This folder contains three different type of data
- Deep learning trained models for the localization module. Including the SL predictor, QL predictor, Group A segmentation modes
- Excel sheet templates in which Ambia exports data
- Files required for generating tilted 2D allen mouse brain atlas from the 3D atlas in the Allen_files folder
#### Processed folder
This is where the files and reports of the analysis of your mouse brain slices will be saved. Also as an example, we provide analysis of two mouse brain slices from two different slides are in this folder.