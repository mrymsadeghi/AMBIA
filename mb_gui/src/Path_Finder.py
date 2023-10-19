import os
import Switches_Static as st_switches

def return_root_path():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
def return_atlas_path():
    if st_switches.atlas_type == "Adult":
        atlas_prepath= os.path.join(return_root_path(), "Gui_Atlases", "Adult_full_atlases")
    
    elif st_switches.atlas_type == "Rat":
        atlas_prepath= os.path.join(return_root_path(), "Gui_Atlases", "Rat_atlases")
    return atlas_prepath
def return_prepath():
    prepath=os.path.join(return_root_path(), "Processed") 
    if "Processed" in os.listdir(return_root_path()):
        if os.path.isdir(prepath):
            print ("Processed folder already exist, skipping mkdir")
        else : 
            os.mkdir(prepath)
            print ("Processed folder created")
    else :
        os.mkdir(prepath)
        print ("Processed folder created")
    return prepath