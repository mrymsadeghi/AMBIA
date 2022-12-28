
import os
from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtWidgets import QMessageBox
from gui.CustomWidgests import BlobDetectionParameter
from ControlMainWindow import ControlMainWindow
import GuiFunctions
from easysettings import EasySettings
import Switches_Static as st_switches
import Switches_Dynamic as dy_switches
from AMBIA_M4_Registration import func_ambia_registration, func_convert_coords, resize_images_for_registration, get_overlayed_registered_img
from AMBIA_M1_CellDetection import calculate_colocalized_blobs

################################################
###settings and configs
settings = EasySettings("myconfigfile.conf")
################################################

rootpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
dy_switches.set_rootpath(rootpath)
print(f"rootpath is: {rootpath}")

if st_switches.section_classifier_on:
    import AMBIA_M2_Localization as m2
if st_switches.atlas_type == "Adult":
    atlas_prepath= os.path.join(rootpath, "Gui_Atlases", "Adult_full_atlases")
elif st_switches.atlas_type == "P56":
    atlas_prepath= os.path.join(rootpath, "Gui_Atlases", "P56_full_atlases")


prepath = st_switches.PROCESSED_PATH
#slides_prepath= "C:/Slides/MB_slides/validation_slides/Rack1"
#slides_prepath= "E:/Slides/MB_Validation_slides"
slides_prepath= st_switches.SLIDES_PREPATH

def resize(event):
    global brainboundcoords
    # Resize PhotoViewers
    MainWindow.resize_photo_viewers()
    active_step=MainWindow.active_step
    try:
        # if active_step>1:
        #     return
        # Resize selection Image
        selected_brain_index=MainWindow.selected_brain
        MainWindow.create_brain_selection_regions(brainboundcoords)
        my_item=MainWindow.selection_items[selected_brain_index-1]
        my_item.setChecked(True)
        # Restore active step
        MainWindow.change_active_step(active_step)
    except Exception as e:
        # print(e)
        pass


def Main():
    global MainWindow
    import sys
    global slide_fullpath
    global brnum
    brnum = 1
    slide_fullpath = '1'
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = ControlMainWindow()
    MainWindow.resizeEvent = resize
    bind_functions()
    MainWindow.show()
    # geometry = app.desktop().availableGeometry()
    sys.exit(app.exec_())


def bind_functions():
    MainWindow.bind_loading_slide_operation(loading_slide_operation)
    MainWindow.bind_brain_detecting_operation(section_detection_operation)
    MainWindow.bind_select_brain_operation(section_select_operation)
    MainWindow.bind_blob_flip_operation(blob_flip_operation)
    MainWindow.bind_blob_detection_operation(blob_detection_operation)
    MainWindow.bind_blob_detection_accept_operation(blob_detection_accept)
    MainWindow.bind_landmark_auto_detect_opration(load_landmark_detection)
    MainWindow.bind_change_opacity_operation(registration_opacity_apply)
    MainWindow.bind_registration_accept_opration(registration_accept)
    MainWindow.bind_registration_preview_accept_opration(registration_preview_accept)
    MainWindow.bind_save_report_opration(save_report_operation)
    # MainWindow.bind_browse_atlas_operation(browse_atlas_operation)


def get_images_name_in_folder(labeled_atlas_folder):
    '''lists the atlas numbers available in the specified folder, based on the atlas img filenames
    :labeled_atlas_folder: the folder containing the atlas images
    '''
    file_list = []
    with os.scandir(labeled_atlas_folder) as entries:
        for entry in entries:
            if entry.is_file() and entries:
                file_name, extension = (os.path.splitext(entry.name))
                # print(extension)
                if extension == ".png":
                    file_list.append(file_name)
    try:
        file_list.sort(key=int)
    except:
        print('error')
        file_list.sort()
    return file_list


def browse_atlas_operation(labeled_atlas_folder=None):
    ''' Sets the atlas numbers list in the GUI '''
    if not labeled_atlas_folder:
        labeled_atlas_folder = MainWindow.get_atlas_folder(atlas_prepath)
    images_list = get_images_name_in_folder(labeled_atlas_folder)
    MainWindow.set_file_list(images_list)
    MainWindow.set_default_atlas_item_by_name ("42")


def loading_slide_operation():
    ''' This function is called when Load Slide button is clicked'''
    
    global slide_fullpath, slidename
    error_msg_1 = QMessageBox()
    error_msg_1.setWindowTitle("Error")
    error_msg_1.setText("The slide name format is not correct.")
    MainWindow.set_status_bar_text('Select the Slide Image...')
    slide_fullpath, slide_fullname = MainWindow.open_slide_dialog(slides_prepath)
    if not slide_fullpath:
        return
    MainWindow.enable_detect_brain()
    slidename = slide_fullname.split(".")[0]
    slide_prepath=os.path.split(slide_fullpath)[0]
    # Save Slide path
    settings.set("slide_prepath", slide_prepath)
    settings.save()
    MainWindow.change_status_bar_waiting()
    MainWindow.set_status_bar_text('Loading the Slide Image...')
    print('Loading the Slide Image...')
    MainWindow.reject_selection_region()
    read_slide_output = GuiFunctions.funcReadSlide(slide_fullpath, prepath)
    if read_slide_output == "em1":
        error_msg_1.exec_()
        return
    else:
        slide_image_name = read_slide_output
    MainWindow.set_slide_name_in_statusbar(slide_fullname)
    MainWindow.set_slide_image(slide_image_name)
    MainWindow.change_status_bar_default()
    MainWindow.set_status_bar_text('Loading Slide Image Done!')
    MainWindow.clear_landmark_nodes()


def section_detection_operation():
    MainWindow.clear_landmark_nodes()

    ''' This function is called when Detect button is clicked'''
    global brainboundcoords,tissuemask_fullpath
    MainWindow.change_status_bar_waiting()
    MainWindow.set_status_bar_text('Please Wait: Section Detection is running...')
    print('Section detection operation...')
    # ToDo: technical Dept
    try:
        brainboundcoords, tissuemask_fullpath = GuiFunctions.funcSectionDetection()
        MainWindow.set_brain_detection_image(tissuemask_fullpath)
        MainWindow.create_brain_selection_regions(brainboundcoords)
        
        MainWindow.change_status_bar_default()
        MainWindow.set_status_bar_text('Section Detection Done!')
    except Exception as e:
        print(e)
        MainWindow.set_status_bar_text('')


def section_select_operation():
    ''' This function is called when Select button is clicked'''
    global blob_detection_file_name, brnum
    
    MainWindow.change_status_bar_waiting()
    MainWindow.set_status_bar_text("Please Wait")
    brnum = MainWindow.accept_selection_region()
    dy_switches.set_ardent_reg_done_to_false()
    try:
        blob_detection_file_name, tissue_lm_detection_filename = GuiFunctions.get_section_images(brnum, brainboundcoords)
    except Exception as e:
        print(e)
    #blob_detection_file_name = "E:/PyProjects/MB_GUI/Processed/A1_R2_S8/S2/blevel_eq.png"
    #tissue_lm_detection_filename = "E:/PyProjects/MB_GUI/Processed/A1_R2_S8/S2/alevel_eq.png"


    if brnum:
        MainWindow.set_brain_number_in_statusbar(brnum)
        MainWindow.remove_all_blobs()
        MainWindow.change_status_bar_default()
        status_message = "Section : {} has been selected".format(brnum)
        MainWindow.finish_step(1)
        MainWindow.go_to_next_step()
        MainWindow.set_blob_detection_image(blob_detection_file_name)
        MainWindow.set_tissue_landmark_detection_image(tissue_lm_detection_filename)

    else:
        status_message = "No Sections Selected!"
    MainWindow.set_status_bar_text(status_message)
    MainWindow.clear_landmark_nodes()

    

def blob_settings_set_operation():
    ''' Reads the chosen parameters by the user and saves them'''
    # read properties
    blob_detection_parameters = MainWindow.get_current_blob_detection_parameters()
    new_blob_parameters = BlobDetectionParameter(blob_detection_parameters)
    blob_name = new_blob_parameters.color
    # set in dictionary
    try:
        self.blob_detection_parameters[blob_name] = new_blob_parameters
    except Exception as e:
        print(e)


def blob_flip_operation():
    ''' this function is called when flip button in NeroDetection Stage is clicked'''
    
    global blob_detection_file_name,brnum
    MainWindow.change_status_bar_waiting()
    MainWindow.set_status_bar_text("Please Wait")
    MainWindow.remove_all_blobs()
    MainWindow.set_blob_detection_cells_count(0, 0, 0) # Sets the counted Neuron numbers in GUI

    blob_detection_file_name, tissue_lm_detection_filename = GuiFunctions.section_flip_operation()

    MainWindow.set_blob_detection_image(blob_detection_file_name)
    MainWindow.set_tissue_landmark_detection_image(tissue_lm_detection_filename)
    
    status_message = "Section : {} has been selected".format(brnum)
    MainWindow.set_status_bar_text(status_message)
    MainWindow.change_status_bar_default()  



def blob_detection_operation():
    ''' This function is called when Apply button in NeuroDetection Stage is clicked'''

    #MainWindow.set_blob_detection_cells_count(10,10,10)
    if MainWindow.step_allowed_to_change(2):
        MainWindow.remove_all_blobs()
        blobs_parameters = MainWindow.get_all_blob_detection_parameters()
        MainWindow.change_status_bar_waiting()
        MainWindow.set_status_bar_text("Please wait: Neuron detection is running...")
        RED_COUNT, GREEN_COUNT,CO_COUNT,_,blobs_log_r,blobs_log_g,colocalized_blobs = GuiFunctions.funcBlobDetection(brnum, blobs_parameters)
        MainWindow.add_auto_detect_blobs(blobs_log_r,blobs_log_g,colocalized_blobs) # Displays the detected Neurons with circles in GUI
        #MainWindow.set_blob_detection_image(blob_detection_file_name)
        MainWindow.set_blob_detection_cells_count(RED_COUNT, GREEN_COUNT, CO_COUNT) # Sets the counted Neuron numbers in GUI
        MainWindow.blob_detection_perform() # Flags that blob detection is done
        status_message = "Neuron detection is Done!"
        MainWindow.change_status_bar_default()  
    else:
        status_message = "The Section detection is required to perform this step"
    MainWindow.set_status_bar_text(status_message)


def blob_detection_accept():
    ''' This function is called when Accept button in NeuroDetection Stage is clicked'''
    
    blob_detection_result = MainWindow.is_blob_detection_perform()
    if blob_detection_result:
        status_message = "Neuron is detection accepted"
        red_blobs, green_blobs = MainWindow.get_blobs()
        _, colocalized_blobs = calculate_colocalized_blobs(red_blobs, green_blobs)
        MainWindow.set_coloc_blobs(colocalized_blobs)
        MainWindow.finish_step(2)
        MainWindow.go_to_next_step()
        # Set Atlas PreviewFile  Default
    else:
        status_message = "You should perform Neuro detection first"
    
    set_default_atlas_number()
    MainWindow.set_status_bar_text(status_message)
    MainWindow.clear_landmark_nodes()

    

def set_default_atlas_number():
    ''' This function determines the default atlas number for next stage when Accept button in NeuroDetection Stage is clicked
    Determines atlas number either by manual selection or deep learning suggestion
    sets the chosen atlas as the atlas diplayed on the screen.
    '''
    global sectionfolder
    sectionfolder = os.path.join(prepath, slidename,f'S{brnum}') 
    atlas_folder = os.path.join(atlas_prepath,"labeled_atlases")
    MainWindow.set_default_atlas(atlas_folder)
    images_list = get_images_name_in_folder(atlas_folder)
    MainWindow.set_file_list(images_list)
    pred_imgpath = os.path.join(sectionfolder, 'alevel_eq.png')
    if st_switches.section_classifier_on:
        predicted_atlasnum = m2.predict_atlasnum(pred_imgpath)
        predicted_atlasquads = m2.predict_atlasQs(pred_imgpath)
        MainWindow.set_Qs_textbox_parameters(predicted_atlasquads)

    else:
        predicted_atlasnum = 45
        predicted_atlasquads = [0, 0, 0, 0]

    MainWindow.set_default_atlas_item_by_name(str(predicted_atlasnum))
    #MainWindow.set_default_atlas_item_by_index(42)
    MainWindow.load_preview_atlas()


def load_landmark_detection():
    atlas_address_path = MainWindow.get_atlas_preview_name()
    landmark_auto_detect_operation(atlas_address=atlas_address_path)


def landmark_auto_detect_operation(atlas_address=None):
    ''' This function is called when Apply button in LMDetection Stage is clicked	
    This func was originally intended for automatic LM detection. 	
    '''	
    global atlas_landmark_detection_file_name	
    global tissue_landmark_detection_file_name

    atlasnum = MainWindow.get_atlas_number()	
    labeled_atlas_LM_filepath = os.path.join(atlas_prepath,"labeled_atlases", str(atlasnum)+".png")
    unlabeled_atlas_LM_filepath = os.path.join(atlas_prepath,"unlabeled_atlases", str(atlasnum)+".png")
    general_unlabeled_atlas_filepath =  unlabeled_atlas_LM_filepath.replace("Adult_full_atlases","Adult_atlases")
    MainWindow.change_status_bar_waiting()	
    MainWindow.set_status_bar_text('Please wait: Atlas Selection is running')	
    tissue_landmark_detection_file_name = os.path.join(sectionfolder, 'alevel_eq.png')	
    atlas_landmark_detection_file_name = MainWindow.get_atlas_preview_name()
    if atlas_address:	
        if st_switches.section_QL_on:	
            predicted_atlasquads = MainWindow.get_Qs_textbox_parameters()	
            if MainWindow.get_atlas_mode():
                unlabeled_atlas_LM_filepath = m2.generate_tilted_atlas(predicted_atlasquads, sectionfolder)
                labeled_atlas_LM_filepath = unlabeled_atlas_LM_filepath
    atlas_unlabled_img_path = atlas_landmark_detection_file_name.replace("labeled","unlabeled")
    general_unlabeled_atlas_filepath = atlas_unlabled_img_path.replace("Adult_full_atlases","Adult_atlases")
    print("In Main ")
    print(tissue_landmark_detection_file_name)
    print(labeled_atlas_LM_filepath)
    print(unlabeled_atlas_LM_filepath)
    print(general_unlabeled_atlas_filepath)
    resized_tissue_LM_file_name, resized_atlas_LM_file_name = resize_images_for_registration(tissue_landmark_detection_file_name,labeled_atlas_LM_filepath,unlabeled_atlas_LM_filepath, sectionfolder, general_unlabeled_atlas_filepath)	
    MainWindow.set_tissue_landmark_detection_image(resized_tissue_LM_file_name) #################### must be done in earlier stages or removed from here?	
    MainWindow.set_atlas_landmark_detection_image(labeled_atlas_LM_filepath)	
    # This part applies if there is a code for automatic LM detection in the funcLandmarkDetection funtion	
    tissue_auto_landmarks = GuiFunctions.funcLandmarkDetection(tissue_landmark_detection_file_name, 400)	
    atlas_auto_landmarks = GuiFunctions.funcLandmarkDetection(atlas_landmark_detection_file_name, 500)	
    list_of_autodetected_nodes_list = 1	
    # displays the automatically detected LMs in the GUI	
    MainWindow.add_auto_detect_landmark(list_of_autodetected_nodes_list, tissue_auto_landmarks, atlas_auto_landmarks)	
    # Flags that Auto LM-detection is done	
    MainWindow.landmarks_detection_perform() 	
    MainWindow.change_status_bar_default()	
    MainWindow.set_status_bar_text('Atlas selection done')	
    dy_switches.set_ardent_reg_done_to_false()


def registration_opacity_apply():
    '''this function is called when Apply(opacity) button in Registration Sage is called'''
    opacity_value = MainWindow.get_opacity_value()
    overlayed_registered_imgs_path = get_overlayed_registered_img(opacity_value, sectionfolder)
    #print(f'opacity called - value:{opacity_value}')
    MainWindow.set_registration_image(overlayed_registered_imgs_path)


def registration_accept():
    ''' This function is called when Register button in LMDetection Stage is clicked
    Uses the funcAMBIAregister / outputs the registered section or atlas
    '''
    
    global registraion_file_name
    global reg_code
    # :((
    error_msg_3 = QMessageBox()
    error_msg_3.setWindowTitle("Error")
    error_msg_3.setText("The number of selected landmarks for both images must be equal.")

    MainWindow.change_status_bar_waiting()
    status_message = "Please Wait while Automatic registration is being performed..."
    MainWindow.set_status_bar_text(status_message)
    img1coords = []
    img2coords = []
    landmarks_detection = MainWindow.is_landmarks_detection_perform()
    #if landmarks_detection:
    MainWindow.finish_step(3)
    MainWindow.go_to_next_step()
    node_list1 = MainWindow.get_landmark_nodes()
    node_list2 = MainWindow.get_atlas_landmark_nodes()

    for item in node_list1:
        img1coords.append((int(item.x), int(item.y)))   # Section image LMs
    for item in node_list2:
        img2coords.append((int(item.x), int(item.y)))   # Atlas image LMs
    section_img_path = os.path.join(sectionfolder, 'alevel_eq.png')
    atlas_img_path = MainWindow.get_atlas_preview_name()
    atlasnum = MainWindow.get_atlas_number()
    print("atlasnum ", atlasnum)
    if len(img1coords)==len(img2coords):
        if atlasnum < 22:
            overlayed_registered_imgs_path, _ = m2.smartfunc_segment_1_20(section_img_path, sectionfolder)
        else:
            # source_img = section_img / target_img = atlas_img
            
            auto_registration_state = MainWindow.get_registration_mode()
            Ardent_reg_done = dy_switches.get_status_of_ardent_reg_done(auto_registration_state)
            target_img_path = os.path.join(sectionfolder, "atlas_labeled.png")
            source_img_path = os.path.join(sectionfolder, "source_img.png")
            overlayed_registered_imgs_path, reg_code = func_ambia_registration(sectionfolder, source_img_path, target_img_path, img1coords, img2coords, Ardent_reg_done, atlas_img_path, auto_registration_state)

            dy_switches.set_ardent_reg_done_to_true()
            if reg_code == "ardent_reg" or reg_code == "ardent_delauney_reg":
                registered_img_path = os.path.join(sectionfolder, 'ardent_reg_img.png')
            else:
                registered_img_path = os.path.join(sectionfolder, 'source_img.png')
            resized_atlas_path = os.path.join(sectionfolder, 'atlas_labeled.png')
            MainWindow.set_tissue_landmark_detection_image(registered_img_path)
            MainWindow.set_atlas_landmark_detection_image(resized_atlas_path)
    else:
        print("The number of Landmarks selected for both images must be the same")
        overlayed_registered_imgs_path = atlas_img_path
        error_msg_3.exec_()

    MainWindow.set_registration_image(overlayed_registered_imgs_path)
    MainWindow.change_status_bar_default()
    MainWindow.set_status_bar_text("Registration is done")


def registration_preview_accept():
    ''' This function is called when Accept button in Registration Preview Stage is clicked
    Gets the report image and report data(reportfile.txt) from funcAnalysis function
    Displays the report image and data
    '''

    global report_image_file_name
    error_msg_2 = QMessageBox()
    error_msg_2.setWindowTitle("Error")
    error_msg_2.setText("Report File might be open in Excel. Please close it and try again.")
    red_blobs_modified, green_blobs_modified = MainWindow.get_blobs()
    GuiFunctions.calculate_fp_fn_blobs(red_blobs_modified, green_blobs_modified)
    red_converted_coords, green_converted_coords, colocalized_converted_coords  = [], [], []
    colocalized_blobs_coords = MainWindow.get_coloc_blobs()
    blobs_coords_orig = {'red': red_blobs_modified, 'green': green_blobs_modified, 'coloc': colocalized_blobs_coords}
    if len(red_blobs_modified)>0:
        red_converted_coords = func_convert_coords(reg_code, red_blobs_modified, 'red')
    if len(green_blobs_modified)>0:
        green_converted_coords = func_convert_coords(reg_code, green_blobs_modified, 'green')
    if len(colocalized_blobs_coords)>0:
        colocalized_converted_coords = func_convert_coords(reg_code, colocalized_blobs_coords, 'coloc')
    GuiFunctions.save_to_saved_data_pickle(blobs_coords_orig, 'blobs_coords_orig')
    #GuiFunctions.save_to_pkl("blobs_coords_orig.pkl", blobs_coords_orig)
    reg_code_status = dy_switches.get_reg_code()
    atlas_path = MainWindow.get_atlas_preview_name()
    atlasnum = os.path.split(atlas_path)[-1].split(".")[0]
    mappingimgpath, report_image_file_name = GuiFunctions.funcAnalysis(atlasnum, brnum, atlas_prepath, red_converted_coords, green_converted_coords, colocalized_converted_coords)
    if report_image_file_name == "em2":
        error_msg_2.exec_()
        return
    else:
        MainWindow.set_report_image(report_image_file_name)
    f = open(os.path.join(mappingimgpath,  "reportfile_low.txt"), 'r+')
    message = f.read()
    MainWindow.set_report_text(message)
    MainWindow.finish_step(4)
    MainWindow.go_to_next_step()
    GuiFunctions.save_the_pickle()


def save_report_operation():
    MainWindow.save_report_dialog()


if __name__ == '__main__':
    Main()
