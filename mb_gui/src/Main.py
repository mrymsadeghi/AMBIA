if __name__ == '__main__':
    import os
    import Path_Finder 
    from PyQt5 import QtCore, QtGui, QtWidgets, Qt
    from PyQt5.QtWidgets import QMessageBox
    from gui.CustomWidgests import BlobDetectionParameter
    from ControlMainWindow import ControlMainWindow
    import GuiFunctions as guif
    from easysettings import EasySettings
    import Switches_Static as st_switches
    import Switches_Dynamic as dy_switches
    from AMBIA_M1_CellDetection import calculate_colocalized_blobs
    from AMBIA_M4_Registration import func_ambia_registration, func_convert_coords, resize_images_for_registration, get_overlayed_registered_img
    import cv2
    from PyQt5.QtWidgets import (
        QApplication,
        QWidget,
        QLabel,
        QSlider,
        QHBoxLayout,
        QVBoxLayout,
    )
    from PyQt5.QtGui import QPixmap, QImage, QIcon
    from PyQt5.QtCore import Qt
    import utils.img_processing as imgprc
    from PIL import Image
    import numpy as np
    import sys
    ################################################
    ###settings and configs
    settings = EasySettings("myconfigfile.conf")
    ################################################

    rootpath=Path_Finder.return_root_path()
    dy_switches.set_rootpath(rootpath)
    atlas_prepath=Path_Finder.return_atlas_path()
    print(f"rootpath is: {rootpath}")

    #if st_switches.section_classifier_on:
    import AMBIA_M2_Localization as m2



    try :
        if os.path.isdir(settings.get("slide_prepath")):
            slides_prepath= settings.get("slide_prepath")
        else :
            slides_prepath=rootpath
    except :
        slides_prepath=rootpath

    #brainboundcoords, tissuemask_fullpath = GuiFunctions.funcSectionDetection()
    prepath=Path_Finder.return_prepath()



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

            MainWindow.change_active_step(active_step)
        except Exception as e:
            # print(e)
            pass




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
            #print('error lin 88')
            tmp=[] 
            tmp1=[]
            for j in file_list:
                try :
                    int (j)
                    tmp.append(j)
                except :
                    tmp1.append(j)
            tmp.sort(key=int)
            tmp1.sort()
            return tmp+tmp1
        return file_list


    def browse_atlas_operation(labeled_atlas_folder=None):
        ''' Sets the atlas numbers list in the GUI '''
        if not labeled_atlas_folder:
            labeled_atlas_folder = MainWindow.get_atlas_folder(atlas_prepath)
        images_list = get_images_name_in_folder(labeled_atlas_folder)
        #MainWindow.set_file_list(images_list)
        MainWindow.set_default_atlas_item_by_name (images_list[0])




        

    def blob_settings_set_operation():
        ''' Reads the chosen parameters by the user and saves them'''
        # read properties
        blob_detection_parameters = MainWindow.get_current_blob_detection_parameters()
        new_blob_parameters = BlobDetectionParameter(blob_detection_parameters)
        blob_name = new_blob_parameters.color
        # set in dictionary
        try:
            #self.blob_detection_parameters[blob_name] = new_blob_parameters
            blob_detection_parameters[blob_name] = new_blob_parameters
        except Exception as e:

            print(e,"error line 188")

  
        

    


    class Window(ControlMainWindow):
        def __init__(self,Functions_object):
            super().__init__()
            #self.setWindowFlags (QtCore.Qt.WindowCloseButtonHint | QtCore.Qt.WindowMinimizeButtonHint)
            

            #self.setWindowFlags( QtCore.Qt.WindowMinimizeButtonHint | QtCore.Qt.WindowCloseButtonHint )
            self.showMaximized()
            self.setFixedSize(self.size())
            self.reg_done=False
            self.GuiFunctions=Functions_object
            self.bind_functions()
            self.brnum=1
            self.slide_fullpath="1"
            #self.resizeEvent=self.resize

        def save_report_operation(self):
            self.save_report_dialog()

        def registration_preview_accept(self):
            ''' This function is called when Accept button in Registration Preview Stage is clicked
            Gets the report image and report data(reportfile.txt) from funcAnalysis function
            Displays the report image and data
            '''
            #ouput of detection
            #self.number_of_blobs,self.CO_COUNT,self._,self.blobs_log,self.colocalized_blobs 
            #global report_image_file_name
            error_msg_2 = QMessageBox()
            error_msg_2.setWindowTitle("Error")
            error_msg_2.setText("Report File might be open in Excel. Please close it and try again.")
            #red_blobs_modified, green_blobs_modified = self.get_blobs()
            self.GuiFunctions.calculate_fp_fn_blobs(self.final_blobs,self.blobs_log)#red_blobs_modified, green_blobs_modified)
            #red_converted_coords, green_converted_coords, colocalized_converted_coords  = [], [], []
            #colocalized_blobs_coords = self.get_coloc_blobs()
            #blobs_coords_orig = {'red': red_blobs_modified, 'green': green_blobs_modified, 'coloc': colocalized_blobs_coords}
            blobs_coords_orig={}
            registered_coords={}
            colocalized_converted_coords=[]
            for i in range(len(st_switches.num_channels)):
                blobs_coords_orig[i]=self.final_blobs[i]
                if len(self.final_blobs)>0:
                    try:registered_coords[i]=func_convert_coords(self.reg_code,self.final_blobs[i], str(i),self.GuiFunctions)
                    except : registered_coords[i]=[]
            for i in  self.final_colocalized_blobs:
                tmp=[]
                #for j in i :
                try:tmp.append( func_convert_coords(self.reg_code, i, 'coloc',self.GuiFunctions))
                except:
                    tmp.append([])
                colocalized_converted_coords.append( tmp)
            registered_coords["coloc"]=colocalized_converted_coords
            """if len(red_blobs_modified)>0:
                red_converted_coords = func_convert_coords(self.reg_code, red_blobs_modified, 'red',self.GuiFunctions)
            if len(green_blobs_modified)>0:
                green_converted_coords = func_convert_coords(self.reg_code, green_blobs_modified, 'green',self.GuiFunctions)
            if len(colocalized_blobs_coords)>0:
                colocalized_converted_coords = func_convert_coords(self.reg_code,
                colocalized_blobs_coords, 'coloc',self.GuiFunctions)"""
            guif.save_to_saved_data_pickle(blobs_coords_orig, 'blobs_coords_orig')
            #GuiFunctions.save_to_pkl("blobs_coords_orig.pkl", blobs_coords_orig)
            reg_code_status = dy_switches.get_reg_code()
            atlas_path = self.get_atlas_preview_name()
            atlasnum = os.path.split(atlas_path)[-1].split(".")[0]
            blobs_parameters = self.get_all_blob_detection_parameters()
            mappingimgpath, report_image_file_name = self.GuiFunctions.funcAnalysis(atlasnum, self.brnum, atlas_prepath,
                                                                                    registered_coords,
                                                                                    colocalized_converted_coords,blobs_parameters)
            if report_image_file_name == "em2":
                error_msg_2.exec_()
                return
            else:
                self.set_report_image(report_image_file_name)
            f = open(os.path.join(mappingimgpath,  "reportfile_low.txt"), 'r+')
            message = f.read()
            self.set_report_text(message)
            self.finish_step(4)
            self.go_to_next_step()
            self.GuiFunctions.save_the_pickle()


        def bind_functions (self):
            self.bind_loading_slide_operation(self.loading_slide_operation)
            self.bind_brain_detecting_operation(self.section_detection_operation)
            self.bind_select_brain_operation(self.section_select_operation)
            self.bind_blob_flip_operation(self.blob_flip_operation)
            self.bind_blob_detection_operation(self.blob_detection_operation)
            self.bind_blob_detection_accept_operation(self.blob_detection_accept)
            self.bind_landmark_auto_detect_opration(self.load_landmark_detection)
            self.bind_change_opacity_operation(self.registration_opacity_apply)
            self.bind_registration_accept_opration(self.registration_accept)
            self.bind_registration_preview_accept_opration(self.registration_preview_accept)
            self.bind_save_report_opration(self.save_report_operation)

        def registration_accept(self):
            ''' This function is called when Register button in LMDetection Stage is clicked
            Uses the funcAMBIAregister / outputs the registered section or atlas
            '''



            # :((
            error_msg_3 = QMessageBox()
            error_msg_3.setWindowTitle("Error")
            error_msg_3.setText("The number of selected landmarks for both images must be equal.")

            self.change_status_bar_waiting()
            status_message = "Please Wait while Automatic registration is being performed..."
            self.set_status_bar_text(status_message)
            img1coords = []
            img2coords = []
            #landmarks_detection = self.is_landmarks_detection_perform()
            #if landmarks_detection:
            self.finish_step(3)
            self.go_to_next_step()
            node_list1 = self.get_landmark_nodes()
            node_list2 = self.get_atlas_landmark_nodes()

            for item in node_list1:
                img1coords.append((int(item.x), int(item.y)))   # Section image LMs
            for item in node_list2:
                img2coords.append((int(item.x), int(item.y)))   # Atlas image LMs
            section_img_path = os.path.join(self.sectionfolder, 'alevel_eq.png')
            atlas_img_path = self.get_atlas_preview_name()
            atlasnum = self.get_atlas_number()
            if len(img1coords)==len(img2coords):
                
                #if st_switches.atlas_type!="Rat" and atlasnum < 22:
                if st_switches.segmentation_1_20_on and atlasnum<22:
                    overlayed_registered_imgs_path, _ = m2.smartfunc_segment_1_20(section_img_path, self.sectionfolder)
                else:
                    # source_img = section_img / target_img = atlas_img
                    
                    auto_registration_state = self.get_registration_mode()
                    Ardent_reg_done = dy_switches.get_status_of_ardent_reg_done(auto_registration_state)
                    target_img_path = os.path.join(self.sectionfolder, "atlas_labeled.png")
                    source_img_path = os.path.join(self.sectionfolder, "source_img.png")
                    overlayed_registered_imgs_path, self.reg_code = func_ambia_registration(self.sectionfolder, source_img_path, target_img_path, img1coords, img2coords, Ardent_reg_done, atlas_img_path, auto_registration_state)

                    dy_switches.set_ardent_reg_done_to_true()
                    if self.reg_code == "ardent_reg" or self.reg_code == "ardent_delauney_reg":
                        registered_img_path = os.path.join(self.sectionfolder, 'ardent_reg_img.png')
                    else:
                        registered_img_path = os.path.join(self.sectionfolder, 'source_img.png')
                    resized_atlas_path = os.path.join(self.sectionfolder, 'atlas_labeled.png')
                    self.set_tissue_landmark_detection_image(registered_img_path)
                    self.set_atlas_landmark_detection_image(resized_atlas_path)
            else:
                print("The number of Landmarks selected for both images must be the same")
                overlayed_registered_imgs_path = atlas_img_path
                error_msg_3.exec_()

            self.set_registration_image(overlayed_registered_imgs_path)
            self.change_status_bar_default()
            self.set_status_bar_text("Registration is done")
            self.reg_done=True

        def registration_opacity_apply(self):
            '''this function is called when Apply(opacity) button in Registration Sage is called'''
            opacity_value = self.get_opacity_value()
            overlayed_registered_imgs_path = get_overlayed_registered_img(opacity_value, self.sectionfolder)
            #print(f'opacity called - value:{opacity_value}')
            self.set_registration_image(overlayed_registered_imgs_path)
        
        def load_landmark_detection(self):
            #atlas_address_path = self.get_atlas_preview_name()
            self.landmark_auto_detect_operation(atlas_address=self.get_atlas_preview_name())

        def section_detection_operation(self):
            self.brainboundcoords, self.tissuemask_fullpath = self.GuiFunctions.funcSectionDetection()
            self.clear_landmark_nodes()

            ''' This function is called when Detect button is clicked'''
            #global brainboundcoords,tissuemask_fullpath
            self.change_status_bar_waiting()
            self.set_status_bar_text('Please Wait: Section Detection is running...')
            print('Section detection operation...')
            # ToDo: technical Dept
            try:
                #brainboundcoords, tissuemask_fullpath = GuiFunctions.funcSectionDetection()
                #print (tissuemask_fullpath)
                self.set_brain_detection_image(self.tissuemask_fullpath)
                self.create_brain_selection_regions(self.brainboundcoords)
                
                self.change_status_bar_default()
                self.set_status_bar_text('Section Detection Done!')
            except Exception as e:
                print(e,"error line 286")
                self.set_status_bar_text('Error!')
            

        def blob_detection_accept(self):
            ''' This function is called when Accept button in NeuroDetection Stage is clicked'''
            
            #blob_detection_result = self.is_blob_detection_perform()
            if self.is_blob_detection_perform():
                status_message = "Neuron detection  is accepted"
                #red_blobs, green_blobs = self.get_blobs()
                self.final_blobs=self.get_blobs()
                #_, colocalized_blobs = calculate_colocalized_blobs(red_blobs, green_blobs)
                self.final_colocalized_blobs_count, self.final_colocalized_blobs = calculate_colocalized_blobs(self.final_blobs)
                self.set_coloc_blobs(self.final_colocalized_blobs)
                self.finish_step(2)
                self.go_to_next_step()
                # Set Atlas PreviewFile  Default
            else:
                status_message = "You should perform Neuro detection first"
            
            if not self.reg_done :self.set_default_atlas_number()
            self.set_status_bar_text(status_message)
            self.clear_landmark_nodes()

        def set_default_atlas_number(self):
            if self.reg_done:return 
            ''' This function determines the default atlas number for next stage when Accept button in NeuroDetection Stage is clicked
            Determines atlas number either by manual selection or deep learning suggestion
            sets the chosen atlas as the atlas diplayed on the screen.
            '''
            #global sectionfolder
            #self.sectionfolder = os.path.join(prepath, self.slidename,f'S{self.brnum}') 
            atlas_folder = os.path.join(atlas_prepath,"labeled_atlases")
            self.set_default_atlas(atlas_folder)
            images_list = get_images_name_in_folder(atlas_folder)
            self.set_file_list(images_list)
            pred_imgpath = os.path.join(self.sectionfolder, 'alevel_eq.png')
            if st_switches.section_classifier_on:
                predicted_atlasnum = m2.predict_atlasnum(pred_imgpath)
                predicted_atlasquads = m2.predict_atlasQs(pred_imgpath)
                self.set_Qs_textbox_parameters(predicted_atlasquads)

            else:
                predicted_atlasnum = 16
                predicted_atlasquads = [0, 0, 0, 0]

            self.set_default_atlas_item_by_name(str(predicted_atlasnum))
            #MainWindow.set_default_atlas_item_by_index(42)
            self.load_preview_atlas()


        def blob_detection_operation(self):
            ''' This function is called when Apply button in NeuroDetection Stage is clicked'''
            
            #MainWindow.set_blob_detection_cells_count(10,10,10)
            if self.step_allowed_to_change(2):
                self.remove_all_blobs()
                blobs_parameters = self.get_all_blob_detection_parameters()
                self.change_status_bar_waiting()
                self.set_status_bar_text("Please wait: Neuron detection is running...")


                
                self.number_of_blobs,self.CO_COUNT,self._,self.blobs_log,self.colocalized_blobs = self.GuiFunctions.funcBlobDetection(self.brnum, blobs_parameters)
                self.add_auto_detect_blobs(self.blobs_log,self.colocalized_blobs) # Displays the detected Neurons with circles in GUI
                #MainWindow.set_blob_detection_image(blob_detection_file_name)
                self.set_blob_detection_cells_count(self.number_of_blobs[0], self.number_of_blobs[1], self.CO_COUNT)
                #self.set_blob_detection_cells_count(RED_COUNT, GREEN_COUNT, CO_COUNT) # Sets the counted Neuron numbers in GUI
                self.blob_detection_perform() # Flags that blob detection is done
                status_message = "Neuron detection is Done!"
                self.change_status_bar_default()  
            else:
                status_message = "The Section detection is required to perform this step"
            self.set_status_bar_text(status_message)

        def blob_flip_operation(self):
            ''' this function is called when flip button in NeroDetection Stage is clicked'''
            
            self.change_status_bar_waiting()
            self.set_status_bar_text("Please Wait")
            self.remove_all_blobs()
            self.set_blob_detection_cells_count(0,0 ,[0]) # Sets the counted Neuron numbers in GUI

            blob_detection_file_name, tissue_lm_detection_filename = self.GuiFunctions.section_flip_operation()

            self.set_blob_detection_image(blob_detection_file_name)
            self.set_tissue_landmark_detection_image(tissue_lm_detection_filename)
            status_message = "Section : {} has been selected".format(self.brnum)
            self.set_status_bar_text(status_message)
            self.change_status_bar_default()  
        def slider_value_change(self):
            self.angle_value=self.slider.value()
            self.angle_value_label.setText(f"Rotation degree : {self.angle_value}")
            arr=self.angle_img
            self.slider.value()
            arr=imgprc.rotate_by_angle(arr, int(self.angle_value))
            cv2.imwrite("angle_image.png",arr)
            #arr2 = np.require(arr, np.uint8, 'C')
            #qImg = QImage(arr2, arr.shape[1], arr.shape[0],QImage.Format_RGB888)
            pixmap = QPixmap("angle_image.png")
            pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)  # Scale to fit
            self.image_label.setPixmap(pixmap)

        def section_select_operation(self):
            self.brnum = self.accept_selection_region()
            self.angle_img=self.GuiFunctions.get_section_images_angle(self.brnum, self.brainboundcoords)
            #print (self.brnum,"brnummmmmmmmmmmmmmmmmmmmmmmmm")
            cv2.imwrite("angle_image.png",self.angle_img)
            qImg = QImage("angle_image.png")#self.angle_img, self.angle_img.shape[1], self.angle_img.shape[0],QImage.Format_RGB888)
            pixmap = QPixmap(qImg)
            if self.angle_img.shape[1]> self.angle_img.shape[0]:
                by_width=1
            else : 
                by_width=0
            ratio=self.angle_img.shape[1]/ self.angle_img.shape[0]
            

            self.rotate_dialog=QtWidgets.QDialog(self)
            #self.rotate_dialog.ret
            self.rotate_dialog.setWindowTitle("Rotate Dialog")
            self.rotate_dialog.setFixedSize(QtCore.QSize(600,600))
            self.image_label = QLabel(self.rotate_dialog)
            h=580//ratio
            if by_width :
                self.image_label.setFixedSize(580,int(580/ratio))
            else :
                self.image_label.setFixedSize(int(580*ratio),580)
            self.image_label.move(10,0)
            self.image_label.setAlignment(Qt.AlignCenter)  # Center image within label
            pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio)  # Scale to fit
            self.image_label.setPixmap(pixmap)
            self.angle_value=0

            hint_label=QtWidgets.QLabel(self.rotate_dialog)
            hint_label.setStyleSheet("QLabel { font-size : 14px; color : white; }")
            hint_label.setText("""Use arrow keys to modify the rotation degree (-45 to +45)\nPress 'Esc' key to cancel operation 'Enter' to accept rotation or just close the window
                               """)
            hint_label.move(10,510)
            self.angle_value_label=QtWidgets.QLabel(self.rotate_dialog)
            self.angle_value_label.setText(f"Rotation degree : {self.angle_value}        ")
            self.angle_value_label.setStyleSheet("QLabel { font-size : 16px; color : white; }")
            self.angle_value_label.move(10,545)
            
            self.slider = QSlider(Qt.Horizontal)
            self.slider.setParent(self.rotate_dialog)
            self.slider.setFixedSize(QtCore.QSize(580,20))
            self.slider.move(10,570)
            self.slider.setValue(0)
            self.slider.setMinimum(-45) 
            self.slider.setMaximum(45)  
            self.slider.setTickPosition(QSlider.TicksBelow)  
            self.slider.setTickInterval(1)
            self.angle_accept_button=QtWidgets.QPushButton(self.rotate_dialog)
            self.angle_accept_button.move(50,545)
            self.angle_accept_button.resize(0,0)
            self.angle_accept_button.setStyleSheet("QPushButton { font-size : 16px; color : white; }")
            self.angle_accept_button.clicked.connect(self.rotate_dialog.close)  
            self.rotate_dialog.closeEvent=self.__section_select_operation
            self.slider.valueChanged.connect(self.slider_value_change)
            self.rotate_dialog.exec_()


        def __section_select_operation(self,Nothing):
            ''' This function is called when Select button is clicked'''
            #self.angle_value_label.close()
            #print ("destroyed")
            os.remove("angle_image.png")
            self.change_status_bar_waiting()
            self.set_status_bar_text("Please Wait")
            self.brnum = self.accept_selection_region()
            print("brnum ",self.brnum)
            dy_switches.set_ardent_reg_done_to_false()
            

            self.blob_detection_file_name, self.tissue_lm_detection_filename = self.GuiFunctions.get_section_images(self.brnum, self.brainboundcoords,self.angle_value)
            


            if self.brnum:
                self.reg_done=False
                #self.set_default_atlas_number()
                self.sectionfolder = os.path.join(prepath, self.slidename,f'S{self.brnum}') 
                self.set_default_atlas_number()
                #self.ui.attlasViewSecond.clear()
                #for item in self.ui.attlasViewSecond.items():
                #
                try:
                    self.remove_pics()
                    #if not self.reg_done : self.set_default_atlas_number()
                    
                except : pass
                self.set_brain_number_in_statusbar(self.brnum)
                self.remove_all_blobs()
                self.change_status_bar_default()
                
                status_message = "Section : {} has been selected".format(self.brnum)
                self.finish_step(1)
                self.go_to_next_step()
                self.set_blob_detection_image(self.blob_detection_file_name)
                self.set_tissue_landmark_detection_image(self.tissue_lm_detection_filename)

            else:
                status_message = "No Sections Selected!"
            self.set_status_bar_text(status_message)
            self.clear_landmark_nodes()


        def loading_slide_operation(self):
            ''' This function is called when Load Slide button is clicked'''
            
            error_msg_1 = QMessageBox()
            error_msg_1.setWindowTitle("Error")
            error_msg_1.setText("The slide name format is not correct.")
            self.set_status_bar_text('Select the Slide Image...')
            self.slide_fullpath, self.slide_fullname = self.open_slide_dialog(slides_prepath)
            if not self.slide_fullpath:
                return
            self.enable_detect_brain()
            self.slidename = self.slide_fullname.split(".")[0]
            slide_prepath=os.path.split(self.slide_fullpath)[0]
            # Save Slide path
            settings.set("slide_prepath", slide_prepath)
            settings.save()
            self.change_status_bar_waiting()
            self.set_status_bar_text('Loading the Slide Image...')
            print('Loading the Slide Image...')
            self.reject_selection_region()
            read_slide_output = self.GuiFunctions.funcReadSlide(self.slide_fullpath)
            if read_slide_output == "em1":
                error_msg_1.exec_()
                return
            else:
                slide_image_name = read_slide_output
            self.set_slide_name_in_statusbar(self.slide_fullname)
            self.set_slide_image(slide_image_name)
            self.change_status_bar_default()
            self.set_status_bar_text('Loading Slide Image Done!')
            self.clear_landmark_nodes()
            #self.section_detection_operation()


        def landmark_auto_detect_operation(self,atlas_address=None):
            ''' This function is called when Apply button in LMDetection Stage is clicked	
            This func was originally intended for automatic LM detection. 	
            '''	
            
            print ("landmark")
            atlasnum = self.get_atlas_number()
            dy_switches.set_atlasnum(atlasnum)
            labeled_atlas_LM_filepath = os.path.join(atlas_prepath,"labeled_atlases", str(atlasnum)+".png")
            unlabeled_atlas_LM_filepath = os.path.join(atlas_prepath,"unlabeled_atlases", str(atlasnum)+".png")
            if st_switches.atlas_type == "Adult":
                general_unlabeled_atlas_filepath =  unlabeled_atlas_LM_filepath.replace("Adult_full_atlases","Adult_atlases")
            elif st_switches.atlas_type == "Rat":
                general_unlabeled_atlas_filepath =  unlabeled_atlas_LM_filepath.replace("Rat_atlases","Rat_atlases")
            else:
                print ("Incorrect atlas type, modify static switches,exiting")
                sys.exit()
                #general_unlabeled_atlas_filepath =  unlabeled_atlas_LM_filepath.replace("P56_full_atlases","P56_atlases")
            self.change_status_bar_waiting()	
            self.set_status_bar_text('Please wait: Atlas Selection is running')	
            tissue_landmark_detection_file_name = os.path.join(self.sectionfolder, 'alevel_eq.png')	
            atlas_landmark_detection_file_name = self.get_atlas_preview_name()
            if atlas_address:	
                if st_switches.section_QL_on:	
                    predicted_atlasquads = self.get_Qs_textbox_parameters()	
                    if self.get_atlas_mode():
                        unlabeled_atlas_LM_filepath = m2.generate_tilted_atlas(predicted_atlasquads, self.sectionfolder)
                        labeled_atlas_LM_filepath = unlabeled_atlas_LM_filepath
            atlas_unlabled_img_path = atlas_landmark_detection_file_name.replace("labeled","unlabeled")
            if st_switches.atlas_type == "Adult":
                general_unlabeled_atlas_filepath =  unlabeled_atlas_LM_filepath.replace("Adult_full_atlases","Adult_atlases")
            
            elif st_switches.atlas_type == "Rat":
                general_unlabeled_atlas_filepath =  unlabeled_atlas_LM_filepath.replace("Rat_atlases","Rat_atlases")
            
                #general_unlabeled_atlas_filepath =  unlabeled_atlas_LM_filepath.replace("P56_full_atlases","P56_atlases")
            resized_tissue_LM_file_name, resized_atlas_LM_file_name = resize_images_for_registration(tissue_landmark_detection_file_name,labeled_atlas_LM_filepath,unlabeled_atlas_LM_filepath, self.sectionfolder, general_unlabeled_atlas_filepath)	
            self.set_tissue_landmark_detection_image(resized_tissue_LM_file_name) #################### must be done in earlier stages or removed from here?	
            self.set_atlas_landmark_detection_image(labeled_atlas_LM_filepath)	
            # This part applies if there is a code for automatic LM detection in the funcLandmarkDetection funtion	
            tissue_auto_landmarks = []	
            atlas_auto_landmarks = []
            list_of_autodetected_nodes_list = 1	
            # displays the automatically detected LMs in the GUI	
            self.add_auto_detect_landmark(list_of_autodetected_nodes_list, tissue_auto_landmarks, atlas_auto_landmarks)	
            # Flags that Auto LM-detection is done	
            self.landmarks_detection_perform() 	
            self.change_status_bar_default()	
            self.set_status_bar_text('Atlas selection done')	
            dy_switches.set_ardent_reg_done_to_false()
            













        









    def save_report_operation():
        MainWindow.save_report_dialog()



    #brnum = 1
    #slide_fullpath = '1'
    app = QtWidgets.QApplication(sys.argv)
    Functions_object=guif.Slide_Operator(prepath)
    MainWindow = Window(Functions_object)
    MainWindow.resizeEvent = resize
    #bind_functions()
    MainWindow.show()
    # geometry = app.desktop().availableGeometry()
    sys.exit(app.exec_())



