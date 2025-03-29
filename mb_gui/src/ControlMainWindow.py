from gui.CustomWidgests import BlobColor,BlobColor_,BlobColor_object,names
from gui.GUI import Ui_MainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import QPoint,Qt
import pathlib
import Path_Finder
from PyQt5.QtWidgets import (QMainWindow, QTextEdit, QGraphicsScene,QFrame,
                             QAction, QFileDialog, QApplication, QPushButton, QRadioButton, QTableWidgetItem)
from PyQt5.QtGui import QPixmap
from gui.CustomWidgests import CustomTreeItem, PhotoViewer, BlobDetectionParameter
import os
import time
from pathlib import Path 
from easysettings import EasySettings
from Switches_Static import coloc_permutation, blob_sizes# blob1_size_red, blob1_size_green, blob1_size_yellow


settings = EasySettings("myconfigfile.conf")
#some settings
"""BLOB_SIZE_GREEN = blob1_size_green
BLOB_SIZE_RED = blob1_size_red #Size of circles in the blob detection step displayed on the
BLOB_SIZE_YELLOW = blob1_size_yellow"""


# 
GREEN_BLOB_NUM_SIGMA=5
RED_BLOB_NUM_SIGMA=5



class VLine(QFrame):
    def __init__(self):
        super(VLine, self).__init__()
        self.setFrameShape(self.VLine|self.Sunken)
        self.setStyleSheet("color:#fff")


class ControlMainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(ControlMainWindow, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.set_custom_status_bar()
        
        self.set_defaults()
        self.binding_elements()
        self.set_Qs_textbox_parameters()



    def set_custom_status_bar(self):
        self.ui.lblSlideName = QtWidgets.QLabel("Slide Name: -") 
        self.ui.lblBrainNumber = QtWidgets.QLabel("Section Number: -") 
        self.ui.progressBar =QtWidgets.QProgressBar()
        
        self.ui.lblSlideName.setStyleSheet("background-color: rgba(0, 0, 0, 0); color:#fff;font-size:11px; padding-right:10px") 
        self.ui.lblBrainNumber.setStyleSheet("background-color: rgba(0, 0, 0, 0); color:#fff;font-size:11px; padding-right:20") 
        
        self.ui.progressBar.setStyleSheet("""
        QProgressBar {
        max-height: 10px;
        border-radius: 6px;
        padding-right:200px;
        }
        QProgressBar::chunk {
        border-radius: 6px;
        background-color: #fff;
        }
        """
        )
        self.ui.progressBar.setTextVisible(False)
        self.ui.progressBar.setVisible(False)
        self.ui.statusbar.addPermanentWidget(self.ui.progressBar)
        # self.ui.statusbar.addPermanentWidget(verticalSpacer) 
        self.ui.statusbar.addPermanentWidget(self.ui.lblSlideName) 
        self.ui.statusbar.addPermanentWidget(self.ui.lblBrainNumber) 

        # # setting up the border 

        
    def set_defaults(self):
        self.double_click_timer=0
        #from Main import prepath, atlas_prepath
        prepath=Path_Finder.return_prepath()
        atlas_prepath=Path_Finder.return_atlas_path()
        #prepath = pathlib.Path(__file__).parents[2].absolute()
        atlas_prepath = os.path.join(atlas_prepath, "labeled_atlases")
        print("atlas prepath", atlas_prepath)
        self.folder_path=atlas_prepath
        # self.active_step_icon=self.change_step_icon(0)
        self.states_state = [0, 0, 0, 0, 0]
        self.selection_items = []
        self.selected_brain = 0
        self.accepted_selection = 0
        self.blob_detection = 0

        self.landmarks_detection = 0
        self.active_step = 1
        self.atlas_preview_folder = None
        self.atlas_preview_full_name = None
        self.set_default_step()
        # Test
        # self.ui.viewInput.enable_toogle()
        # self.ui.viewInput.enable_zoom()
        
        self.ui.viewBlobDetection.enable_zoom()
        self.ui.viewBlobDetection.enable_toogle()
        self.ui.viewBlobDetection.caption_mode=False
        self.ui.viewBlobDetection.enable_blob_mode()
        
        self.ui.attlasViewFrist.enable_zoom()
        self.ui.attlasViewFrist.enable_toogle()
        

        self.ui.attlasViewSecond.enable_zoom()
        self.ui.attlasViewSecond.enable_toogle()

        self.ui.viewRegistration.enable_zoom()


        self.setWindowIcon(QtGui.QIcon(':/newPrefix/Icons/brain.svg'))

        #load Default value 
        self.load_default_blob_parameters()
        
    def load_default_blob_parameters(self):
        try:
            self.ui.cmbRedTypes.setCurrentText(settings.get("c0_blob_type"))
            self.ui.txtRedMinSize.setValue(settings.get("c0_blob_min_size"))
            self.ui.txtRedThresh.setValue(settings.get("c0_blob_rabies_thresh"))
            self.ui.txtRedCorrelation.setValue(settings.get("c0_blob_correlation"))
            self.ui.txtRedStride.setValue(settings.get("c0_blob_stride"))
            self.ui.txtRedMinSigma.setValue(settings.get("c0_blob_min_sigma"))
            self.ui.txtRedMaxSigma.setValue(settings.get("c0_blob_max_sigma"))
            # self.ui.txtRedNumSigma.setValue(settings.get("red_blob_num_sigma"))
            self.ui.txtRedNumSigma.setValue(settings.get("c0_blob_cfos_thresh1"))
            self.ui.txtRedThreshold2.setValue(settings.get("c0_blob_cfos_thresh2"))

        
           
            self.ui.cmbGreenTypes.setCurrentText(settings.get("c1_blob_type"))
            self.ui.txtGreenMinSize.setValue(settings.get("c1_blob_min_size"))
            self.ui.txtGreenThresh.setValue(settings.get("c1_blob_rabies_thresh"))
            self.ui.txtGreenCorrelation.setValue(settings.get("c1_blob_correlation"))
            self.ui.txtGreenStride.setValue(settings.get("c1_blob_stride"))
            self.ui.txtGreenMinSigma.setValue(settings.get("c1_blob_min_sigma"))
            self.ui.txtGreenMaxSigma.setValue(settings.get("c1_blob_max_sigma"))
            # self.ui.txtGreenNumSigma.setValue(settings.get("green_blob_num_sigma"))
            self.ui.txtGreenNumSigma.setValue(settings.get("c1_blob_cfos_thresh1"))
            self.ui.txtGreenThreshold2.setValue(settings.get("c1_blob_cfos_thresh2"))

           
        except:
            pass

        self.cmb_cell_type_change()
        
    def binding_elements(self):
        self.binding_step_buttons()
        self.bind_landmarks_lists()
        self.binding_action_buttons()
        self.bind_blob_types()


    def binding_step_buttons(self):
        self.ui.btnBrainDetection.clicked.connect(self.brain_detection_step_active)
        self.ui.btnBlobDetection.clicked.connect(self.blob_detection_step_active)
        self.ui.btnLandmarks.clicked.connect(self.landmarks_detection_step_active)
        self.ui.btnRegistraion.clicked.connect(self.registration_detection_step_active)
        self.ui.btnAnalysis.clicked.connect(self.final_naylize_detection_step_active)
        self.ui.btnPreStep.clicked.connect(self.go_to_pre_step)
        self.ui.btnNextStep.clicked.connect(self.go_to_next_step)


    def binding_action_buttons(self):
    
        self.ui.cmbSelectImage.activated.connect(self.load_preview_atlas)
        self.ui.cmbRedTypes.activated.connect(self.cmb_cell_type_change)
        self.ui.cmbGreenTypes.activated.connect(self.cmb_cell_type_change)
        self.ui.tglAtlasModeSelector.valueChanged.connect(self.change_attlas_state)


    def bind_blob_types(self):
        self.initilize_blob_type()



    def change_attlas_state(self):
        tilted_state=self.get_atlas_mode()
        self.ui.stkAtlasState.setCurrentIndex(tilted_state)

    def enable_detect_brain(self):
        self.ui.btnSelectBrain.setEnabled(True)
        self.ui.btnBrainDetecting.setEnabled(True)

        
    def cmb_cell_type_change(self):
        green_type=self.ui.cmbGreenTypes.currentText()
        red_type=self.ui.cmbRedTypes.currentText()
        
        
        if green_type=="Rabies":
            self.ui.greenPropertySetting.setCurrentIndex(0)
        elif green_type=="MoG":
            self.ui.greenPropertySetting.setCurrentIndex(1)
        elif green_type=="cFos":
            self.ui.greenPropertySetting.setCurrentIndex(2)

        if red_type=="Rabies":
            self.ui.redPropertySetting.setCurrentIndex(0)
        elif red_type=="MoG":
            self.ui.redPropertySetting.setCurrentIndex(1)            
        elif red_type=="cFos":
            self.ui.redPropertySetting.setCurrentIndex(2)            
        
        return green_type,red_type
    def cmb_color_change(self):
        blob_name = self.get_current_blob_name()
        # self.show_settings_in_table()
        current_settings = self.blob_settings[blob_name]
        self.set_blob_parameters_ui(current_settings)


    def bind_change_opacity_operation(self, fun):
        self.slider_change_opacity_operation = fun
        self.ui.sliderOpacity.valueChanged.connect(self.slider_change_opacity_operation)
    # def bind_browse_atlas_operation(self, fun):
    #     self.btn_browse_atlas_operation = fun
    #     self.ui.btnBrowseImage.clicked.connect(self.btn_browse_atlas_operation)

    def bind_loading_slide_operation(self, fun):
        self.btn_loading_slide_operaion = fun
        self.ui.btnLoadSlide.clicked.connect(self.btn_loading_slide_operaion)

    def bind_brain_detecting_operation(self, fun):
        self.btn_detection_brain_operaion = fun
        self.ui.btnBrainDetecting.clicked.connect(self.btn_detection_brain_operaion)

    def bind_select_brain_operation(self, fun):
        self.btn_accept_selection_operaion = fun
        self.ui.btnSelectBrain.clicked.connect(self.btn_accept_selection_operaion)

    def bind_blob_detection_parameter_set_operation(self, fun):
        self.btn_set_settings = fun
        self.ui.btnSetSettings.clicked.connect(self.btn_set_settings)

    def bind_blob_flip_operation(self, fun):
        self.btn_blob_flip_operation= fun
        self.ui.btnFlip.clicked.connect(self.btn_blob_flip_operation)
        
    def bind_blob_detection_operation(self, fun):
        self.btn_blob_detection_operaion = fun
        # self.ui.btnBlobDetecting.clicked.connect(self.btn_blob_detection_operaion)
        self.ui.btnApply.clicked.connect(self.btn_blob_detection_operaion)

    def bind_blob_detection_accept_operation(self, fun):
        self.btn_blob_detection_accept_operaion = fun
        self.ui.btnAcceptBlobDetection.clicked.connect(self.btn_blob_detection_accept_operaion)

    def bind_landmark_auto_detect_opration(self, fun):
        self.btn_landmark_auto_detect_operaion = fun
        self.ui.btnAutoDetect.clicked.connect(self.btn_landmark_auto_detect_operaion)


    def bind_registration_accept_opration(self, fun):
        self.btn_landmark_detection_accept_operaion = fun
        self.ui.btnAcceptNodes.clicked.connect(self.btn_landmark_detection_accept_operaion)

    def bind_registration_preview_accept_opration(self, fun):
        self.btn_registraion_accept_operaion = fun
        self.ui.btnAcceptRegistration.clicked.connect(self.btn_registraion_accept_operaion)

    def bind_save_report_opration(self, fun):
        self.btn_save_report_operaion = fun
        self.ui.btnSave.clicked.connect(self.btn_save_report_operaion)

    def bind_landmarks_lists(self):
        self.initialize_list(self.ui.attlasListFrist)
        self.initialize_list(self.ui.attlasListSecond)

        self.ui.attlasViewFrist.get_list_widget(self.ui.attlasListFrist)
        self.ui.attlasViewSecond.get_list_widget(self.ui.attlasListSecond)

        self.ui.attlasListFrist.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.attlasListFrist.customContextMenuRequested.connect(self.open_tissue_context_menu)

        self.ui.attlasListSecond.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.ui.attlasListSecond.customContextMenuRequested.connect(self.open_atlas_context_menu)

    def open_tissue_context_menu(self):
        menu = QtWidgets.QMenu()
        editMenu = QtWidgets.QAction("Remove All", self)
        editMenu.triggered.connect(self.remove_all_tissue_points)
        menu.addAction(editMenu)
        menu.exec_(QtGui.QCursor.pos())

    def open_atlas_context_menu(self):
        menu = QtWidgets.QMenu()
        editMenu = QtWidgets.QAction("Remove All", self)
        editMenu.triggered.connect(self.remove_all_atlas_points)
        menu.addAction(editMenu)
        menu.exec_(QtGui.QCursor.pos())

    def initialize_list(self, treeW):
        HEADERS = ("Point number", "data", "action")
        treeW.setColumnCount(len(HEADERS))
        treeW.setHeaderLabels(HEADERS)
        treeW.setColumnWidth(0, 100)
        treeW.setColumnWidth(1, 200)

    def remove_all_atlas_points(self):
        self.ui.attlasViewSecond.deleteAllItems()
        self.ui.attlasListSecond.clear()

    def remove_all_tissue_points(self):
        self.ui.attlasViewFrist.deleteAllItems()
        self.ui.attlasListFrist.clear()
    
    def remove_all_blobs(self):
        self.ui.viewBlobDetection.deleteAllItems()
        self.ui.viewBlobDetection.point_list.clear()


    def initilize_blob_type(self):
        values=['Rabies','MoG','cFos']
        self.ui.cmbRedTypes.addItems(values)
        self.ui.cmbGreenTypes.addItems(values)

    def load_preview_atlas(self):
        extention = ".png"
        file_name = f"{str(self.ui.cmbSelectImage.currentText())}{extention}"
        # folder_address = self.ui.txtbrowse.text()
        full_address = os.path.join(self.folder_path, file_name)
        self.atlas_preview_full_name = full_address
        self.ui.atlasPreview.setPhoto(QtGui.QPixmap(full_address))
        
    def load_preview_as_atlas(self):
        full_address = self.atlas_preview_full_name
        
        
        self.ui.attlasViewSecond.setPhoto(QtGui.QPixmap(full_address))

    ### Steps
    def brain_detection_step_active(self):
        self.active_step = 1

        self.ui.mainTab.setCurrentIndex(0)
        self.ui.toolsStack.setCurrentIndex(0)

        self.change_all_steps_icon()
        step_icon = self.get_step_state_icon(1)
        self.ui.btnBrainDetection.setIcon(step_icon)

    def blob_detection_step_active(self):
        self.active_step = 2

        self.ui.mainTab.setCurrentIndex(1)
        self.ui.toolsStack.setCurrentIndex(1)

        self.change_all_steps_icon()
        step_icon = self.get_step_state_icon(1)
        self.ui.btnBlobDetection.setIcon(step_icon)

    def landmarks_detection_step_active(self):
        self.active_step = 3

        self.ui.mainTab.setCurrentIndex(2)
        self.ui.toolsStack.setCurrentIndex(2)

        self.change_all_steps_icon()
        step_icon = self.get_step_state_icon(1)
        self.ui.btnLandmarks.setIcon(step_icon)

        self.ui.attlasViewFrist.fitInView()
        self.ui.attlasViewSecond.fitInView()


    def registration_detection_step_active(self):
        self.active_step = 4
        self.ui.mainTab.setCurrentIndex(4)
        self.ui.mainTab.setCurrentIndex(3)
        self.ui.toolsStack.setCurrentIndex(3)

        self.change_all_steps_icon()
        step_icon = self.get_step_state_icon(1)
        self.ui.btnRegistraion.setIcon(step_icon)

    def final_naylize_detection_step_active(self):
        self.active_step = 5

        self.ui.mainTab.setCurrentIndex(4)
        self.ui.toolsStack.setCurrentIndex(4)

        self.change_all_steps_icon()
        step_icon = self.get_step_state_icon(1)
        self.ui.btnAnalysis.setIcon(step_icon)

    def set_default_step(self):
        self.brain_detection_step_active()

    def change_active_step(self, step_number):
        if step_number == 1:
            self.brain_detection_step_active()
        elif step_number == 2:
            self.blob_detection_step_active()
        elif step_number == 3:
            self.landmarks_detection_step_active()
        elif step_number == 4:
            self.registration_detection_step_active()
        elif step_number == 5:
            self.final_naylize_detection_step_active()

    def go_to_pre_step(self):
        if self.active_step > 1:
            self.active_step = self.active_step - 1
        self.change_active_step(self.active_step)

    def go_to_next_step(self):
        if self.active_step < 5:
            self.active_step = self.active_step + 1
        self.change_active_step(self.active_step)

    ## manage icons
    def get_step_state_icon(self, icon_state):
        this_icon = QtGui.QIcon()
        if icon_state == 0:
            this_icon.addPixmap(QtGui.QPixmap(":/newPrefix/Icons/wait.svg"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        elif icon_state == 1:
            this_icon.addPixmap(QtGui.QPixmap(":/newPrefix/Icons/progress-blue.svg"), QtGui.QIcon.Normal,
                                QtGui.QIcon.Off)
        elif icon_state == 2:
            this_icon.addPixmap(QtGui.QPixmap(":/newPrefix/Icons/check-lighblue.svg"), QtGui.QIcon.Normal,
                                QtGui.QIcon.Off)

        return this_icon

    def change_active_step_icon(self, icon_state):
        self.active_step_icon = self.get_step_state_icon(icon_state)

    def change_all_steps_icon(self):
        step_icon = self.get_step_state_icon(1)
        step_index = 1
        for this_step in [self.ui.btnBrainDetection,
                          self.ui.btnBlobDetection,
                          self.ui.btnLandmarks,
                          self.ui.btnRegistraion,
                          self.ui.btnAnalysis]:
            step_state = self.get_step_state(step_index)
            step_icon = self.get_step_state_icon(step_state)
            this_step.setIcon(step_icon)
            step_index = step_index + 1

    ## Manage Steps
    def set_brain_number_and_slide_name(self,slide_name,brain_number):
        
        self.set_slide_name_in_statusbar(slide_name)
        self.set_brain_number_in_statusbar(brain_number)

    def set_slide_name_in_statusbar(self,slide_name):
        slide_caption=f"Slide Name: {slide_name}"
        self.ui.lblSlideName.setText(slide_caption)
    
    def set_brain_number_in_statusbar(self,brain_number):
        brain_caption=f"Section Number: {brain_number}"
        self.ui.lblBrainNumber.setText(brain_caption)

    def set_atlas_preview_name(self, full_address):
        self.atlas_preview_full_name = full_address

    def get_atlas_preview_name(self):
        return self.atlas_preview_full_name
        
    def get_atlas_number(self):
        atlasfilename= os.path.split(self.atlas_preview_full_name)[-1]
        atlasnum = atlasfilename.split(".")[0]
        return int(atlasnum)

    def get_step_state(self, step):
        return self.states_state[step - 1]
    
    def get_registration_mode(self):
        return bool(self.ui.tglRegistrationMode.value())
        
    def get_atlas_mode(self):                           ############ True or False for Tilted vs Standard atlas
        return bool(self.ui.tglAtlasModeSelector.value())

    def set_step_state(self, step, state):
        self.states_state[step - 1] = state

    def set_all_steps_state(self, states):
        self.states_state = states

    def step_allowed_to_change(self, step):
        if step == 1:
            return (self.accepted_selection > 0)
        elif step > 1:
            last_step_state = self.get_step_state(step - 1)
            return (last_step_state == 2)

    def finish_step(self, step):
        if self.step_allowed_to_change(step):
            self.set_step_state(step, 2)
            for next_step in range(step + 1, 6):
                self.set_step_state(next_step, 0)

    ## operations
    def set_status_bar_text(self, my_text):
        self.ui.statusbar.showMessage(my_text)
        QApplication.processEvents()
        QApplication.processEvents()
        QApplication.processEvents()
    
    def set_progress_bar(self,value):
        if 99>value>0:
            self.ui.progressBar.setVisible(True)
            self.ui.progressBar.setValue(value)
        else:
            self.ui.progressBar.setVisible(False)
            self.ui.progressBar.setValue(0)
            
        
    def change_status_bar_waiting(self):
        self.ui.statusbar.setStyleSheet("background-color: #ff6c00; ")

    def change_status_bar_default(self):
        self.ui.statusbar.setStyleSheet("background-color: #007Acc; ")

    def open_slide_dialog(self, slides_prepath):
        file_dialog = QFileDialog.getOpenFileName(None, "Open File", slides_prepath,"Format(*.mrxs *.czi)")
        if file_dialog:
            file_name = file_dialog[0]
            return file_name,Path(file_name).name
        return False,False   
    def save_report_dialog(self):
        file_dialog = QFileDialog.getSaveFileName(None, "Open File", QtCore.QDir.currentPath())
        file_name = file_dialog[0]
        return file_name

    def get_atlas_folder(self, atlas_prepath):

        folder_path = str(QFileDialog.getExistingDirectory(self, "Select Directory", atlas_prepath,
                                                           QFileDialog.ShowDirsOnly))
        self.atlas_preview_folder = folder_path
        # self.ui.txtbrowse.setText(folder_path)
        # print(folder_path)
        return folder_path
        # load images list


    def resize_photo_viewers(self):
        self.ui.viewInput.fitInView()
        self.ui.viewMask.fitInView()
        self.ui.viewBlobDetection.fitInView()
        self.ui.attlasViewFrist.fitInView()
        self.ui.attlasViewSecond.fitInView()
        self.ui.viewRegistration.fitInView()
        self.ui.viewReport.fitInView()
    def get_list_index_by_name(self,str):
               
        all_items = [self.ui.cmbSelectImage.itemText(i) for i in range(self.ui.cmbSelectImage.count())]
        index=all_items.index(str)
        return index
    def set_default_atlas_item_by_index(self,index=0):


        self.ui.cmbSelectImage.setCurrentIndex(index)
        self.ui.cmbSelectImage.setItemData(index,QtGui.QColor(0,122,204), Qt.ForegroundRole)


    def set_default_atlas_item_by_name(self,str):
        item_index=self.get_list_index_by_name(str)
        self.set_default_atlas_item_by_index(item_index)



    def set_default_atlas(self, folder_path):
        self.atlas_preview_folder = folder_path


    def set_file_list(self, file_list):
        self.ui.cmbSelectImage.addItems(file_list)

    def set_preview_atlas_preview(self, file_address):
        self.ui.atlasPreview.setPhoto(QtGui.QPixmap(file_address))

    def set_slide_image(self, file_address):
        self.ui.viewInput.setPhoto(QtGui.QPixmap(file_address))

    def set_brain_detection_image(self, file_address):
        self.ui.viewMask.setPhoto(QtGui.QPixmap(file_address))

    def set_blob_detection_image(self, file_address):
        self.ui.viewBlobDetection.setPhoto(QtGui.QPixmap(file_address))

    # def set_blob_detection_parameters_table(self):
    #     self.ui.tblParameters.setItem('1', '3', 4)
    def set_blob_detection_cells_count(self,red_count,green_count,co_count):
        self.ui.lblRedBlobCount.setText(str(red_count))
        self.ui.lblGreenBlobCount.setText(str(green_count))
        c=""
        for j in co_count:
            c+=str(j)+","
        self.ui.lblCoBlobCount.setText(c)#str(co_count))


    def set_tissue_landmark_detection_image(self, file_address):
        self.ui.attlasViewFrist.fitInView()
        self.ui.attlasViewFrist.setPhoto(QtGui.QPixmap(file_address))
         
    def set_Qs_textbox_parameters(self,parameters=[67,70,67,70]):
        self.ui.txtQ1.setValue(parameters[0])
        self.ui.txtQ2.setValue(parameters[1])
        self.ui.txtQ3.setValue(parameters[2])
        self.ui.txtQ4.setValue(parameters[3])

      
    def set_atlas_landmark_detection_image(self, file_address):
        self.img=QtGui.QPixmap(file_address)
        self.ui.attlasViewSecond.setPhoto(QtGui.QPixmap(file_address))
        self.ui.attlasViewSecond.fitInView()

    def remove_pics(self):
        self.ui.attlasViewSecond.setPhoto(None)
        self.set_registration_image(None)

    def set_registration_image(self, file_address):
        self.ui.viewRegistration.setPhoto(QtGui.QPixmap(file_address))

    def set_report_image(self, file_address):
        self.ui.viewReport.setPhoto(QtGui.QPixmap(file_address))

    def set_report_text(self, content):
        self.ui.txtReport.setText(content)
    

    def add_selection_item(self, x=100, y=100, w=100, h=100, caption='item detected'):
        x1, y1 = self.ui.viewMask.scale_point(x, y)
        x2, y2 = self.ui.viewMask.scale_point(x + w, y + h)
        w_scale = x2 - x1
        h_scale = y2 - y1
        selection_item = QtWidgets.QRadioButton(self.ui.viewMask)
        selection_item.setGeometry(QtCore.QRect(x1, y1, w_scale, h_scale))
        selection_item.setText(caption)
        selection_item.toggled.connect(lambda: self.brain_select_action(selection_item))
        #selection_item.doubleClicked.connect(lambda: self.handle_selection_double_click(selection_item))
        selection_item.show()

        # selection_item.geometry().height
        self.selection_items.append(selection_item)




    def brain_select_action(self, item):
        if self.double_click_timer==0:
            
            if item.isChecked() == True:
                self.selected_brain = int(item.text())
                selection_message = "Section {} is selected".format(self.selected_brain)
                self.set_status_bar_text(selection_message)
            self.double_click_timer=time.time()*1000
            print (self.double_click_timer)
        else :
            if time.time()*1000 - self.double_click_timer<301:
                self.double_click_timer=0
                self.accept_selection_region()
            else :
                self.double_click_timer=0
        #######to be continued
            

    def create_brain_selection_regions(self, brainboundcoords):
        # Hide old selections item
        for item in self.selection_items:
            item.hide()
        self.set_defaults()
        # print(self.selection_items)
        # YOU_SHOULD_MODIFY THIS SECTION AS TEMPLATE
        for brnum in range(brainboundcoords.shape[0]):
            xb = brainboundcoords[brnum][0]
            yb = brainboundcoords[brnum][1]
            wb = brainboundcoords[brnum][2]
            hb = brainboundcoords[brnum][3]
            self.add_selection_item(x=xb, y=yb, w=wb, h=hb, caption=str(brnum + 1))

    def accept_selection_region(self):
        if self.selected_brain:
            self.accepted_selection = self.selected_brain
        return self.accepted_selection

    def reject_selection_region(self):
        self.selected_brain = 0
        self.accepted_selection = 0

    def return_selection_region_info(self):
        # YOU_SHOULD_MODIFY THIS SECTION AS TEMPLATE
        return self.accepted_selection



    def get_all_blob_detection_parameters(self):
        """
        Return A Dictionay of Blob parameters

        Returns:
            [red_blob_type]: [description]
            [red_blob_thresh]: [description]
            [green_blob_type]: [description]
            [green_blob_thresh]: [description]

        """
        blob_detection_parameters = {}

        ## Red Blob parameters
        blob_detection_parameters['c0_blob_type'] = self.ui.cmbRedTypes.currentText()

        blob_detection_parameters['c0_blob_min_size'] = self.ui.txtRedMinSize.value()
        blob_detection_parameters['c0_blob_rabies_thresh'] = self.ui.txtRedThresh.value()

        blob_detection_parameters['c0_blob_correlation'] = self.ui.txtRedCorrelation.value()
        blob_detection_parameters['c0_blob_stride'] = self.ui.txtRedStride.value()

        blob_detection_parameters['c0_blob_min_sigma'] = self.ui.txtRedMinSigma.value()
        blob_detection_parameters['c0_blob_max_sigma'] = self.ui.txtRedMaxSigma.value()
        
        # Change hard code
        blob_detection_parameters['c0_blob_num_sigma'] = RED_BLOB_NUM_SIGMA
        # blob_detection_parameters['red_blob_num_sigma'] = self.ui.txtRedNumSigma.value()

        blob_detection_parameters['c0_blob_cfos_thresh1'] = self.ui.txtRedNumSigma.value()
        blob_detection_parameters['c0_blob_cfos_thresh2'] = self.ui.txtRedThreshold2.value()
        

        ## Green Blob parameters
        blob_detection_parameters['c1_blob_type'] = self.ui.cmbGreenTypes.currentText()

        blob_detection_parameters['c1_blob_min_size'] = self.ui.txtGreenMinSize.value()
        blob_detection_parameters['c1_blob_rabies_thresh'] = self.ui.txtGreenThresh.value()

        blob_detection_parameters['c1_blob_correlation'] = self.ui.txtGreenCorrelation.value()
        blob_detection_parameters['c1_blob_stride'] = self.ui.txtGreenStride.value()

        blob_detection_parameters['c1_blob_min_sigma'] = self.ui.txtGreenMinSigma.value()
        blob_detection_parameters['c1_blob_max_sigma'] = self.ui.txtGreenMaxSigma.value()

        # Change hard code
        blob_detection_parameters['c1_blob_num_sigma'] = GREEN_BLOB_NUM_SIGMA
        # blob_detection_parameters['green_blob_num_sigma'] = self.ui.txtGreenNumSigma.value()

        blob_detection_parameters['c1_blob_cfos_thresh1'] = self.ui.txtGreenNumSigma.value()
        blob_detection_parameters['c1_blob_cfos_thresh2'] = self.ui.txtGreenThreshold2.value()
        
      # Save Settings
        
        settings.set("c0_blob_type", blob_detection_parameters['c0_blob_type'])
        
        settings.set("c0_blob_min_size", blob_detection_parameters['c0_blob_min_size'])
        settings.set("c0_blob_rabies_thresh", blob_detection_parameters['c0_blob_rabies_thresh'])

        settings.set("c0_blob_correlation", blob_detection_parameters['c0_blob_correlation'])
        settings.set("c0_blob_stride", blob_detection_parameters['c0_blob_stride'])
        settings.set("c0_blob_min_sigma", blob_detection_parameters['c0_blob_min_sigma'])
        settings.set("c0_blob_max_sigma", blob_detection_parameters['c0_blob_max_sigma'])
        settings.set("c0_blob_num_sigma", blob_detection_parameters['c0_blob_num_sigma'])
        settings.set("c0_blob_cfos_thresh1", blob_detection_parameters['c0_blob_cfos_thresh1'])
        settings.set("c0_blob_cfos_thresh2", blob_detection_parameters['c0_blob_cfos_thresh2'])


        
        settings.set("c1_blob_type", blob_detection_parameters['c1_blob_type'])
        settings.set("c1_blob_min_size", blob_detection_parameters['c1_blob_min_size'])
        settings.set("c1_blob_rabies_thresh", blob_detection_parameters['c1_blob_rabies_thresh'])
        settings.set("c1_blob_correlation", blob_detection_parameters['c1_blob_correlation'])
        settings.set("c1_blob_stride", blob_detection_parameters['c1_blob_stride'])
        settings.set("c1_blob_min_sigma", blob_detection_parameters['c1_blob_min_sigma'])
        settings.set("c1_blob_max_sigma", blob_detection_parameters['c1_blob_max_sigma'])
        settings.set("c1_blob_num_sigma", blob_detection_parameters['c1_blob_num_sigma'])
        settings.set("c1_blob_cfos_thresh1", blob_detection_parameters['c1_blob_cfos_thresh1'])
        settings.set("c1_blob_cfos_thresh2", blob_detection_parameters['c1_blob_cfos_thresh2'])
        settings.save()

        return blob_detection_parameters

    def blob_detection_perform(self):
        self.blob_detection = 1

    def is_blob_detection_perform(self):
        return self.blob_detection

    # should be change
    def add_auto_detect_landmark(self, list_of_nodes, tissue_auto_landmarks, atlas_auto_landmarks):
        self.clear_landmark_nodes()
        for tp in range(len(tissue_auto_landmarks)):
            tpoint = tissue_auto_landmarks[tp]
            self.ui.attlasViewFrist.add_point(tpoint[1], tpoint[0], is_auto_detect=1)
        for ta in range(len(atlas_auto_landmarks)):
            apoint = atlas_auto_landmarks[ta]
            self.ui.attlasViewSecond.add_point(apoint[1], apoint[0], is_auto_detect=1)

    def clear_landmark_nodes(self):
        self.ui.attlasViewFrist.deleteAllItems()
        self.ui.attlasListFrist.clear()

        self.ui.attlasViewSecond.deleteAllItems()
        self.ui.attlasListSecond.clear()
        
    def get_opacity_value(self):
        return self.ui.sliderOpacity.value()

    def get_landmark_nodes(self):
        return self.ui.attlasViewFrist.point_list

    def get_atlas_landmark_nodes(self):
        return self.ui.attlasViewSecond.point_list

    def get_Qs_textbox_parameters(self):
        return [
            self.ui.txtQ1.value(),
            self.ui.txtQ2.value(),
            self.ui.txtQ3.value(),
            self.ui.txtQ4.value(),
        ]

    def landmarks_detection_perform(self):
        self.landmarks_detection = 1

    def is_landmarks_detection_perform(self):
        return self.landmarks_detection

    def set_coloc_blobs(self, coloc_blobs_coords):
        self.coloc_blobs = coloc_blobs_coords
        

    def get_coloc_blobs(self):
        return self.coloc_blobs

    def get_blobs(self):
        return self.ui.viewBlobDetection.return_all_blobs()


    def add_auto_detect_blobs(self,list_of_blobs,list_of_co_blobs_permute):
        for i in range(len(list_of_blobs)):
            print (f"channel {i} color : {names[i]}")
            for j in list_of_blobs[i]:
                self.ui.viewBlobDetection.add_point(
                j[1], j[0],
                point_type=BlobColor_object[i],
                has_caption=False,
                size=blob_sizes[i]
                )
        index=4
        
        for j,nodes in enumerate(list_of_co_blobs_permute):
            print (f"permute {j} {coloc_permutation[j]} color : {names[index]}")
            for yellow_node in nodes:
                self.ui.viewBlobDetection.add_point(
                    yellow_node[1], yellow_node[0],
                    point_type=BlobColor_object[index],
                    has_caption=False,
                    size=blob_sizes[-1])
            index+=1
