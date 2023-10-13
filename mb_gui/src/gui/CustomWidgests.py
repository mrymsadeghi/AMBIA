
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPen, QBrush,QCursor
from PyQt5.QtCore import QPointF,QPoint
from PyQt5.Qt import Qt
from enum import Enum

BLOB_SIZE=30
PEN_SIZE=2
class BlobColor(Enum):
    green=[0,255,0]
    red=[255,0,0]
    yellow=[255,255,0]
    white=[255,255,255]

class PhotoViewer(QtWidgets.QGraphicsView):
    photoClicked = QtCore.pyqtSignal(QtCore.QPoint)

    def __init__(self, parent):
        super(PhotoViewer, self).__init__(parent)
        self._zoom = 0
        self._enable_double_mode=0
        self._enable_zoom=0
        self._blob_mode=0
        self._blob_color=BlobColor.red
        self.point_list=[]
        self.label_list=[]
        self._empty = True
        self._scene = QtWidgets.QGraphicsScene(self)
        self._photo = QtWidgets.QGraphicsPixmapItem()
        self._list_widget=None
        self._scene.addItem(self._photo)
        self.setScene(self._scene)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        # self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(00, 250, 00)))
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

        self.greenBrush = QBrush(Qt.green)
        self.grayBrush = QBrush(Qt.gray)
        self.redBrush = QBrush(Qt.red)
        self.pen = QPen(Qt.red)


        self.auto_detect=0
        self.caption_mode=True
        self.point_size=15
        self.border_color=[00,00,255]#None
        self.fill_color=None
        # self.create_pointer()
        self.setCursor(QCursor(Qt.CrossCursor))
        

    def get_list_widget(self,list_widget):
        self._list_widget=list_widget

        



    def create_pointer(self):
        self._pointer_current_x=800
        self._pointer_current_y=10

        self._pointer = MyPointer(self._pointer_current_x,self._pointer_current_y, 100, 100)
        print('pointer created successfly')
        self.point_selctor = self._scene.addItem( self._pointer)
        self._pointer.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable)

    def hasPhoto(self):
        return not self._empty

    def fitInView(self, scale=True):
        rect = QtCore.QRectF(self._photo.pixmap().rect())
        if not rect.isNull():
            self.setSceneRect(rect)
            if self.hasPhoto():
                unity = self.transform().mapRect(QtCore.QRectF(0, 0, 1, 1))
                self.scale(1 / unity.width(), 1 / unity.height())
                viewrect = self.viewport().rect()
                scenerect = self.transform().mapRect(rect)
                factor = min(viewrect.width() / scenerect.width(),
                             viewrect.height() / scenerect.height())
                self.scale(factor, factor)
            self._zoom = 0

    def setPhoto(self, pixmap=None):
        self._zoom = 0
        if pixmap and not pixmap.isNull():
            self._empty = False
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
            self._photo.setPixmap(pixmap)
        else:
            self._empty = True
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
            self._photo.setPixmap(QtGui.QPixmap())
        self.fitInView()

    # def add_point(self,x,y,is_auto_detect=0,has_caption=True,border_color=[BlobColor.red],fill_color=None):
    def add_point(self,x,y, **kwargs):

        is_auto_detect=kwargs.get('is_auto_detect',0)
        has_caption=kwargs.get('has_caption',True)
        border_color=kwargs.get('border_color',None)
        fill_color=kwargs.get('fill_color',None)
        point_size=kwargs.get('size',self.point_size)
        # point_type=kwargs.get('point_type',"default")
        currentcode=len(self.point_list)+1
        # print(currentcode)
        # self.mylastPoint= Point(x,y,w,h,currentcode,is_auto_detect,border_color,fill_color,self.deletItemByCode,point_type="sa")
        self.mylastPoint= Point(x,y,currentcode,w=point_size,h=point_size,delete_function=self.deletItemByCode,**kwargs)
        self.point_list.append(self.mylastPoint)
        self._scene.addItem(self.mylastPoint)
        if has_caption:
            self.add_caption(x,y,str(currentcode),is_auto_detect)
        xy="(x,y) = ({},{})".format(int(x),int(y))
        point_number=str(len(self.point_list))
        if self._list_widget is not None:
            CustomTreeItem(self._list_widget, point_number,xy,self.deletItemByCode)

    def add_caption(self,x,y,text,is_auto_detect=0):
        
        self.mylast_text =QtWidgets.QGraphicsTextItem(text)
        self.mylast_text.setDefaultTextColor(QtCore.Qt.red)
        if is_auto_detect:
            self.mylast_text.setDefaultTextColor(QtCore.Qt.white)
        # text.setDefaultTextColor(self.text_label_color)
        # text.setPlainText("str(int(w/2))")
        self.mylast_text.setPos(x-7.5, y-12.5)
        self.mylast_text.setZValue(1.5)
        # print("this is test :{}".format(self.mylast_text.toPlainText()))
        self._scene.addItem(self.mylast_text)    
        self.label_list.append(self.mylast_text)


    def get_all_red_blob(self):
        nodes=[(node.x ,node.y)  for node in self.point_list if node.point_type==BlobColor.red]
        return nodes

    def get_all_green_blob(self):
        nodes=[(node.x ,node.y)  for node in self.point_list if node.point_type==BlobColor.green]
        return nodes

    def deleteLastPoint(self):
        print('we want to delet items')
        try:
            self._scene.removeItem(self.mylastPoint)
            self._scene.removeItem(self.mylast_text)
        except:
            pass
    def deletItemByCode(self,item_code):

        item_code=int(item_code)
        try:

            self._scene.removeItem(next((x for x in self.point_list if x.code == item_code), None))
            self._scene.removeItem(next((x for x in self.label_list if x.toPlainText() == str(item_code)), None))
            self.point_list.remove(next((x for x in self.point_list if x.code == item_code), None))
        except:
            self._scene.removeItem(next((x for x in self.label_list if x.toPlainText() == str(item_code)), None))
            pass
    def deleteAllItems(self):

        for item in self.label_list:
            self._scene.removeItem(item)

        for item in self.point_list:
            self._scene.removeItem(item)
        self.point_list=[]
        self.label_list=[]

    def enable_zoom(self):
        self._enable_zoom=1

    def enable_toogle(self):
        self._enable_double_mode=1
    def enable_blob_mode(self):
        self._blob_mode=1
        self.setStyleSheet("border: 1px solid red;")


    def toggleDragMode(self):
        if self.dragMode() == QtWidgets.QGraphicsView.ScrollHandDrag:
            self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        elif not self._photo.pixmap().isNull():
            self.setCursor(QCursor(Qt.CrossCursor))
            self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)

    def toogleState(self):
        return (self.dragMode() == QtWidgets.QGraphicsView.NoDrag)
    def scale_point(self,x,y):
        p0 = QPoint(x, y)
        xx=self.mapFromScene(p0).x()
        yy=self.mapFromScene(p0).y()   
        return xx,yy



    def handleLeftButton(self,event):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if not modifiers==QtCore.Qt.AltModifier:
            xx=int(self.mapToScene(event.pos()).x())
            yy=int(self.mapToScene(event.pos()).y())
            if self._blob_mode:
                blob_color=self._blob_color.value
                blob_name=self._blob_color.name
                # self.add_point(xx,yy,self.auto_detect,False,blob_color,None)
                self.add_point(xx,yy,has_caption=False,border_color=blob_color,
                point_type=self._blob_color,size=BLOB_SIZE)
            else:

                # self.add_point(xx,yy,self.auto_detect,self.caption_mode)
                self.add_point(xx,yy,fill_color=BlobColor.white)

    # def add_point(self,x,y,is_auto_detect=0,has_caption=True,border_color=[BlobColor.red],fill_color=None):
          
            self.photoClicked.emit(self.mapToScene(event.pos()).toPoint())
                    
    # Over Ride Events
    ## OverRide Mouse Event
    #ToDo: change level 
    def mousePressEvent(self, event):
        if self._enable_double_mode:
            if  event.button() == Qt.RightButton:
                self.toggleDragMode()
            elif event.button()==Qt.MidButton:
                self.change_blob_mode()
            if self.dragMode() == QtWidgets.QGraphicsView.NoDrag:
                if event.button() == Qt.LeftButton and  self._photo.isUnderMouse():
                    self.handleLeftButton(event)

        super(PhotoViewer, self).mousePressEvent(event)




    def wheelEvent(self, event):
        if self._enable_zoom:
            if self.hasPhoto():
                if event.angleDelta().y() > 0:
                    factor = 1.25
                    self._zoom += 1
                else:
                    factor = 0.8
                    self._zoom -= 1
                if self._zoom > 0:
                    self.scale(factor, factor)
                elif self._zoom == 0:
                    self.fitInView()
                else:
                    self._zoom = 0

    ## Override  key event
    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_Shift and self._blob_mode:
            self.change_blob_mode()
        super(PhotoViewer, self).keyPressEvent(event)


    def change_blob_mode(self):
        if self._blob_color==BlobColor.red:
            self._blob_color=BlobColor.green
            self.setStyleSheet("border: 1px solid green;")

        else:
            self._blob_color=BlobColor.red
            self.setStyleSheet("border: 1px solid red;")

class MyPointer(QtWidgets.QGraphicsEllipseItem):
    """
    [summary]

    Args:
        QtWidgets ([type]): [description]
    """    
    def __init__(self, *args, **kwargs):
        
        super(MyPointer, self).__init__(*args, **kwargs)
        self.setPen((QtGui.QColor(200,0,0)))
        self.setBrush( QBrush(Qt.red))

class Point(QtWidgets.QGraphicsEllipseItem):
    """
    [summary]

    Args:
        QtWidgets ([type]): [description]
    """    

    def __init__(self,*args, **kwargs):

        self.x=args[0]
        self.y=args[1]
        self.code=args[2]

        w=kwargs.get('w',15)
        h=kwargs.get('h',15)

        self.point_type=kwargs.get('point_type',"default")

        border_color=kwargs.get('border_color',None)
        fill_color=kwargs.get('fill_color',None)

        is_auto_detect=kwargs.get('is_auto_detect',0)
        self.delete_function=kwargs.get('delete_function',None)

        x_center= self.x-(w/2)
        y_center= self.y-(h/2)

        super(Point, self).__init__( x_center,  y_center,w,h)

        if is_auto_detect:
             self.point_color=self.setBrush( QBrush(Qt.magenta))
             return


        if border_color is not None:
            self.point_border=self.setPen((QtGui.QColor(border_color[0],border_color[1],border_color[2])))
        else:
            pen = QPen(QtGui.QColor(200,0,0))
            pen.setWidth=2
            self.point_border=self.setPen(pen)
        
        if fill_color is not None:
            color=fill_color.value
            self.point_color=self.setBrush(QtGui.QColor(color[0],color[1],color[2]))

        else:
            color=self.point_type.value
            pen = QPen(QtGui.QColor(color[0],color[1],color[2]))
            pen.setWidth(PEN_SIZE)
            self.point_border=self.setPen(pen)

    def __str__(self):
        return f"code:{self.code} x:{self.x} y:{self.y} type:{self.point_type}"
    def handleButton(self):
        modifiers = QtWidgets.QApplication.keyboardModifiers()
        if modifiers==QtCore.Qt.AltModifier:
            self.delete_function(self.code)
        elif modifiers==QtCore.Qt.ControlModifier:
            print(self)
        # elif modifiers == (QtCore.Qt.ControlModifier |
        #                    QtCore.Qt.ShiftModifier):
        #     pass
        # else:
        #    pass


    def mousePressEvent(self, event):

        self.handleButton()
        # super(Point, self).mousePressEvent(event)


class CustomTreeItem( QtWidgets.QTreeWidgetItem ):
    '''CustomTreeItem2.py
    Custom QTreeWidgetItem with Widgets
    '''

    def __init__( self, parent,point,data,fun):
        '''
        parent (QTreeWidget) : Item's QTreeWidget parent.
        name   (str)         : Item's name. just an example.
        '''

        ## Init super class ( QtWidgets.QTreeWidgetItem )
        super( CustomTreeItem, self ).__init__( parent )
        self.my_function = fun
        # self.my_args = args
        
        ## Column 0 - Text:
        self.setText( 0, point )

        ## Column 1 - SpinBox:
        self.setText( 1, data )

        ## Column 2 - Button:
        self.button = QtWidgets.QPushButton()
        self.button.setText( "Del")
        # self.button.setFixedWidth(50)
        self.treeWidget().setItemWidget( self, 2, self.button )
        self.treeWidget().scrollToBottom()
        ## Signals
        self.button.clicked.connect(self.deleteButtons)


        
    def deleteButtons(self):
        listItems=self.treeWidget().selectedItems()
        if not listItems: return   
        for item in listItems:
            
            itemIndex=self.treeWidget().indexOfTopLevelItem(item)
            self.my_function(self.text(0))
            self.treeWidget().takeTopLevelItem(itemIndex)
    


 
class BlobDetectionParameter():
        
    def __init__(self,properties):
        self.color=properties['blob_color']
        self.min_blob_thresh=properties['min_blob_thresh']
        self.min_blob_area=properties['min_blob_area']
        self.min_blob_circularity=properties['min_blob_circularity']
        self.min_blob_inertia=properties['min_blob_inertia']

    @staticmethod
    def create_settings(b_color,min_blob_thresh,min_blob_area,min_blob_circularity,min_blob_inertia):
        properties={}
        properties['blob_color']=b_color
        properties['min_blob_thresh']=min_blob_thresh
        properties['min_blob_area']=min_blob_area
        properties['min_blob_circularity']=min_blob_circularity
        properties['min_blob_inertia']=min_blob_inertia
        return properties

    # def __init__(self,min_blob_thresh=2,min_blob_area=5,min_blob_circularity=5,min_blob_inertia=0.06):
    #     self.min_blob_thresh=min_blob_thresh
    #     self.min_blob_area=min_blob_area
    #     self.min_blob_circularity=min_blob_circularity
    #     self.min_blob_inertia=min_blob_inertia



class MySwitch(QtWidgets.QSlider):
    
    def change_style(self):
        if self.value():
            self.setStyleSheet("""
                MySwitch::groove:horizontal {
                    border: 1px solid #00BCD4;
                    height: 15px;
                    background: #00BCD4;
                    margin: 0px;
                    border-radius: 8px;
                }
                MySwitch::handle:horizontal {
                    background: #007ACC;
                    border: 1px solid #007ACC;
                    width: 16px;
                    height: 15px;
                    border-radius: 7px;
                    margin:-2px 0px -2px 0px;
                }
            """)
        else:
            self.setStyleSheet("""
                MySwitch::groove:horizontal {
                    border: 1px solid #ccc;
                    height: 15px;
                    background: #ccc;
                    margin: 0px;
                    border-radius: 8px;
                }
                MySwitch::handle:horizontal {
                    background: #007ACC;
                    border: 1px solid #007ACC;
                    width: 16px;
                    height: 15px;
                    border-radius: 7px;
                    margin:-2px 0px -2px 0px;
                }
            """)

    def __init__(self, parent):
        super(QtWidgets.QSlider, self).__init__(parent)
        self.valueChanged.connect(self.change_style)

    # def  paintEvent(self, e):
    #     print('hi')