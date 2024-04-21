 # Stracture

- create your Qt app
- create instanse of your GUI class
- show your GUI and execute your app  
`app = QtWidgets.QApplication(sys.argv)`  
`MainWindow =ControlMainWindow()`  
`MainWindow.show()`  
`sys.exit(app.exec_())`  



## Convert Resouse files and UI files to python
### Resoucre file :
- `pyrcc5 resource.qrc -o resource_rc.py`

### UI file :
- ` pyuic5 main.ui>main.py`
or  
- `pyuic5 -x fileName -o outputfile.py`


## Modify  Gui items  
Replace QGraphoicView objects with PhotoViewer(our custom photo Viewer)
for example:  
- **first:**
- `self.attlasViewFrist = QtWidgets.QGraphicsView(self.tabBrainDetection)`
- **modify to:** 
- `self.attlasViewFrist = PhotoViewer(self.tabBrainDetection)`  

### add list structre
```
self.initilize_list(self.attlasListFrist)
    self.attlasViewFrist.get_list_widget(self.attlasListFrist)
    def initilize_list(self,treeW):
        HEADERS = ( "Point number", "data", "action" )
        treeW.setColumnCount( len(HEADERS) )
        treeW.setHeaderLabels( HEADERS )
        treeW.setColumnWidth(0,100)
        treeW.setColumnWidth(1,200)
```