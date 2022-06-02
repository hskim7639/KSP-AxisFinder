
import sys
sys.path.append('../../Phovision/CVQtVisPro')

from PyQt5.QtWidgets import QMenu,  QFileDialog
from AxisFinder import *
from CVQtWidget import *

class AxisFinderWidget(CVQtWidget):
    def __init__(self,  parent=None, winflags=Qt.WindowFlags(),  **kwargs):
        super().__init__(parent,  winflags)
        self._initProperties()
        
    def _initProperties(self):
        self.pseudocolor=False
        tempfn = 'coreless_template_01.jpg'
        self.template = cv.imread(tempfn)
        self.ntemp_h = 20 # 2*ntemp_h is the # of rows of templates
        htemp,  wtemp = self.template.shape[:2]
        c0 = htemp//2; c1=c0-self.ntemp_h; c2=c0+self.ntemp_h
        self.tempRange = (c1, c2) # template row ranges
        self.template = self.template[c1:c2, :]
        
        
    def onNewImageAvailable(self,  imgdata):
        cv_img,  opt = imgdata
        # this method is called by the worker when an image is ready to render 
        self.ledCapture.blink()
        QMutexLocker(self.mutexWork)
        #cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
        #image,  info = self._analyzeImage(cv_img,  pseudocolor=True)
        h, w = cv_img.shape[:2]
        res = cv.matchTemplate(cv_img[h//2-self.ntemp_h:h//2+self.ntemp_h, :],  self.template,  cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        top_left = max_loc; bottom_right = (top_left[0] + w, top_left[1] + h)
        #x = top_left[0]; y = top_left[1]
        temp_h,  temp_w = self.template.shape[:2]
        cx = top_left[0] + temp_w//2
        
        image,  info = AxisFinder._analyzeImage(cv_img,  tm_cx=cx,  tm_hafspan=117,  pseudocolor=self.pseudocolor)
        img =self.convertCVImgeToQImage(image)
        img2 = self.drawOverlayOnImage(img)

        self.canvas.setImage(img2)
        #self.canvas.resize(img2.size())
        self.repaint()
        
    def onPseudocolorSelected(self):
        print('pseudocolor')
        self.pseudocolor = not self.pseudocolor
        
    def onSaveImage(self):
        print('save image')
        QMutexLocker(self.mutexWork)
        fn = "cap_ovl_" + CVQtTool.getTimeStamp() + '.png'
        if self.canvas.img is not None:
            self.canvas.img.save(fn)
        
    def onSaveImageAs(self):
        print('save image as')
        QMutexLocker(self.mutexWork)
        filter = "Images (*.png *.jpg)"
        fn = QFileDialog.getSaveFileName(self,  'save image to file',  'cap_untitled.png',  filter)
        if len(fn[0])>0 and self.canvas.img is not None:
            fn = fn[0]
            self.canvas.img.save(fn)
        

    def contextMenuEvent(self, event):
        print('contex menu event')
        menu = QMenu(self)
        actPC = menu.addAction("pseudo color",  self.onPseudocolorSelected)
        actPC.setCheckable(True)
        actPC.setChecked(self.pseudocolor)
        menu.addSeparator()
        actSaveImage = menu.addAction("save image", self.onSaveImage)
        actSaveImageAs = menu.addAction("save image as..",  self.onSaveImageAs)
        menu.exec(event.globalPos())
        
        
        
if __name__=='__main__':
    app = QApplication(sys.argv)
    normalOperation = True
    probeCamera = False
    for argv in sys.argv[1:]:
        if '--help' in argv:
            normalOperation = False
        if '--probe' in argv:
            print ('probe cameras')
            normalOperation = False
            caminfo_list = CVQtCaptureDevice.findCameras()
            print(caminfo_list)
    if normalOperation:
        w = AxisFinderWidget()
        config =  CVQtVisionProcessorConfig()
        config.loadFromJsonFile('./CVQtVisionConfig.conf')
        w.setCamConfig(config.camConfigs[0],  config.dirInfo)
        w.show()
        app.exec()
