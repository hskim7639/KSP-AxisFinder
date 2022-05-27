
import sys
sys.path.append('../../Phovision/CVQtVisPro')

from CVQtWidget import *

class AxisFinderWidget(CVQtWidget):
    def __init__(self,  parent=None, winflags=Qt.WindowFlags(),  **kwargs):
        super().__init__(parent,  winflags)
        
        
    def onNewImageAvailable(self,  cv_img):
        # this method is called by the worker when an image is ready to render 
        self.ledCapture.blink()
        QMutexLocker(self.mutexWork)
        #cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2GRAY)
        image,  info = self._analyzeImage(cv_img,  pseudocolor=True)
        img =self.convertCVImgeToQImage(image)
        img2 = self.drawOverlayOnImage(img)

        self.canvas.setImage(img2)
        #self.canvas.resize(img2.size())
        self.repaint()
        
    def _analyzeImage(self,  cv_img,  **kwargs):
        info = {}
        if kwargs.get('pseudocolor'):
            img_pseudo = cv.applyColorMap(cv_img, cv.COLORMAP_JET)
        else:
            img_pseudo = cv_img
        geo = cv_img.shape # (h,w,color)
        h,  w = geo[:2]
        r_st,  navg = (100,  82)
        grayimg = cv.cvtColor(cv_img,  cv.COLOR_BGR2GRAY)
        row_avg = np.average(grayimg[r_st:r_st+navg, :],  axis=0)
        eh = 256
        avg_data = np.ndarray((w, 2), np.int32)
        avg_data[:, 0]=np.arange(w); avg_data[:, 1]=eh -row_avg - 1
        gr_img = np.zeros([eh, w],  np.uint8); gr_img.fill(255)
        #cv.polylines(gr_img,  avg_data.reshape((-1, 1, 2)),  True, (128) )
        for x in range(w):
            y = eh - int( row_avg[x]) -1
            gr_img[y:256, x] = 0
        #image = cv.copyMakeBorder(img_pseudo, eh, 0, 0, 0, cv.BORDER_CONSTANT,value=[255,255, 255])
        image = cv.copyMakeBorder(img_pseudo, eh, 0, 0, 0, cv.BORDER_CONSTANT,value=[255,255, 255])
        image[0:eh, :, 1] = gr_img
        return [image,  info]
        
        
        
        
if __name__=='__main__':
    app = QApplication(sys.argv)
    w = AxisFinderWidget()
    config =  CVQtVisionProcessorConfig()
    config.loadFromJsonFile('./CVQtVisionConfig.conf')
    w.setCamConfig(config.camConfigs[0],  config.dirInfo)
    w.show()
    app.exec()
