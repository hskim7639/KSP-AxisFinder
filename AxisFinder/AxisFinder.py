import sys
import glob
import time
import numpy as np
import cv2 as cv


class AxisFinder():
    def __init__(self):
        pass

    @staticmethod
    def _analyzeImage(cv_img,  **kwargs):
        info = {}
        img_is_color = (len(cv_img.shape)==3)
        if kwargs.get('pseudocolor')==False:
            if img_is_color:
                img_pseudo = cv_img
            else:
                img_pseudo = cv.cvtColor(cv_img, cv.COLOR_GRAY2RGB)
        else:
            img_pseudo = cv.applyColorMap(cv_img, cv.COLORMAP_JET)
        h, w = cv_img.shape[:2] # (h,w)
        r_st,  navg = (100,  82)
        if img_is_color:
            grayimg = cv.cvtColor(cv_img,  cv.COLOR_BGR2GRAY)
        else:
            grayimg = cv_img
        # ----------------
        tm_thre = 0.5
        row_avg = np.average(grayimg[r_st:r_st+navg, :],  axis=0) # average of navg rows
        row_avgInt = np.array(row_avg,  np.uint8)
        eh = 256
        #avg_data = np.ndarray((w, 2), np.int32)
        #avg_data[:, 0]=np.arange(w); avg_data[:, 1]=eh -row_avgInt - 1
        gr_img = np.ones(eh* w * 3,  np.uint8).reshape((eh, w, 3))*255;  # graph
        #gr_img.fill([255, 255, 255])
        #cv.polylines(gr_img,  avg_data.reshape((-1, 1, 2)),  True, (128) )
        tm_guide_color = [0, 0, 255]
        tm_cx = kwargs.get('tm_cx')
        tm_cnt_hspan = 20; #  num of columns for average divided by 2 at center
        # average value of central zone
        tm_cavg = np.average(row_avg[tm_cx-tm_cnt_hspan:tm_cx+tm_cnt_hspan])
        if isinstance(tm_cx, int):
            gr_img[:,  tm_cx] = tm_guide_color
        tm_halfspan = kwargs.get('tm_hafspan')
        if isinstance(tm_halfspan,  int):
            gr_img[:,  tm_cx-tm_halfspan] = tm_guide_color
            gr_img[:,  tm_cx+tm_halfspan] = tm_guide_color
        else:
            tm_halfspan = w//2
        roiMargin = 15
        roiL = tm_cx-tm_halfspan+ roiMargin; roiR = tm_cx+tm_halfspan-roiMargin
        tm_cmin = np.min(row_avg[roiL:roiR])
        # ----- finding zerocrossings
        tm_vthre = ( (tm_cavg-tm_cmin)*tm_thre + tm_cmin )
        row_avg[0:roiL]=0.0; row_avg[roiR+1:]=0.0
        row_avg -= tm_vthre
        row_avg_sign = np.sign(row_avg)
        row_avg_diff = np.diff(row_avg_sign)
        zerocrossings = np.where(row_avg_diff != 0)[0]
        threL = zerocrossings[0]; threR = zerocrossings[1]
        threW = threR - threL
        info['guardwidth'] = threW
        gr_img[:, threL] = tm_guide_color
        gr_img[:,  threR] = tm_guide_color
        for x in range(w):
            y = eh -  row_avgInt[x] -1
            gr_img[y:256, x, :] = np.array([128, 128, 0], np.uint8)
            rx = (2*tm_cx - x)-1
            if (rx >= 0) and rx< w:
                y2 = eh - row_avgInt[rx]+10; 
                if y2<=0 or y2>=eh:
                    y2 =0
                gr_img[y2:256, x] = np.array([128, 0, 128],  np.uint8)
        txt = 'GW:{:3d}'.format(threW); pos = (10, 40)
        cv.putText(gr_img, txt,  pos,  cv.FONT_HERSHEY_SIMPLEX,  1,  (255, 0, 0),  2 )
        # ----- end of "finding zerocrossings
        
        #image = cv.copyMakeBorder(img_pseudo, eh, 0, 0, 0, cv.BORDER_CONSTANT,value=[255,255, 255])
        image = cv.copyMakeBorder(img_pseudo, eh, 0, 0, 0, cv.BORDER_CONSTANT,value=[255,255, 255])
        image[0:eh, :, :] = gr_img
        return [image,  info]


def main(argv):
    axfinder = AxisFinder()
    dir = '../CorelessFiber/Heights/'
    #dir = '../CorelessFiber/FineHeightGamm1/'
    dir = '../SMF/CoarseHeight/'
    #dir = '../SMF/FineHeight/'
    dir = '../SMF/FineHeightGamma1/'
    #dir = '../PMF/Heights/'
    #dir = '../PMF/CoarseRotation/'
    #dir = '../PMF/FineRotation/'
    #dir = './Images/'
    ext = '.jpg'
    files = glob.glob(dir + '*' + ext)
    files.sort()
    tempfn = 'coreless_template_01.jpg'
    #template = cv.imread(tempfn, cv.IMREAD_GRAYSCALE)
    template = cv.imread(tempfn)
    htemp,  wtemp = template.shape[:2]
    ntemp_h = 20; c0 = htemp//2; c1=c0-ntemp_h; c2=c0+ntemp_h
    template = template[c1:c2, :]
    #for fn in files[:]:
    for fn in files[0:]:
        print(fn)
        dts = np.array([], np.float64)
        dts = np.append(dts,  time.time())
        newfn =  fn.split(ext)[0] + '_PCxx.png'
        #cvimg = cv.imread(fn,  cv.IMREAD_GRAYSCALE)
        cvimg = cv.imread(fn)
        dts = np.append(dts,  time.time())
        h, w = template.shape[:2]
        res = cv.matchTemplate(cvimg[h//2-ntemp_h:h//2+ntemp_h, :],  template,  cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        dts = np.append(dts,  time.time())
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        x = top_left[0]; y = top_left[1]
        cx = top_left[0] + w//2
        #cvimg = cvimg[y:y+h,  x:x+w]
        dw = 50; dw2 = 70; guides = [w//2-dw2,  w//2-dw, w//2,  w//2+dw,  w//2+dw2]
        dts = np.append(dts,  time.time())
        ret_img,  optv = axfinder._analyzeImage(cvimg,  guides=guides,  tm_cx=cx,  tm_hafspan=117)
        print(optv)
        dts = np.append(dts,  time.time())
        cv.imwrite(newfn,  ret_img)
        dts = np.append(dts,  time.time())
        dts = (dts[1:] - dts[:-1])*1e3
        np.set_printoptions(precision=2)
        print('read, tm, ??, analysis, write')
        print(dts)

    return



if __name__=='__main__':
    main(sys.argv)
