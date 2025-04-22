#moveviewpoint.py
#2023/5/10 tagawakota
#finished version
import cv2
import sys
import math
import csv
import numpy as np
from perspective import PixelLength,SightVector,CalcAngle,CoordTransform

def ReadFile(csvfile):
    with open(csvfile) as f:
        data = csv.reader(f)
        header = next(f)
        datas = [line for line in data]
    viewangle = datas[0]
    sightangle = datas[1]
    return viewangle,sightangle

### 画角から画像サイズ計算
def SizeOutimg(sizein, viewangle):
    theta = viewangle[0]
    phi = viewangle[1]
    w = int(2*math.tan(theta/2)*sizein[1]/(2*math.pi))+1
    h = int(2*math.tan(phi/2)*sizein[0]/math.pi)+1
    sizeout = (h,w)
    return sizeout

## 視線移動
def MoveViewpoint(sightvector,sightangle):
    t = sightangle[0] ### as theta
    p = sightangle[1] ### as phi
    e = sightangle[2] ### as epsilon
    ### y軸周りの回転行列
    Rmt1 = np.array(((math.cos(t),0,math.sin(t)),(0,1,0),(-1*math.sin(t),0,math.cos(t))))
    ### x軸周りの回転行列
    Rmt2 = np.array(((1,0,0),(0,math.cos(p),-1*math.sin(p)),(0,math.sin(p),math.cos(p))))
    ### 回転軸の計算
    Rmt12 = Rmt1@Rmt2
    v = np.array((0,0,1)).T
    l = Rmt12@v
    lx = l[0]
    ly = l[1]
    lz = l[2]
    ### 光軸(z軸)周りの回転行列
    Rmt3 = np.array(((lx*lx*(1-math.cos(e))+math.cos(e), \
        lx*ly*(1-math.cos(e))-lz*math.sin(e), \
        lz*lx*(1-math.cos(e))+ly*math.sin(e)), \
        (lx*ly*(1-math.cos(e))+lz*math.sin(e), \
        ly*ly*(1-math.cos(e))+math.cos(e), \
        ly*lz*(1-math.cos(e))-lx*math.sin(e)), \
        (lz*lx*(1-math.cos(e))-ly*math.sin(e), \
        ly*lz*(1-math.cos(e))+lz*math.sin(e), \
        lz*lz*(1-math.cos(e))+math.cos(e))))
    ## 視線ベクトルの回転
    viewpointvector = Rmt3@Rmt12@(sightvector.T)
    return viewpointvector

def CalcPerspectiveimage(size, sizein, viewangle, sightangle, img_omnidirectional):
    img_perspective = np.zeros((size[0],size[1],3))
    pixellength = PixelLength(size, viewangle)
    for vp in range(size[0]):
        for up in range(size[1]):
            pixel = [up,vp]
            sightvector = SightVector(pixel, size, pixellength)
            viewpointvector = MoveViewpoint(sightvector, sightangle)
            angle = CalcAngle(viewpointvector)
            ue,ve = CoordTransform(sizein, angle)
            img_perspective[vp,up,:] = img_omnidirectional[ve,ue,:]
    return img_perspective

if __name__ == "__main__":
    num_args = len(sys.argv)
    if (num_args > 4):
        print("usage : python 全方位画像.png input.csv [水平方向画角]")
        sys.exit()

    img_omnidirectional = cv2.imread(sys.argv[1])
    sizein = img_omnidirectional.shape
    
    viewangle,sightangle = ReadFile(sys.argv[2])
    viewangle = [math.radians(float(x)) for x in viewangle]
    sightangle = [math.radians(float(x)) for x in sightangle]
    
    ## シェルを用いる場合(引数3つ)
    if num_args == 4:
        filenum = sys.argv[3]
        theta = (float(sys.argv[3]) - 18.0)*10.0
        sightangle[0] = math.radians(theta)
    else:
        filenum = None
        pass
    
    sizeout = SizeOutimg(sizein,viewangle)

    img_perspective = CalcPerspectiveimage(sizeout, sizein, viewangle, sightangle, img_omnidirectional)
    cv2.imwrite(f'output/img_perspective{filenum}.jpg',img_perspective)
    cv2.waitKey()