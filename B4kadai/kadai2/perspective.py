#perspectve.py
#2023/4/27 tagawakota
#finished version
import cv2
import sys
import math
import csv
import numpy as np

def ReadFile(csvfile):
    with open(csvfile) as f:
        data = csv.reader(f)
        header = next(f)
        datas = [line for line in data]
    size = datas[0]
    viewangle = datas[1]
    return size,viewangle
    
def PixelLength(size, viewangle):
    theta = viewangle[0]
    phi = viewangle[1]
    deltax = 2*math.tan(theta/2)/size[1]
    deltay = 2*math.tan(phi/2)/size[0]
    pixellength = (deltax,deltay)
    return pixellength

def SightVector(pixel, size, pixellength):
    deltax = pixellength[0]
    deltay = pixellength[1]
    x = (pixel[0] - size[1]/2)*deltax
    y = (pixel[1] - size[0]/2)*deltay
    z = 1.0
    sightvector = np.array((x,y,z))
    return sightvector

def CalcAngle(sightvector):
    x = sightvector[0]
    y = sightvector[1]
    z = sightvector[2]
    theta = math.atan2(x,z)
    phi = -1*math.atan2(y,math.sqrt(x**2+z**2))
    angle = (theta,phi)
    return angle

def CoordTransform(sizein, angle):
    theta = angle[0]
    phi = angle[1]
    ue = int((theta+math.pi)*sizein[1]/(2*math.pi))
    ve = int((math.pi/2-phi)*sizein[0]/math.pi)
    return ue,ve

def CalcPerspectiveimage(size, sizein, viewangle, img_omnidirectional):
    img_perspective = np.zeros((size[0],size[1],3))
    pixellength = PixelLength(size, viewangle)
    for vp in range(size[0]):
        for up in range(size[1]):
            pixel = [up,vp]
            sightvector = SightVector(pixel, size, pixellength)
            angle = CalcAngle(sightvector)
            ue,ve = CoordTransform(sizein, angle)
            img_perspective[vp,up,:] = img_omnidirectional[ve,ue,:]
    return img_perspective


if __name__ == "__main__":
    num_args = len(sys.argv)
    if (num_args != 3):
        print("usage : python 全方位画像.png input.csv")
        sys.exit()

    img_omnidirectional = cv2.imread(sys.argv[1])
    sizein = img_omnidirectional.shape
    
    
    size,viewangle = ReadFile(sys.argv[2])
    size = [int(x) for x in size]
    viewangle = [math.radians(float(x)) for x in viewangle]
       
    img_perspective = CalcPerspectiveimage(size, sizein, viewangle, img_omnidirectional)
    cv2.imwrite('img_perspective.jpg',img_perspective)
    cv2.waitKey()