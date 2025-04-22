#panoramasize.py
#2023/4/19 tagawakota
#finished version
from projectiontransform import ProjectionTransform, InputCoordinate
import numpy as np
import cv2
import sys

## 画像の読み込み
def Imgread(imgname1,imgname2):
    img1 = cv2.imread(imgname1)
    img2 = cv2.imread(imgname2)
    return img1,img2

## パノラマ画像サイズの決定（わからん）
def PanoramaSize(h,shape1,shape2):
    hinv = np.linalg.inv(h)
    ### 第二画像のサイズ
    co1 = (np.array([0.0,0.0,1.0])).T
    co2 = (np.array([float(shape2[1]),0.0,1.0])).T
    co3 = (np.array([0.0,float(shape2[0]),1.0])).T
    co4 = (np.array([float(shape2[1]),float(shape2[0]),1.0])).T
    ### 正規化
    co1panorama = [w/(hinv@co1)[2] for w in hinv@co1]
    co2panorama = [x/(hinv@co2)[2] for x in hinv@co2]
    co3panorama = [y/(hinv@co3)[2] for y in hinv@co3]
    co4panorama = [z/(hinv@co4)[2] for z in hinv@co4]
    ### サイズの決定
    maxwidth = max(shape1[1],co1panorama[0],co2panorama[0],co3panorama[0],co4panorama[0])
    minwidth = min(0.0,co1panorama[0],co2panorama[0],co3panorama[0],co4panorama[0])
    maxheight = max(shape1[0],co1panorama[1],co2panorama[1],co3panorama[1],co4panorama[1])
    minheight = min(0.0,co1panorama[1],co2panorama[1],co3panorama[1],co4panorama[1])
    print(maxwidth)
    print(minwidth)
    print(maxheight)
    print(minheight)
    width = maxwidth - minwidth
    height = maxheight - minheight
    size = (int(height),int(width))
    offset = (int(minheight),int(minwidth))
    return size,offset


if __name__ == "__main__":
    num_args = len(sys.argv)
    if (num_args != 3):
        print("usage : python img1.jpg img2.jpg")
        sys.exit()
    img1,img2 = Imgread(sys.argv[1],sys.argv[2])

    point1,point2 = InputCoordinate()
    h = ProjectionTransform(point1,point2)
    panoramasize,ofset = PanoramaSize(h,img1.shape,img2.shape)
    print(panoramasize)

