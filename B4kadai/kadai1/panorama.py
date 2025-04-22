#panorama.py
#2023/4/20 tagawakota
#unfinished version
from panoramasize import Imgread, PanoramaSize
from projectiontransform import ProjectionTransform, InputCoordinate
import cv2
import sys
import math
import numpy as np
## 双一次補完(4/20)
def PrimaryComplement(x,y,img2):
    ### math.modf return tuple, (decimal, integer)
    x = math.modf(x)
    y = math.modf(y)
    b1,g1,r1 = img2[y,x]
    b2,g2,r2 = img2[y,x+1]
    b3,g3,r3 = img2[y+1,x]
    b4,g4,r4 = img2[y+1,x+1]
    b = (1-y[0])*(([1-x[0]])*b1+x[0]*b2)+y[0]*((1-x[0])*b3+x[0]*b4)
    g = (1-y[0])*(([1-x[0]])*g1+x[0]*g2)+y[0]*((1-x[0])*g3+x[0]*g4)
    r = (1-y[0])*(([1-x[0]])*r1+x[0]*r2)+y[0]*((1-x[0])*r3+x[0]*r4)
    return (b,g,r)
## パノラマ画像出力
def Panorama1(img1,img2,h,size,offset):
    img1_transform = cv2.warpPerspective(img2,np.linalg.inv(h),(size[1],size[0]))
    img_panorama = img1_transform.copy()
    img_panorama[-offset[0]:img1.shape[0]-offset[0],-offset[1]:img1.shape[1]-offset[1]] = img1.copy()
    return img_panorama

## パノラマ画像出力(画素抽出)
def Panorama2(img1,img2,h,size,offset):
    ### 単色画像生成
    img_panorama = np.zeros((size[0],size[1],3))
    ### 出力画像に入力1の画素を、オフセットを減算してコピー
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            x = j-offset[1]
            y = i-offset[0]
            img_panorama[y,x] = img1[i,j]
    print(img_panorama[50-offset[0],50-offset[1]])
    print(img1[50,50])
    ### 出力画像にオフセットを加算して射影変換
    i=0
    j=0
    for i in range(img_panorama.shape[0]):
        for j in range(img_panorama.shape[1]):
            w = [j+offset[1], i+offset[0], 1]
            x = int((h@w)[0]/(h@w)[2])
            y = int((h@w)[1]/(h@w)[2])
            ### 射影変換後の座標が入力2の領域に含まれるか確認し、入力2の画素をコピー
            if 0<x<img2.shape[1] and 0<y<img2.shape[0]:
                ### 双一時補完(4/20)
                img_panorama[i,j] = PrimaryComplement(x,y,img2)
                ### img_panorama[i,j] = img2[y,x]
    return img_panorama

if __name__ == "__main__":
    num_args = len(sys.argv)
    if (num_args != 3):
        print("usage : python img1.jpg img2.jpg")
        sys.exit()
    img1,img2 = Imgread(sys.argv[1],sys.argv[2])
    point1,point2 = InputCoordinate()
    h = ProjectionTransform(point1,point2)
    panoramasize,offset = PanoramaSize(h,img1.shape,img2.shape)
    img_panorama = Panorama2(img1,img2,h,panoramasize,offset)
    cv2.imwrite('Blender_Suzanne_panorama2.jpg',img_panorama)
    cv2.waitKey()
