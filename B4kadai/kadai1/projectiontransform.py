#projectiontransform.py
#2023/4/17 tagawakota
#finished version
import numpy as np
import cv2

## 座標の入力
def InputCoordinate():
    ### 変換前の座標の入力
    print("変換前の座標を入力")
    print("入力方法 -> x座標,y座標")
    print("終了する場合は q")

    point1 = []
    point2 = []
    count = 0
    while(True):
        a = input()
        if(a =="q"):
            break
        else:
            a = list(map(float,a.split(',')))
            point1.append(a)
            count+=1

    pointcount = count

    ### 変換後の座標の入力
    print("変換後の座標を入力")
    print("入力方法 -> x座標,y座標")
    count = 0
    while(True):
        a = list(map(float,input().split(',')))
        point2.append(a)
        count+=1
        if(pointcount == count):
            break

    return point1,point2

## 射影変換
def ProjectionTransform(point1,point2):
    pointcount = len(point1)
    ### 係数行列
    a = []
    for i in range(pointcount):
        a.append([point1[i][0],point1[i][1],1,0,0,0,-1*point1[i][0]*point2[i][0],-1*point1[i][1]*point2[i][0]])
    for j in range(pointcount):
        a.append([0,0,0,point1[j][0],point1[j][1],1,-1*point1[j][0]*point2[j][1],-1*point1[j][1]*point2[j][1]])
    a = np.array(a)

    ### 解の行列
    b = []
    for i in range(pointcount):
        b.append(point2[i][0])
    for j in range(pointcount):
        b.append(point2[j][1])
    b = np.array(b)

    ### ホモグラフィ変換行列
    h = np.linalg.inv(a.T@a)@a.T@b
    ### 演算しやすいように変形
    hsqu = np.reshape(np.insert(h,8,float(1.0)),(3,3))
    return hsqu



if __name__ == "__main__":
    point1,point2 = InputCoordinate()
    print(point1)
    print(point2)
    tm = ProjectionTransform(point1,point2)
    print(tm)
    print(cv2.getPerspectiveTransform(np.float32(point1), np.float32(point2)))


