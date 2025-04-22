# pnpl.py
# editor : tagawa kota, sugano yasuyuki
# last edited : 2025/3/13
# Estimate camera parameters from 2D-3D correspondence using PnPL method

from pose_estimation import function as fc
from pose_estimation import generate_camdata as gc
from pose_estimation import camera_data as cd

# 汎用ライブラリ(以下に追加)
import numpy as np
import cv2
import os
from datetime import datetime

#============================================
# 2次元データ、3次元データを対応付け
# folder_name = generate_cam_data(dir_list, focus, pimage_list, R_list):
#
# 引数
#   dir_list：透視投影画像のパスを含むリスト
#   focus : 全方位カメラの焦点距離
#   pimage_list : 透視投影画像を含むリスト
#   R_list : 透視投影画像の回転行列
#
# 返り値
#   folder_name : PnPLの計算に必要なカメラデータを含むフォルダパス
# 
# 概要
# - 3次元データと透視投影画像の2次元座標を対応付けする
# - 3次元の点特徴と線特徴は事前にファイルで指定されている
# - それに対応する二次元の点特徴、線特徴の座標を透視投影画像から手動で選択する
# - 対応付けされたデータから、カメラパラメータ推定に必要なデータを生成する
#================================================    

def generate_cam_data(dir_list, focus, pimage_list, R_list):
    for path, img in zip(dir_list, pimage_list):
        files = set(os.listdir(path))
        
        # 3Dデータを取得
        if "3Ddata.dat" in files:
            point_3D = np.loadtxt(os.path.join(path, "3Ddata.dat"))
            if len(point_3D)%2 != 0:
                print(f"{path}/3Ddata.dat に含まれるデータ数は偶数にしてください。")
                return -1
        else:
            print(f"{path}/3Ddata.dat が存在しません。")
            return -1
        
        # 2Dデータを取得（手動入力 or 既存データ）
        if not "2Ddata.dat" in files:
            point_2D = get2D(len(point_3D), img, mode = "point")
            np.savetxt(os.path.join(path, "2Ddata.dat"), point_2D)
            
            if len(point_2D) != len(point_3D):
                print("2D-3D対応点の個数が異なります。")
                return -1
        
    # データ生成
    cam_list = gc.generate_camdata(focus, dir_list, R_list)
    return cam_list
    
#============================================
# マウスイベント
# onMouse(self, event, x, y, flag, params)
#
# 引数
#   event : イベント受付
#   x,y : マウスのカーソルの座標
#   params : 入力パラメータ
#
# 概要
# - マウスクリックで点・線特徴を取得
#============================================
def onMouse(event, x, y, flag, params):
    mode, raw_img, scale_factor, point_list, max_points = params.values()

    # 選択された座標を元のスケールに変換
    orig_x, orig_y = int(x * scale_factor), int(y * scale_factor)

    # 2次元点特徴取得
    if mode == "point":
        point_list = params["list"]
        point_num = params["num"]
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(point_list) < point_num:
                point_list.append([orig_x, orig_y])
        if event == cv2.EVENT_RBUTTONDOWN:
            if len(point_list) > 0:
                point_list.pop(-1)

        img = raw_img.copy()
        for i in range(len(point_list)):
            # 表示用に縮小座標で描画
            disp_x, disp_y = int(point_list[i][0] / scale_factor), int(point_list[i][1] / scale_factor)
            cv2.circle(img, (disp_x, disp_y), 3, (0, 0, 255), 3)
    
    ## 座標情報をテキストで出力
    cv2.putText(img, "({0}, {1})".format(orig_x, orig_y), (0, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow(mode, img)   
    
#============================================
# 透視投影画像から２次元座標を手動で取得
# shifted_point_array = get2D(point_num, pimage, mode = "point")
# 
# 引数
#   point_num：特徴を対応付けする数
#   pimage：透視投影画像
#   mode：点特徴(point)か線特徴(line)
#
# 返り値
#   shifted_point_array : 中心が原点になるようにシフトされた座標
#============================================                
def get2D(point_num, pimage, mode):
    point_list = []
    img_height, img_width = pimage.shape[:2]
    display_height, display_width = 900, 1600

    scale_factor = max(img_height / display_height, img_width / display_width)
    resized_image = cv2.resize(pimage, (int(img_width / scale_factor), int(img_height / scale_factor)))

    shiftx = img_width / 2
    shifty = img_height / 2

    params = {
        "mode": mode,
        "img": resized_image,
        "scale_factor": scale_factor,
        "list": point_list,
        "num": point_num
    }

    cv2.namedWindow(mode, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(mode, onMouse, params)
    cv2.imshow(mode, resized_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # 光軸中心が原点になるようにシフト
    point_array = np.array(point_list)
    shifted_point_array = point_array - np.array([shiftx, shifty])
    return shifted_point_array

#=================================================================
# 誤差なしでPnPLを計算
# flag, R_PnPL, t_PnPL = PnPL_noError(folder_path, true_data)
# 
# 引数
#   folder_path : カメラデータのサブフォルダが入ったフォルダのパス
#   true_data : カメラのパラメータの実測値
#
# 返り値
#   flag : 推定失敗→1
#   R_PnPL : 推定されたカメラの回転行列
#   t_PnPL : 推定されたカメラの並進ベクトル
#
# 概要
# - 2D-3Dの点対応、線対応によって生成されたデータを用いて、誤差なしでPnPLを計算する
# - 実行結果は /CameraData/eimage_/data/param に保存
#=================================================================
def PnPL_noError(cam_list, true_data):
    # 保存する
    save_data = True
    
    # 収束判定のしきい値
    thd = 1.0e-8

    # 回転行列の初期値
    R_0 = np.array([[1,0,0],
                    [0,0,-1],
                    [0,1,0]],
                    dtype=np.float64)

    # # カメラデータの読み込み
    # print(folder_path)
    # cam_list = []
    # subfolders = [f for f in os.listdir(folder_path) 
    #               if os.path.isdir(os.path.join(folder_path, f)) and f.startswith("cam")]
    # if subfolders:
    #     for subfolder in subfolders:
    #         subfolder_path = os.path.join(folder_path, subfolder)
    #         cam = cd.CameraData()
    #         cam.loadData(subfolder_path)
    #         cam_list.append(cam)
            
    # else:
    #     cam = cd.CameraData()
    #     cam.loadData(folder_path)
    #     cam_list.append(cam)
    
    R_PnPL,t_PnPL,k_PnPL,flag,E =fc.PnPL(thd, cam_list, R_0)

    if flag == 1:
        print("データが不適切です")

    # カメラ位置を計算
    C = -np.dot(R_PnPL.T,t_PnPL)

    np.set_printoptions(precision=8,suppress=True,floatmode='fixed')
    print("E(目的関数) : ", E)
    print("反復回数 : ", k_PnPL)
    print("PnPLで求めたカメラ位置\n" , C , "\n")
    print("PnPLで求めた回転行列\n", R_PnPL, "\n")

    # 回転行列とカメラ位置の真値
    if true_data is not None:
        R0 = true_data[:, :3]
        t = true_data[:, 3].reshape(-1, 1)
        c = -np.dot(R0.T, t)
        print("カメラ位置(真値):\n", c, "\n")
        print("回転行列(真値):\n", R0)
        print("R_error:\n",R_PnPL-R0)
        print("C_error:\n",C-c)
    
    if save_data:
        # フォルダが存在しない場合作成
        folder_name = os.path.join('result/params/', datetime.now().strftime('%Y%m%d'))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' has been created.")
        np.savetxt(folder_name+"/k_PnPL.txt", [k_PnPL])
        np.savetxt(folder_name+"/C.dat", C)
        np.savetxt(folder_name+"/R_PnPL.dat", R_PnPL)
        np.savetxt(folder_name+"/t_PnPL.dat", t_PnPL)
        if true_data is not None:
            np.savetxt(folder_name+"/R_error.dat", R_PnPL-R0)
            np.savetxt(folder_name+"/C_error.dat", C-c)
        
    return flag, R_PnPL, t_PnPL
