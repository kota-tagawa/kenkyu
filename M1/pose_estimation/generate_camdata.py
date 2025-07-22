import os
from datetime import datetime
import numpy as np
from pose_estimation import function as fc
from pose_estimation import camera_data as cd

#============================================
# データを生成する関数
#generate_camdata(focus, dir_list, R_list):
#
# main関数
#============================================
def generate_camdata(focus, dir_list, R_list):

    #----------------------------------
    # 推定に関する設定
    #----------------------------------

    # 基準カメラの回転行列
    R0 = np.eye(3, dtype=np.float64)

    # 仮想カメラの数
    ncams = len(R_list)

    # データの保存(基本False)
    save_data = False

    #----------------------------------------------------------------------------
    # データ生成
    #----------------------------------------------------------------------------
    
    # カメラオブジェクトを生成
    cam_list = []
    for i in range(ncams):
        cam_list.append(cd.CameraData())

    for i, cam in enumerate(cam_list):
        # 回転行列
        cam.R_k0 = np.dot(R0, R_list[i].T)
        
        # 点特徴と線特徴に使用する点を選択
        point_3D = np.loadtxt(os.path.join(dir_list[i], "3Ddata.dat")).reshape(-1, 3, 1)
        point_2D = np.loadtxt(os.path.join(dir_list[i], "2Ddata.dat"))
        focus_point = np.full((point_2D.shape[0], 1), focus)
        point_2D = np.hstack((point_2D, focus_point)).reshape(-1, 3, 1)
        line_3D = point_3D
        line_2D = point_2D

        cam.p_alpha = point_3D
        cam.v_alpha = point_2D
        cam.r_alpha = line_3D[::2] # 偶数番目のみ使用
        cam.d_alpha = fc.calc_d_alpha(line_3D)
        cam.n_alpha = fc.calc_n_alpha(line_2D)

        # 射影行列を計算
        cam.V_alpha = np.zeros((cam.v_alpha.shape[0],3,3),dtype=np.float64)
        cam.K_alpha = np.zeros((cam.n_alpha.shape[0],3,3),dtype=np.float64)
        for j, v in enumerate(cam.v_alpha):
            cam.V_alpha[j] = fc.calc_V_alpha_kappa(v, cam.R_k0)
        for j, n in enumerate(cam.n_alpha):
            cam.K_alpha[j] = fc.calc_K_alpha_kappa(n, cam.R_k0)
        
        cam.p_prime, cam.r_prime = fc.calc_pr_prime(cam.p_alpha, cam.r_alpha)

        # データの保存
        if save_data:
            folder_name = os.path.join('data', datetime.now().strftime('%Y%m%d'))
            folder_path = os.path.join(dir_list[i], folder_name)
            # フォルダが存在しない場合は作成
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Folder ({folder_path}) has been created.")
            else:
                # フォルダが存在する場合は、その中のファイルを空にする
                for file_name in os.listdir(folder_path):
                    file_path = os.path.join(folder_path, file_name)
                    open(file_path, 'w').close()  # ファイルを空にする
                # print(f"Data in folder ({folder_path}) has been cleared.")

            cam.saveData(folder_path, i)
        
    return cam_list
    
    #-----------------------------------------------------------------------------------
    

if __name__ == "__main__":
    generate_camdata()