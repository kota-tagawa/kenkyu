import os
import numpy as np


class CameraData:

    def __init__(self):
        self.p_alpha = None  # 推定に使用する3次元点（p_in_range_P[0:npoints]で指定）
        self.v_alpha = None  # 推定に使用する2次元点（v_in_range_P[0:npoints]で指定）
        self.d_alpha = None  # 推定に使用する3次元直線のベクトル
        self.r_alpha = None  # 推定に使用する3次元直線上の点(p_in_range_L[0:nlines*2:2]で指定)
        self.n_alpha = None  # 推定に使用する画像面の直線パラメータ
        self.R_k0    = None  # 仮想カメラと基準カメラ間の回転行列
        self.V_alpha = None
        self.K_alpha = None
        self.p_prime = None
        self.r_prime = None
    
    #================================================
    # 推定に使用するデータをファイルに保存する関数
    # saveData(folder_name, i)
    # folder_name: データを保存するフォルダ
    # i: 保存するカメラの番号
    #================================================
    def saveData(self, folder_name, i):

        arrays = {
            'R_k0.dat': self.R_k0,
            'p_alpha.dat': self.p_alpha.reshape(-1, 3),
            'v_alpha.dat': self.v_alpha.reshape(-1, 3),
            'r_alpha.dat': self.r_alpha.reshape(-1, 3),
            'd_alpha.dat': self.d_alpha.reshape(-1, 3),
            'n_alpha.dat': self.n_alpha.reshape(-1, 3),
            'lv_alpha.dat': self.V_alpha.reshape(-1, 3),
            'lk_alpha.dat': self.K_alpha.reshape(-1, 3),
            'p_prime.dat': self.p_prime.reshape(-1, 3),
            'r_prime.dat': self.r_prime.reshape(-1, 3),
        }

        for file_name, array in arrays.items():
            file_path = os.path.join(folder_name, file_name)
            np.savetxt(file_path, array)
    
    #=================================================================
    # データファイルを読み込む関数
    # loadData(folder_name)
    # folder_name: p_alpha.dat などのファイルが保存されているフォルダ
    #=================================================================
    def loadData(self, folder_name):

        self.p_alpha = np.loadtxt(os.path.join(folder_name, 'p_alpha.dat')).reshape(-1, 3, 1)
        self.v_alpha = np.loadtxt(os.path.join(folder_name, 'v_alpha.dat')).reshape(-1, 3, 1)
        self.d_alpha = np.loadtxt(os.path.join(folder_name, 'd_alpha.dat')).reshape(-1, 3, 1)
        self.r_alpha = np.loadtxt(os.path.join(folder_name, 'r_alpha.dat')).reshape(-1, 3, 1)
        self.n_alpha = np.loadtxt(os.path.join(folder_name, 'n_alpha.dat')).reshape(-1, 3, 1)
        self.R_k0 = np.loadtxt(os.path.join(folder_name, 'R_k0.dat'))
        self.V_alpha = np.loadtxt(os.path.join(folder_name, 'lv_alpha.dat')).reshape(-1, 3, 3)
        self.K_alpha = np.loadtxt(os.path.join(folder_name, 'lk_alpha.dat')).reshape(-1, 3, 3)
        self.p_prime = np.loadtxt(os.path.join(folder_name, 'p_prime.dat')).reshape(-1, 3, 1)
        self.r_prime = np.loadtxt(os.path.join(folder_name, 'r_prime.dat')).reshape(-1, 3, 1)