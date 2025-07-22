# application.py
# editor : tagawa kota, sugano yasuyuki
# last edited : 2025/7/22
# Main loop of app. Assign texture from a 360-degree image to a 3D model.

# 全方位画像から透視投影画像生成
from e2p import E2P
# 外部パラメータ推定
from pose_estimation import pnpl
# 3dモデルロード
from mqoloader.loadmqo import LoadMQO
# 3dモデル編集
from create_mqo import CreateMQO

# 汎用ライブラリ(以下に追加)
import numpy as np
import cv2
import os 
import shutil
import math
from PIL import Image
import warnings

class Application:
    
    #================================================
    # アプリケーションクラスのコンストラクタ
    # __init__(self, model_path, config)
    #
    # 概要:
    #   3Dモデルの読み込みと、カメラパラメータ推定・テクスチャ生成に必要なパラメータやフォルダの初期化を行う。
    #
    # 引数:
    #   model_path : 読み込む3次元モデル（.mqo形式など）のファイルパス
    #   config     : 実行オプション（テクスチャ取得を行うかどうか）
    #
    # 主な処理内容:
    #   1. _setup_result_folder(model_path) : 結果出力フォルダの作成
    #   2. _setup_texture_folder() : テクスチャ出力用フォルダの作成
    #   3. _load_3Dmodel(model_path, output_model_path) : 3dモデルの読み込みとテクスチャ除外リストの取得
    #   4. _initialize_texture_params() : テクスチャ関連パラメータの初期化
    #   5. _initialize_camera_params() : 内部パラメータの読み込みと投影方向候補（Y軸回転角）の生成
    #   6. _initialize_config(config) : 実行オプション（config）に基づくモード指定
    #================================================
    
    def __init__(self, model_path, config):

        # 結果出力用フォルダを作成
        output_model_path = self._setup_result_folder(model_path)
        
        # テクスチャフォルダの準備
        self._setup_texture_folder()
        
        # 3Dモデルの読み込み
        self._load_3Dmodel(model_path, output_model_path)
        
        # テクスチャ関係のパラメータ初期化
        self._initialize_texture_params()
        
        # カメラ関係のパラメータ初期化
        self._initialize_camera_params()

        # 各種モード設定
        self._initialize_config(config)
    
    
    def _setup_result_folder(self,  model_path):
        # 結果出力用フォルダを作成
        result_folder = "result"
        os.makedirs(result_folder, exist_ok=True)

        # 出力用3dモデルファイルパスを生成
        output_model_path = os.path.join(result_folder, "C5F_edited.mqo")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"モデルファイルが存在しません: {model_path}")
        try:
            shutil.copy(model_path, output_model_path)
        except Exception as e:
            raise IOError(f"モデルファイルのコピーに失敗しました: {e}")

        return output_model_path
                
        
    def _setup_texture_folder(self):
        # フォルダごと削除して再作成（サブフォルダも含めて削除)
        folder_name = os.path.join("result","texture")
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)
        os.makedirs(folder_name)
        
        # 三角形メッシュ、四角形メッシュ用のサブフォルダを作成
        result_folders = [os.path.join(folder_name,"3v"), os.path.join(folder_name,"4v")]
        for folder in result_folders:
            os.makedirs(folder, exist_ok=True)
            
        # 単色テクスチャ生成
        width, height = 1600, 900
        colors = ["white"]
        for color in colors:
            image = Image.new("RGB", (width, height), color)
            image.save(f"result/texture/{color}.png")
        
        
    def _load_3Dmodel(self, model_path, output_model_path):
        # 3Dモデルの読み込み
        print('Loading %s ...' % model_path)
        self.use_normal = False
        self.model = LoadMQO(model_path, 0.0, self.use_normal)

        # テクスチャ除外リストの取得
        model_dir = os.path.dirname(model_path)
        texture_exclusion_path = os.path.join(model_dir, "texture_exclusion.dat")
        self.texture_exclusion = []
        if os.path.isfile(texture_exclusion_path):
            self.texture_exclusion = np.loadtxt(texture_exclusion_path, comments="#", delimiter=",", dtype=int)
        
        # テクスチャ情報を記述するインスタンス生成
        self.texture_num = len(self.model.f)
        self.mqo = CreateMQO(model_path, output_model_path, self.texture_num)
        print('Complete Loading !')
        
        
    def _initialize_texture_params(self):
        # テクスチャ用配列の初期化
        fnum = self.texture_num                                         # テクスチャ総数
        self.texture_camnum = np.full(fnum, 0)                          # テクスチャを割り当てるカメラ
        self.texture_dist = np.full(fnum, 100)                          # テクスチャカメラ間の距離
        self.texture_angle = np.full(fnum, 180)                         # テクスチャカメラ間の角度
        self.texture_area = np.full(fnum, 0)                            # 面積
        self.texture_scale = np.full((fnum, 2), 1.0)                    # 拡大スケール
        self.texture_R = np.zeros((fnum, 3, 3), dtype=np.float64)       # 回転行列
        self.texture_V = np.empty(fnum, dtype=object)                   # 頂点の座標(可変長)
        self.texture_uv = np.empty(fnum, dtype=object)                  # 頂点のuv座標(可変長)
        for i in range(fnum):
            self.texture_V[i] = np.zeros((0, 3), dtype=np.float32) 
        for i in range(fnum):
            self.texture_uv[i] = np.zeros((0, 3), dtype=np.float32)                                        
        
        
    def _initialize_camera_params(self):
        # キャリブレーションに用いる内部パラメータの読み込み(1.0倍)
        self.param = np.loadtxt(os.path.join("pose_estimation","mtx.txt"),delimiter=",")
        self.ufocus = self.param[0][0]
        self.vfocus = self.param[1][1]
        self.focus = (self.ufocus + self.vfocus)*0.5
        # 透視投影画像生成に用いる内部パラメータの読み込み(解像度0.5倍)
        self.param_mat = np.loadtxt(os.path.join("pose_estimation","mtx_half.txt"),delimiter=",")
        self.param_scale = 0.5
        
        # カメラパラメータ推定に用いる透視投影画像の角度設定
        def rot_y(theta_deg):
            # y軸周りの回転
            theta = np.deg2rad(theta_deg)
            return np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ], dtype=np.float64)
        
        angles_deg = [0, 180, -90, 90, -30, 30, -150, 150]
        self.R_proj_option = [rot_y(angle) for angle in angles_deg]

    def _initialize_config(self, config):
        # オプションを初期化
        self.estimate_camparam = config.get('estimate_camparam', False)
        
        
    #================================================
    # メインループ
    # mainloop(self, eimage_path)
    #
    # 引数
    #   eimage_path : 全方位画像のパスを含むリスト
    #   
    # 概要
    #  - 全方位画像から透視投影画像を生成し、カメラパラメータを推定する
    #  - カメラパラメータを用いて、全方位画像からテクスチャを生成する
    #  - 全方位カメラ情報を3Dモデルに書き込む
    #================================================    
    
    def mainloop(self, eimage_path):
        print(eimage_path)
        # 全方位画像読み込み
        pimage_paths, pimages, R_proj_list = self.read_eimage(eimage_path)
        # カメラパラメータ推定
        self.estimate_camera_params(pimage_paths, pimages, R_proj_list)

        # テクスチャ生成
        if not self.estimate_camparam:
            # テクスチャ情報を計算
            self.culc_texture_params()
            # カメラ情報書き込み
            self.mqo.write_campos(self.R_cam, self.t_cam, self.cam_num)
            # ３次元モデルファイル書き込み
            self.mqo.write_mqo()

    
    #================================================
    # 全方位画像読み込み  
    # pimage_paths, pimages, R_proj_list = read_eimage(self, eimage_path):
    # 
    # 引数
    #   eimage_path：全方位画像のファイルパス
    #
    # 返り値
    #   pimage_paths : 透視投影画像のパスを含むリスト
    #   pimages : 透視投影画像を含むリスト
    #   R_proj_list :　透視投影画像の回転行列
    # 
    # 概要
    # - 全方位画像を読み込み、複数の視線方向の透視投影画像を生成する
    # - 視線方向は R_proj_list 及び use_R_index.txt で指定する
    #================================================    
    
    def read_eimage(self, eimage_path):
        # 初期化
        W_angle, H_angle = 90, 90
        R_0 = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float64)

        # 結果保存用配列
        pimage_list = []
        R_proj_list = []
        pimage_paths = []
        
        # 全方位画像ファイル名からカメラ番号を取得
        eimage_filename = os.path.splitext(os.path.basename(eimage_path))[0]
        self.cam_num = int(eimage_filename.split('_')[-1])
        
        # 全方位カメラに対応するフォルダを作成
        folder_name = os.path.join('pose_estimation','camera_data','{}'.format(eimage_filename))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        # キャリブレーション用パラメータ計算
        eimage = cv2.imread(eimage_path)
        ewidth, eheight = eimage.shape[1], eimage.shape[0]
        pwidth = int(2 * math.tan((W_angle * np.pi/180.0)/2) * (ewidth/(2*np.pi)))
        pheight = int(2 * math.tan((H_angle * np.pi/180.0)/2) * (eheight/np.pi))
        angle_param_u = pwidth * np.pi / ewidth
        angle_param_v = 0.5 * pheight * np.pi / eheight
        map_e2p = E2P(ewidth, eheight, pwidth, pheight)
        
        # テクスチャ生成用パラメータ保存
        self.ewidth, self.eheight = ewidth*self.param_scale, eheight*self.param_scale
        self.pwidth = int(2 * math.tan((W_angle * np.pi/180.0)/2) * (self.ewidth/(2*np.pi)))
        self.pheight = int(2 * math.tan((H_angle * np.pi/180.0)/2) * (self.eheight/np.pi))
        
        # 仮想カメラの角度(self.R_proj_optionのindex)を指定する。ファイルがない場合は正面(index=0)を指定する。
        R_proj_index_filename = os.path.join(folder_name, 'use_R_index.txt')
        try:
            R_proj_index = np.loadtxt(R_proj_index_filename, delimiter=',', dtype=int)
        except FileNotFoundError:
            print(f"{R_proj_index_filename} が見つかりません。正面のものを使用します")
            R_proj_index = np.array([0], dtype=int)
        if R_proj_index.ndim == 0:  
            R_proj_index = np.array([R_proj_index]) # スカラー（0次元）なら1次元配列に変換
        
        for index in R_proj_index:
            # 仮想カメラ番号に対応するフォルダを作成
            subfolder_name = os.path.join(folder_name, 'cam_{}'.format(index))
            if not os.path.exists(subfolder_name):
                os.makedirs(subfolder_name)
                # print(f"Folder '{subfolder_name}' has been created.")

            # 透視投影画像生成
            R = np.dot(R_0, self.R_proj_option[index].T) # 仮想カメラの向き
            map_e2p.generate_map(angle_param_u, angle_param_v, R, scale = 1.0)
            pimage = map_e2p.generate_image(eimage)
            filename = os.path.join(subfolder_name, "pimage_{}.png".format(index))
            cv2.imwrite(filename, pimage)
            
            # 結果を保存
            pimage_paths.append(subfolder_name)
            pimage_list.append(pimage)
            R_proj_list.append(self.R_proj_option[index])
            
        return pimage_paths, pimage_list, R_proj_list
    
    
    #================================================
    # カメラパラメータ推定 
    # estimate_camera_params(self, pimage_paths, pimages, R_proj_list)
    #
    # 引数
    #   pimage_path : 透視投影画像のパスを含むリスト
    #   pimages : 透視投影画像を含むリスト
    #   R_proj_list : 透視投影画像の回転行列
    # 
    # 概要：
    # - 複数の角度の透視投影画像を用いてカメラの外部パラメータを推定する
    #================================================
    
    def estimate_camera_params(self, pimage_paths, pimages, R_proj_list):
        # カメラパラメータの測定値読み込み
        dir_name = os.path.dirname(pimage_paths[0].rstrip("/"))
        true_data_filename = os.path.join(dir_name, "param_true.dat")
        try:
            true_data = np.loadtxt(true_data_filename)
        except FileNotFoundError:
            # print(f"{true_data_filename} が見つかりません。測定値との比較は行いません。")
            true_data = None
      
        # 3Dモデルと画像を対応付け
        datafolder = pnpl.generate_cam_data(pimage_paths, self.focus, pimages, R_proj_list)
        if datafolder == -1:
            raise ValueError("2D-3D対応付け失敗")

        # カメラの外部パラメータを推定
        R_0 = np.array([[1,0,0],[0,0,-1],[0,1,0]],dtype=np.float64)
        flag, R, t = pnpl.PnPL_noError(datafolder, dir_name, true_data, R_0)
        if flag == 1:
            raise ValueError("カメラパラメータ計算失敗")
        else:
            self.R_cam = R
            self.t_cam = t.flatten()
                
    
    #================================================
    # テクスチャのパラメータ計算
    # culc_texture_params(self)
    #
    # 処理概要
    # - メッシュの座標を変換、距離(角度、面積)の条件が良い場合、テクスチャ情報を更新する
    #   
    # 使用メソッド:
    #   - self.transformation(vertices, i)
    #       入力：メッシュの頂点（世界座標系）、メッシュ番号
    #       出力：射影用回転行列、投影頂点、スケール、距離、角度、面積
    #================================================   
     
    def culc_texture_params(self):

        m = self.model
        for i, face in enumerate(self.model.f):

            # メッシュの世界座標を取得
            v_world = np.array([[m.x[face[idx]], m.y[face[idx]], m.z[face[idx]]] 
                                for idx in range(len(face))])
            # メッシュのuv座標を取得
            uv = np.array(m.uv[i])

            # 世界座標から透視投影画像上の座標に変換
            try:
                R_proj, v_proj, scale, dist, angle, area = self.transformation(v_world, i)
            except ValueError as e:
                # print(e)
                continue

            # 各種テクスチャ情報を更新(より面積が大きい方を採用)
            # 除外対象のテクスチャは取り除く(障害物越しに取得してしまうもの)
            texture_num = np.array([self.cam_num, i], dtype = int)
            is_excluded = False
            if len(self.texture_exclusion) > 0:
                is_excluded = np.any(np.all(self.texture_exclusion == texture_num, axis=1))

            if (self.texture_angle[i] > angle) and (not is_excluded):
                self.texture_camnum[i] = self.cam_num
                self.texture_dist[i] = dist
                self.texture_angle[i] = angle
                self.texture_area[i] = area
                self.texture_R[i] = R_proj
                self.texture_V[i] = v_proj
                self.texture_scale[i] = scale
                self.texture_uv[i] = uv

    
    #================================================
    # 座標系変換
    # R_proj, v_proj, scale, dist, angle, area = transformation(self, v_world, i)
    # 
    # 引数
    #   v_world：世界座標系のメッシュの頂点座標
    #   i : メッシュ番号
    #
    # 戻り値
    #   R_proj : メッシュ重心を向く回転行列
    #   v_proj : 透視投影画像上に投影された座標
    #   dist : メッシュの重心からの距離
    #   angle : メッシュの方向ベクトルと法線ベクトルの角度
    #   area : メッシュの投影面積
    #
    # 処理概要
    #  - 世界座標系からカメラ座標系に変換し、透視投影画像上の座標に変換する
    #  - メッシュが透視投影画像の中に納まるように、透視投影画像のスケールを拡大する
    #================================================ 
    
    def transformation(self, v_world, i):
        # しきい値
        thd_dist = 5
        thd_angle = 80
        max_width = 10000
        max_height = 10000
        
        # 初期化
        R_proj = np.eye(3)
        scale = np.array([1.0, 1.0])
        width, height = self.pwidth, self.pheight

        # カメラ座標系に変換
        v_cam = np.array([np.dot(self.R_cam, v) + self.t_cam for v in v_world])
        
        # メッシュの重心までの距離がしきい値以上のテクスチャは採用しない
        centroid = np.mean(v_cam, axis = 0)
        dist = np.linalg.norm(centroid)
        if dist > thd_dist:
            raise ValueError("texture({}_{}) : dist exceeds the threshold ({}m) ".format(self.cam_num, i, dist))
        
        # メッシュの法線ベクトル、方向ベクトルを計算
        normal = np.cross(v_cam[1] - v_cam[0], v_cam[2] - v_cam[0])
        norm_vec = normal / np.linalg.norm(normal)
        view_vec = centroid / np.linalg.norm(centroid)
        # 法線ベクトルの符号調整
        if np.dot(view_vec, norm_vec) < 0.0:
            raise ValueError("texture({}_{}) : backside".format(self.cam_num, i))
            
        # 方向ベクトルと法線ベクトルのなす角が大きいテクスチャは採用しない
        cos_theta = np.clip(np.dot(view_vec, norm_vec), -1.0, 1.0)
        angle = np.degrees(np.arccos(cos_theta))
        if not (angle < thd_angle):
            raise ValueError("texture({}_{}) : angle exceeds the threshold ({:.2f}°)".format(self.cam_num, i, angle))
            
        # Z軸を重心方向に合わせる(法線ベクトルを用いる時はここをnorm_vecに変更)
        Z = view_vec
        # 射影行列を計算
        P = np.eye(3) - np.outer(Z, Z)
        y = np.dot(P, np.array([0, 1, 0]))
        if np.linalg.norm(y) < 1e-10:
            Y = np.array([0, 0, -1])
            X = np.array([1, 0, 0])
        else:
            Y = y / np.linalg.norm(y)
            X = np.cross(Y, Z)
        R_proj = np.column_stack((X, Y, Z))
        
        # カメラ座標を透視投影画像上の2D座標に変換
        def _project_vertices(v_cam, R_proj, param_mat, scale):
            scaled_param_mat = param_mat.copy() # 内部パラメータに拡大縮小を適用
            scaled_param_mat[0, 2] *= scale[0]
            scaled_param_mat[1, 2] *= scale[1]

            v_proj = np.zeros_like(v_cam)
            for idx, vertex in enumerate(v_cam):
                rotated = np.dot(R_proj.T, vertex)  # メッシュ重心方向への回転
                projected = np.dot(scaled_param_mat, rotated)
                projected[:2] /= projected[2]
                v_proj[idx] = projected

            return v_proj

        while True:
            v_proj = _project_vertices(v_cam, R_proj, self.param_mat, scale)

            # メッシュが透視投影画像内に収まっているかを判定
            is_within_image_bounds = (
                np.all((v_proj[:, 0] > -1, v_proj[:, 0] < width+1)) and
                np.all((v_proj[:, 1] > -1, v_proj[:, 1] < height+1))
            )
            if is_within_image_bounds:
                break

            # 画像中心からの最大距離をもとに透視投影画像のサイズを更新
            image_center = np.array([width / 2, height / 2])
            max_x_offset = np.max(np.abs(v_proj[:, 0] - image_center[0]))
            max_y_offset = np.max(np.abs(v_proj[:, 1] - image_center[1]))
            new_width = max(self.pwidth, int(np.ceil(2 * max_x_offset)))
            new_height = max(self.pheight, int(np.ceil(2 * max_y_offset)))
            if width > max_width or height > max_height:
                warnings.warn(
                    f"texture({self.cam_num}_{i}): scale too large ({new_width}, {new_height}). "
                    f"Consider reducing the input image size to avoid memory issues.",
                    UserWarning
                )
            
            # 透視投影画像のスケールを更新
            width, height = new_width, new_height
            scale = np.array([width / self.pwidth, height / self.pheight])
    

        # 面積計算（四角形の場合、三角形2つ分を合計）
        if len(v_proj) == 4:
            area = (
                0.5 * np.linalg.norm(np.cross(v_proj[1] - v_proj[0], v_proj[2] - v_proj[0])) +
                0.5 * np.linalg.norm(np.cross(v_proj[2] - v_proj[0], v_proj[3] - v_proj[0]))
            )
        else:
            area = 0.5 * np.linalg.norm(np.cross(v_proj[1] - v_proj[0], v_proj[2] - v_proj[0]))

        return R_proj, v_proj, scale, dist, angle, area


    #================================================
    # テクスチャを生成 
    # generate_texture(self, eimage_path)
    #
    # 引数
    #   eimage_path：全方位画像パスを含むリスト
    # 
    # 概要：計算されたテクスチャ情報をもとに全方位画像からテクスチャを生成し、3Dモデルファイルを更新する
    #================================================    
    
    def generate_texture(self, eimage_path):
        # 画像名の時はリストに変換
        if isinstance(eimage_path, str):
            eimage_path = [eimage_path]  

        # 全方位画像の番号を指定して読み込む
        loaded_images = {}
        for path in eimage_path:
            eimage_filename = os.path.splitext(os.path.basename(path))[0]
            cam_num = int(eimage_filename.split('_')[-1])
            eimage = cv2.imread(path)
            loaded_images[cam_num] = cv2.resize(eimage, (0, 0), fx=self.param_scale, fy=self.param_scale)
        
        # 全てのテクスチャ情報を更新
        for i in range(self.texture_num):
            # テクスチャ情報の読み込み
            camnum = self.texture_camnum[i]
            vnum = len(self.texture_V[i])
            
            # テクスチャ生成失敗
            if vnum == 0:
                print(f"texture({i})：texture not generated")
                continue
            else:
                eimage = loaded_images[camnum]

            # 透視投影画像の生成
            pwidth = int(np.ceil(self.pwidth * self.texture_scale[i][0]))
            pheight = int(np.ceil(self.pheight * self.texture_scale[i][1]))
            map = E2P(self.ewidth, self.eheight, pwidth, pheight)
            angle_param_u = pwidth * np.pi / self.ewidth
            angle_param_v = 0.5 * pheight * np.pi / self.eheight
            R = self.texture_R[i]
            map.generate_map(angle_param_u, angle_param_v, R, scale = 1.0)
            img_org = map.generate_image(eimage)

            # 射影変換行列を計算
            v_org = np.array([[point[0], point[1]] for point in self.texture_V[i]], dtype=np.float32)
            v_proj = np.array(self.texture_uv[i], dtype=np.float32)
            v_proj[:, 0] *= pwidth
            v_proj[:, 1] *= pheight
            if vnum == 3:
                M = cv2.getAffineTransform(v_org, v_proj)
                img_proj = cv2.warpAffine(img_org, M, (pwidth, pheight))
            elif vnum == 4:
                H = cv2.getPerspectiveTransform(v_org, v_proj)
                img_proj = cv2.warpPerspective(img_org, H, (pwidth, pheight))
                
            
            def _annotate_texture(img, v):
                # 線と頂点を描画
                # for i in range(len(v)): 
                #     pt1 = int(v[i][0]), int(v[i][1])
                #     pt2 = int(v[(i + 1) % len(v)][0]), int(v[(i + 1) % len(v)][1])
                #     cv2.circle(img, pt1, 1, (255, 255, 255), 1)
                #     cv2.line(img, pt1, pt2, (255, 255, 255), thickness=1)

                # マスク適用
                pts = np.array(v, dtype=np.int32).reshape((-1, 1, 2))
                mask = np.zeros(img.shape[:2], dtype=np.uint8)
                cv2.fillPoly(mask, [pts], 255)
                annotate_img = cv2.bitwise_and(img, img, mask=mask)

                return annotate_img
    
            # テクスチャに様々な情報を付加
            image_with_masked = _annotate_texture(img_proj, v_proj)

            # テクスチャ画像保存
            texture_filename = os.path.join('texture_{}_{}.png'.format(camnum, i))
            subfolder = os.path.join('texture','{}v'.format(str(vnum)))
            texture_path = os.path.join(subfolder, texture_filename)
            cv2.imwrite(os.path.join('result', texture_path), image_with_masked)
            
            # vは透視投影画像サイズで正規化
            v_proj[:, 0] /= pwidth
            v_proj[:, 1] /= pheight
            # 3次元モデル情報のアップデート
            self.mqo.update_mqo(v_proj, texture_path, i)
            # ３次元モデルファイル書き込み
            self.mqo.write_mqo()
            # テクスチャ記述完了
            print(f"texture({i})：The texture was generated [{texture_path}]")