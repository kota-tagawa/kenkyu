# application.py
# editor : tagawa kota, sugano yasuyuki
# last edited : 2025/3/12
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

class Application:
    
    #================================================
    # アプリケーションクラスのコンストラクタ
    # __init__(self, model_path)
    #
    # 引数
    #   model_path : 3次元モデルのファイルパス
    #
    # コンストラクタで使用するメソッド
    #   setup_texture_folder()
    #       - テクスチャフォルダを準備
    #   load_3Dmodel(model_path, output_model_path)
    #       - 3Dモデルのファイルパス model_path を読み込み、モデルとテクスチャのインスタンスを生成
    #   initialize_texture_params()
    #       - テクスチャの情報を初期化
    #   initialize_camera_params
    #       - キャリブレーションデータ読み込み、カメラパラメータ推定に用いる回転行列の設定
    #================================================    
    
    def __init__(self, model_path):
        
        self.eimg_count = 0
        self.pimg_count = 0
        self.cam_num = 0
        self.use_truedata = False   # 真値を使用
        self.use_adjust_texture_brightness = False   # テクスチャの明るさ調整を使用
        
        # 結果出力用フォルダを作成
        output_model_path = self._setup_result_folder(model_path)
        if output_model_path == -1:
            raise ValueError("Failed to set up result folder")
        
        # テクスチャフォルダの準備
        self._setup_texture_folder()
        
        # 3Dモデルの読み込み
        self._load_3Dmodel(model_path, output_model_path)
        
        # テクスチャ関係のパラメータ初期化
        self._initialize_texture_params()
        
        # カメラ関係のパラメータ初期化
        self._initialize_camera_params()
    
    
    def _setup_result_folder(self,  model_path):
        # 結果出力用フォルダを新規作成（存在する場合はそのまま）
        result_folder = "result"
        os.makedirs(result_folder, exist_ok=True)  # 存在しない場合のみ作成
        
        # 出力用3Dファイル作成 (model_pathのファイルをコピー)
        if os.path.exists(model_path):
            output_model_path = os.path.join(result_folder, "C5F_edited.mqo")
            try:
                shutil.copy(model_path, output_model_path)
                print(f"Model file copied to {output_model_path}.")
                return output_model_path
            
            except Exception as e:
                print(f"Error copying file: {e}")
        else:
            print(f"Model file at {model_path} does not exist.")
        
        return -1 
                
        
    def _setup_texture_folder(self):
        # フォルダごと削除して再作成（サブフォルダも含めて削除)
        folder_name = os.path.join("result","texture")
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)
            print(f"All data in Texture folder has been removed.")
        
        # テクスチャフォルダを作成
        os.makedirs(folder_name)
        print(f"Texture Folder has been created.")
        
        # サブフォルダを作成
        result_folders = [os.path.join(folder_name,"3v"), os.path.join(folder_name,"4v")]
        for folder in result_folders:
            os.makedirs(folder, exist_ok=True)
            print(f"3v/4v Folder has been created.")   
            
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
        print('Complete Loading !')
        
        # テクスチャ情報を記述するインスタンス生成
        self.texture_num = len(self.model.f)
        self.mqo = CreateMQO(model_path, output_model_path, self.texture_num)
        
        
    def _initialize_texture_params(self):
        # テクスチャ関連のパラメータを初期化
        fnum = self.texture_num
        self.texture_camnum = np.full(fnum, 0)                          # テクスチャを割り当てるカメラ
        self.texture_dist = np.full(fnum, 100)                          # テクスチャカメラ間の距離
        self.texture_area = np.full(fnum, 0)                            # 面積
        self.texture_scale = np.full((fnum, 2), 1.0)                    # 拡大スケール
        self.texture_R = np.zeros((fnum, 3, 3), dtype=np.float64)       # 回転行列
        self.texture_V = np.empty(fnum, dtype=object)                   # 頂点の座標(可変長)
        for i in range(fnum):
            self.texture_V[i] = np.zeros((0, 3), dtype=np.float64) 
        self.texture_brightness = np.empty(fnum, dtype=object)          # テクスチャの明るさ(先頭が採用するテクスチャの明るさ)
        
        
    def _initialize_camera_params(self):
        # キャリブレーションデータの読み込み(11008*5504)
        self.calib_param_mat = np.loadtxt("camera_calibration/mtx.txt",delimiter=",")
        self.param_mat = np.loadtxt("camera_calibration/mtx_true.txt",delimiter=",")
        self.ufocus = self.calib_param_mat[0][0]
        self.vfocus = self.calib_param_mat[1][1]
        self.focus = (self.ufocus + self.vfocus)*0.5
        
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
        # 全方位画像読み込み
        pimage_paths, pimages, R_proj_list = self.read_eimage(eimage_path)
        # カメラパラメータ推定
        self.estimate_camera_params(pimage_paths, pimages, R_proj_list)
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
        self.pimg_count = 0
        pimage_list = []
        R_proj_list = []
        pimage_paths = []
        # 初期回転行列
        R0 = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float64)
        
        # 全方位画像ファイル名からカメラ番号を取得
        eimage_filename = os.path.splitext(os.path.basename(eimage_path))[0]
        self.cam_num = int(eimage_filename.split('_')[-1])
        
        # 対応するフォルダを作成
        folder_name = os.path.join('pose_estimation','camera_data','{}'.format(eimage_filename))
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Folder '{folder_name}' has been created.")
        
        # 全方位画像読み込み、全方位画像パラメータ保存
        self.eimage = cv2.imread(eimage_path)
        self.eimg_count += 1
        self.ewidth, self.eheight = self.eimage.shape[1], self.eimage.shape[0]
        
        # 透視投影画像パラメータ保存
        W_angle, H_angle = 90, 90
        self.pwidth = int(2 * math.tan((W_angle * np.pi/180.0)/2) * (self.ewidth/(2*np.pi)))
        self.pheight = int(2 * math.tan((H_angle * np.pi/180.0)/2) * (self.eheight/np.pi))
        self.angle_param_u = self.pwidth * np.pi / self.ewidth
        self.angle_param_v = 0.5 * self.pheight * np.pi / self.eheight
        
        # 透視投影マップ生成
        self.map = E2P(self.ewidth, self.eheight, self.pwidth, self.pheight)
        
        # 使用する透視投影画像の角度(self.R_optionのindex)を設定
        R_proj_index_filename = os.path.join(folder_name, 'use_R_index.txt')
        # ファイルの存在を確認して読み込む
        try:
            R_proj_index = np.loadtxt(R_proj_index_filename, delimiter=',', dtype=int)
            if R_proj_index.ndim == 0:  # スカラー（0次元）なら1次元配列に変換
                R_proj_index = np.array([R_proj_index])
        except FileNotFoundError:
            print(f"{R_proj_index_filename} が見つかりません。新しく作成します。")
            with open(R_proj_index_filename, 'w') as f:
                f.write("0\n")
            R_proj_index = np.array([0], dtype=int)
        
        for index in R_proj_index:
            # 仮想カメラの向き(透視投影画像の回転行列)
            R = np.dot(R0, self.R_proj_option[index].T)
            # マップ生成
            self.map.generate_map(self.angle_param_u, self.angle_param_v, R, scale = 1.0)
            pimage = self.map.generate_image(self.eimage)
            # カメラ番号ごとにディレクトリ生成
            subfolder_name = os.path.join(folder_name, 'cam_{}'.format(index))
            if not os.path.exists(subfolder_name):
                os.makedirs(subfolder_name)
                print(f"Folder '{subfolder_name}' has been created.")
            # 透視投影画像の保存
            filename = os.path.join(subfolder_name, "pimage_{}.png".format(index))
            cv2.imwrite(filename, pimage)
            
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
    # - self.use_truedata = 1 の場合は、推定を行わず実測値を使用する
    #================================================
    
    def estimate_camera_params(self, pimage_paths, pimages, R_proj_list):
        # カメラパラメータの測定値読み込み
        dir_name = os.path.dirname(pimage_paths[0].rstrip("/"))
        true_data_filename = os.path.join(dir_name, "param_true.dat")
        try:
            true_data = np.loadtxt(true_data_filename)
        except FileNotFoundError:
            print(f"{true_data_filename} が見つかりません。測定値を使用せずに実行します。")
            self.use_truedata = False
            true_data = None

        # 測定値を使用(推定値は使用しない)
        if self.use_truedata:
            print("カメラパラメータ推定を行わず、代わりに測定値を使用します。")
            self.R_cam = true_data[:, :3]
            self.t_cam = true_data[:, 3].flatten()
      
        else:
            # 3Dモデルと画像を対応付け
            datafolder = pnpl.generate_cam_data(pimage_paths, self.focus, pimages, R_proj_list)
            if datafolder == -1:
                raise ValueError("2D-3D対応付け失敗")

            # カメラの外部パラメータを推定
            flag, R, t = pnpl.PnPL_noError(datafolder, true_data)
            if flag == 1:
                raise ValueError("カメラパラメータ計算失敗")
            else:
                self.R_cam = R
                self.t_cam = t.flatten()
                
    
    #================================================
    # テクスチャのパラメータ計算
    # culc_texture_params(self)
    #
    # 返り値
    # - success : 正しく計算できた場合1
    #
    # 処理概要
    # - メッシュの座標を変換し、テクスチャのパラメータを計算
    # - カメラ番号、メッシュの座標、メッシュ重心を向く回転行列、カメラ中心までの距離、面積を計算
    # - テクスチャの明るさを調整するため、テクスチャの輝度と同一テクスチャの平均輝度を計算
    #   
    # 使用するメソッド
    #   self.transformation(vertices)
    #       - メッシュ頂点の世界座標系 v_world をカメラ座標系、スクリーン座標系に変換する
    #================================================   
     
    def culc_texture_params(self):
        m = self.model
        scale = np.array([1.0, 1.0])
        
        for i, face in enumerate(self.model.f):
            # メッシュの頂点座標を取得
            v_world = np.array([[m.x[face[idx]], m.y[face[idx]], m.z[face[idx]]] 
                                for idx in range(len(face))])
                
            # メッシュの座標変換
            flag, R_proj, v_proj, dist, area = self.transformation(v_world, scale, i)
            if flag == 1:
                continue
            
            # 三角形判定
            is_triangle = (len(face) == 3)
            is_rectangle = (len(face) == 4)
            # メッシュが透視投影画像内に収まっているかを判定
            is_within_image_bounds = (np.all((v_proj[:, 0] <= self.pwidth) & (v_proj[:, 0] >= 0)) and 
                                    np.all((v_proj[:, 1] <= self.pheight) & (v_proj[:, 1] >= 0)))
            
            if is_triangle and is_within_image_bounds:
                if self.use_adjust_texture_brightness:
                    # 平均輝度を計算
                    brightness = self.culc_brightness(R_proj, v_proj)
                    if self.texture_brightness[i] is not None:
                        # すでに輝度が格納されている場合、それに新しい輝度を追加する
                        if self.texture_dist[i] > dist:
                            # 先頭に追加
                            self.texture_brightness[i] = np.vstack([brightness, self.texture_brightness[i]])
                        else:
                            self.texture_brightness[i] = np.vstack([self.texture_brightness[i], brightness])
                    else:
                        # 初回の場合はそのまま格納
                        self.texture_brightness[i] = brightness
                
            elif is_rectangle:
                # 各頂点のx、y座標から重心までの距離を計算
                image_center = np.array([self.pwidth / 2, self.pheight / 2])
                x_distances = np.max(np.abs(v_proj[:, 0] - image_center[0]))
                y_distances = np.max(np.abs(v_proj[:, 1] - image_center[1])) 
                # 画像の幅と高さをvの座標の範囲に基づいて計算
                new_width = int(np.ceil(2*x_distances)) 
                new_height = int(np.ceil(2*y_distances)) 
                # 画像のスケール、マージンを計算
                scalex = max(1.0, new_width / self.pwidth)
                scaley = max(1.0, new_height / self.pheight)
                scale = np.array([scalex, scaley])
                # メッシュの座標変換
                flag, R_proj, v_proj, dist, area = self.transformation(v_world, scale)

            # 各種テクスチャ情報を更新
            if self.texture_dist[i] > dist:
                self.texture_camnum[i] = self.cam_num
                self.texture_dist[i] = dist
                self.texture_area[i] = area
                self.texture_R[i] = R_proj
                self.texture_V[i] = v_proj
                self.texture_scale[i] = scale
                
          
                    
    #================================================
    # テクスチャの輝度を計算
    # culc_brightness(self, R)
    #
    # 引数
    #   R_proj : 透視投影画像の視線方向の回転行列
    #   v_proj : スクリーンに投影された座標
    #
    # 処理概要
    # - 透視投影画像を生成し、画像b,g,rの平均輝度を計算する
    # 
    #================================================   
     
    def culc_brightness(self, R_proj, v_proj):
        self.map.generate_map(self.angle_param_u, self.angle_param_v, R_proj, 1.0)
        img = self.map.generate_image(self.eimage)
        self.pimg_count += 1

        # テクスチャとして用いる領域内のピクセルを抽出
        mask = np.zeros_like(img[:, :, 0], dtype=np.uint8)
        v_2d = v_proj[:, :2].astype(np.int32)
        cv2.fillConvexPoly(mask, v_2d, 255)
        triangle_region = cv2.bitwise_and(img, img, mask=mask)
        
        # 領域内の平均輝度の計算
        mean_b = np.mean(triangle_region[:, :, 0][mask == 255])
        mean_g = np.mean(triangle_region[:, :, 1][mask == 255])
        mean_r = np.mean(triangle_region[:, :, 2][mask == 255])

        return np.array([mean_b, mean_g, mean_r])
    
    
    #================================================
    # 座標変換
    # flag, R, v_proj, dist, area = transformation(self, v_world, scale)
    # 
    # 引数
    #   v_world：世界座標系のメッシュの頂点座標
    #   scale : 透視投影画像生成時に用いるスケール
    #   i : テクスチャ番号
    #
    # 戻り値
    #   flag : 座標変換のフラグ(1:失敗)
    #   R_proj : メッシュ重心を向く回転行列
    #   v_proj : スクリーンに投影された座標
    #   dist : メッシュの重心からの距離
    #   area : メッシュの投影面積
    #
    # 処理概要
    #  - 世界座標系からカメラ座標系に変換し、さらにメッシュ重心を向くように視線変換する
    #================================================ 
    
    def transformation(self, v_world, scale, i):
        # しきい値
        thd_dist = 10
        thd_angle = 70
        
        # 初期化
        flag = 0
        R_proj = np.eye(3)

        # カメラ座標系に変換
        v_cam = np.array([np.dot(self.R_cam, v) + self.t_cam for v in v_world])
        
        # 三角形の重心を計算
        centroid = np.mean(v_cam, axis = 0)
        dist = np.linalg.norm(centroid)
        # メッシュの重心までの距離がしきい値以上のとき消去
        if dist > thd_dist:
            flag = 1
            print("texture({}) : over the threshold({}m)".format(i, dist))
            return flag, R_proj, v_world, dist, 0
        
        # メッシュを重心にシフト
        # v_cam_shifted = v_cam - centroid
        # A_cov = np.zeros((3, 3))
        # for v in v_cam_shifted:
        #     A_cov += np.outer(v, v) # 外積を累積して共分散行列を作成
        # # 法線ベクトルを計算   →　使用していない 
        # eigenvalues, eigenvectors = np.linalg.eig(A_cov) # 固有値計算
        # normal = eigenvectors[:, np.argmin(np.abs(eigenvalues))]
        # # 法線ベクトルの符号調整
        # if np.dot(centroid, normal) < 0.0:
        #     normal = -normal
            
        # 重心方向を向くような回転行列を計算
        v_x, v_y, v_z = centroid[0], centroid[1], centroid[2]
        angle = np.degrees(np.arctan2(v_y, np.sqrt(v_x**2 + v_z**2)))
        # 仰角が大きいテクスチャを消去
        if angle > thd_angle:
            flag = 1
            print("texture({}) : over the angle({}°)".format(i, angle))
            return flag, R_proj, v_world, dist, 0
            
        # Z軸を重心方向に合わせる(法線ベクトルを用いる時はここを変更)
        Z = centroid / np.linalg.norm(centroid)
        # 射影行列を計算
        P = np.eye(3) - np.outer(Z, Z)
        # 回転行列を計算
        y = np.dot(P, np.array([0, 1, 0]))
        if np.linalg.norm(y) < 1e-10:
            Y = np.array([0, 0, -1])
            X = np.array([1, 0, 0])
        else:
            Y = y / np.linalg.norm(y)
            X = np.cross(Y, Z)
        R_proj = np.column_stack((X, Y, Z))
        
        # 内部パラメータ行列にスケールを適用
        scaled_param_mat = self.param_mat.copy()
        scaled_param_mat[0, 2] *= scale[0]
        scaled_param_mat[1, 2] *= scale[1]
        
        # 三角形メッシュの座標を画像面に投影
        v_proj = np.zeros_like(v_cam)
        for idx, vertex in enumerate(v_cam):
            cam_coords = np.dot(R_proj.T, vertex) # メッシュ重心方向への回転を適用
            proj_coords = np.dot(scaled_param_mat, cam_coords) # 内部パラメータ行列を適用
            proj_coords[0] /= proj_coords[2]
            proj_coords[1] /= proj_coords[2]
            v_proj[idx] = proj_coords
            
        # 面積を求める
        # 面積計算（四角形の場合、三角形2つ分を合計）
        if len(v_proj) == 4:
            area = (
                0.5 * np.linalg.norm(np.cross(v_proj[1] - v_proj[0], v_proj[2] - v_proj[0])) +
                0.5 * np.linalg.norm(np.cross(v_proj[2] - v_proj[0], v_proj[3] - v_proj[0]))
            )
        else:
            area = 0.5 * np.linalg.norm(np.cross(v_proj[1] - v_proj[0], v_proj[2] - v_proj[0]))

        return flag, R_proj, v_proj, dist, area
    
                        
    #================================================
    # テクスチャを生成 
    # generate_texture(self, eimage_path)
    #
    # 引数
    #   eimage_path：全方位画像のリスト
    # 
    # 概要：得られたテクスチャ情報をもとに、全方位画像からテクスチャを生成する
    # 画像を開く回数を減らしたい
    # 焦点距離が変わってしまう、、、
    #================================================    
    
    def generate_texture(self, eimage_path):
        # 画像名の時はリストに変換
        if isinstance(eimage_path, str):
            eimage_path = [eimage_path]  
        # 画像を最初にすべて読み込む
        loaded_images = [cv2.imread(path) for path in eimage_path]
        
        # 全てのテクスチャ情報を更新
        for i in range(self.texture_num):
            # テクスチャ情報の読み込み
            camnum = self.texture_camnum[i]
            if len(loaded_images) > 1:
                eimage = loaded_images[camnum]
            else:
                eimage = loaded_images[0]
            vertices = np.array([[point[0], point[1]] for point in self.texture_V[i]], dtype=np.float32)
            vnum = len(self.texture_V[i])
            pwidth = int(np.ceil(self.pwidth * self.texture_scale[i][0]))
            pheight = int(np.ceil(self.pheight * self.texture_scale[i][1]))
            # 透視投影画像の生成
            R = self.texture_R[i]
            self.map.generate_map(self.angle_param_u, self.angle_param_v, R, scale = 1.0)
            
            # 3頂点のテクスチャの処理
            if vnum == 3:
                v_proj = vertices
                img_proj = self.map.generate_image(eimage)
                # テクスチャ画像輝度更新
                if self.use_adjust_texture_brightness:
                    if self.texture_brightness[i].ndim > 1:                                 # テクスチャが複数割り当てられている場合のみ
                        brightness = self.texture_brightness[i][0]                          # 採用されたテクスチャの輝度(先頭に存在する)
                        target_brightness = np.mean(self.texture_brightness[i], axis=0)     # 目標値はテクスチャの輝度平均
                        img_proj = self.apply_average_brightness(img_proj, brightness, target_brightness)
            
            # 4頂点のテクスチャの処理
            elif vnum == 4:
                # ホモグラフィー変換を実行
                v_proj = np.array([[0, 0],[pwidth, 0],[pwidth, pheight],[0, pheight]],dtype=np.float32)
                H, _ = cv2.findHomography(vertices, v_proj)
                
                # 透視投影画像のサイズを変更して生成
                angle_param_u = pwidth * np.pi / self.ewidth
                angle_param_v = 0.5 * pheight * np.pi / self.eheight
                map = E2P(self.ewidth, self.eheight, pwidth, pheight)
                map.generate_map(angle_param_u, angle_param_v, R, scale = 1.0)
                
                # 画像を変換
                img = map.generate_image(eimage)
                img_proj = cv2.warpPerspective(img, H, (pwidth, pheight))
            
            else:
                print(f"texture({i})：texture not generated")
                continue
                
            # テクスチャ画像保存
            image_with_vertices = self.draw_vertices_edge(img_proj, v_proj, camnum)
            texture_filename = os.path.join('texture_{}_{}.png'.format(camnum, i))
            subfolder = os.path.join('texture','{}v'.format(str(vnum)))
            filename = os.path.join('result', subfolder, texture_filename)
            cv2.imwrite(filename, image_with_vertices)
            
            # vは透視投影画像サイズで正規化
            v_proj[:, 0] /= pwidth
            v_proj[:, 1] /= pheight
            # 3次元モデル情報のアップデート
            self.mqo.update_mqo(v_proj, texture_filename, i, self.eimg_count)
            # ３次元モデルファイル書き込み
            self.mqo.write_mqo()
            # テクスチャ記述完了
            print(f"texture({i})：The texture was generated [{filename}]")


    #================================================
    # 平均輝度を画像に適用
    # image = apply_average_brightness(self, image, mean_brightness, target_brightness):
    # 
    # 引数
    #   image : 透視投影画像(テクスチャ)
    #   mean_brightness : 画像b,g,rの平均輝度
    #   target_brightness : 複数視点の画像b,g,rの平均輝度の平均
    # 
    # 返り値
    #   image : 輝度を調整した画像
    #================================================    
    
    def apply_average_brightness(self, image, mean_brightness, target_brightness):
        # 平均色を基に補正係数を計算
        mean_b, mean_g, mean_r = mean_brightness
        ref_b, ref_g, ref_r = target_brightness

        # チャンネルごとのスケール係数を計算
        b_scale = ref_b / mean_b if mean_b != 0 else 1.0
        g_scale = ref_g / mean_g if mean_g != 0 else 1.0
        r_scale = ref_r / mean_r if mean_r != 0 else 1.0

        # 各チャンネルにスケールを適用して輝度を調整
        image[:, :, 0] = cv2.convertScaleAbs(image[:, :, 0], alpha=b_scale)
        image[:, :, 1] = cv2.convertScaleAbs(image[:, :, 1], alpha=g_scale)
        image[:, :, 2] = cv2.convertScaleAbs(image[:, :, 2], alpha=r_scale)

        # 調整した画像を保存
        return image
    
    
    #================================================
    # 頂点、辺を画像に書き込み
    # draw_vertices_edge(self, img, v, camnum):
    #================================================    
    def draw_vertices_edge(self, img, v, camnum):
        if camnum == 0:
            color = (255, 0, 0)
        elif camnum == 1:
            color = (0, 255, 0)
        elif camnum == 2:
            color = (0, 0, 255)
        else:
            color = (0, 0, 0)
        for i in range(len(v)): 
            cv2.circle(img, (int(v[i][0]), int(v[i][1])), 1, color, 1)
            if i+1 < len(v):
                pt1 = int(v[i][0]), int(v[i][1])
                pt2 = int(v[i+1][0]), int(v[i+1][1])
            elif i+1 == len(v):
                pt1 = int(v[i][0]), int(v[i][1])
                pt2 = int(v[0][0]), int(v[0][1])
            cv2.line(img, pt1, pt2, color, thickness=1)
        return img