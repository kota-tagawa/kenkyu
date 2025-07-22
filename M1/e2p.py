# e2p.py
# editor : tagawa kota, sugaya yasuyuki
# last edited : 2025/4/17
# Generate perspective projection images in any viewing direction from omnidirectional images
# referance：菅谷 保之「全方位画像からの透視投影画像の生成: 決定版」

import math
import numpy as np
import cv2
import argparse
import os
import sys
import glob

class E2P:
    def __init__(self, _src_w, _src_h, _dst_w, _dst_h):
        # 画像の大きさ
        self.src_w = _src_w
        self.src_h = _src_h
        self.dst_w = _dst_w
        self.dst_h = _dst_h
    
        # remap用変数
        self.map_u = np.zeros((_dst_h, _dst_w), dtype=np.float32)
        self.map_v = np.zeros((_dst_h, _dst_w), dtype=np.float32)

        # 内挿関数指定
        self.interp_method = cv2.INTER_LINEAR

    # X軸周りの回転行列
    def rotation_x(self, angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        R = np.array([[1.0, 0.0, 0.0],
                    [0.0, cos_a, -sin_a],
                      [0.0, sin_a, cos_a]], dtype=np.float64)
        return R

    # Y軸周りの回転行列
    def rotation_y(self, angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        R = np.array([[cos_a, 0.0, sin_a],
                      [0.0, 1.0, 0.0],
                      [-sin_a, 0.0, cos_a]], dtype=np.float64)
        return R

    # 任意の軸周りの回転行列
    def rotation_by_axis(self, n, angle):
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        l_cos_a = 1.0 - cos_a
        R = np.array([[cos_a + n[0] * n[0] * l_cos_a,
                       n[0] * n[1] * l_cos_a - n[2] * sin_a,
                       n[2] * n[0] * l_cos_a + n[1] * sin_a],
                      [n[0] * n[1] * l_cos_a + n[2] * sin_a,
                       cos_a + n[1] * n[1] * l_cos_a,
                       n[1] * n[2] * l_cos_a - n[0] * sin_a],
                      [n[2] * n[0] * l_cos_a - n[1] * sin_a,
                       n[1] * n[2] * l_cos_a + n[0] * sin_a,
                       cos_a + n[2] * n[2] * l_cos_a]], dtype=np.float64)
        return R

    # 回転行列の計算
    def culc_rotation_matrix(self, angle_u, angle_v, angle_i):

        # すべての角度が0の場合は単位行列を返す
        if all(x == 0.0 for x in (angle_u, angle_v, angle_i)):
            R = np.array([[1,0,0], [0,1,0], [0,0,1]], dtype=np.float64) 
            return R
        
        # ラジアンに変換
        angle_u = np.radians(angle_u)
        angle_v = np.radians(angle_v)
        angle_i = np.radians(angle_i)
        # X軸周りとY軸周りの回転
        R = np.dot(self.rotation_y(angle_u), self.rotation_x(angle_v))
        # 任意の軸周りの回転
        axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        Ri = self.rotation_by_axis(np.dot(R, axis), angle_i)
        R = np.dot(Ri, R)
        return R
    
    # 画像生成用のマップ作成関数
    def generate_map(self,
                     angle_param_u,
                     angle_param_v,
                     R,
                     scale):
            
        # パラメータ
        oWh = 0.5 * self.dst_w
        oHh = 0.5 * self.dst_h
        step_tan_theta = scale * angle_param_u / oWh
        step_tan_phi = scale * angle_param_v / oHh
            
        # 一括計算用の出力画像の画素座標データ
        u = np.arange(0, self.dst_w, 1)
        v = np.arange(0, self.dst_h, 1)
        dst_u, dst_v = np.meshgrid(u, v)
        
        # 視線ベクトルを計算
        x = step_tan_theta * (dst_u - oWh)
        y = step_tan_phi * (dst_v - oHh)
        z = np.ones((self.dst_h, self.dst_w))
        
        # 回転行列で回転
        Xx = R[0][0] * x + R[0][1] * y + R[0][2] * z
        Xy = R[1][0] * x + R[1][1] * y + R[1][2] * z
        Xz = R[2][0] * x + R[2][1] * y + R[2][2] * z
        
        # 視線ベクトルから角度を計算
        theta = np.arctan2(Xx, Xz)
        phi = -np.arctan2(Xy, np.sqrt(Xx**2 + Xz**2))
        
        # 角度から入力画像の座標を計算
        self.map_u = (0.5 * (theta + np.pi) * self.src_w / np.pi).astype(np.float32)
        self.map_v = ((0.5 * np.pi - phi) * self.src_h / np.pi).astype(np.float32)

    # 画像生成
    def generate_image(self, src_img):
        return cv2.remap(src_img, self.map_u, self.map_v, self.interp_method)
    

def main(w_angle, h_angle, theta, phi, psi, input_path): 

    # 入力がファイルかフォルダか判定
    if os.path.isfile(input_path):
        if input_path.lower().endswith(('.jpg', '.jpeg', '.png')):
            eimage_files = [input_path]
            is_single_file = True
        else:
            print(f"対応していないファイル形式: {input_path}")
            sys.exit(1)
    elif os.path.isdir(input_path):
        eimage_files = glob.glob(os.path.join(input_path, '*.jpg')) + \
                       glob.glob(os.path.join(input_path, '*.jpeg')) + \
                       glob.glob(os.path.join(input_path, '*.png'))
        if not eimage_files:
            print(f"入力フォルダ '{input_path}' に画像ファイルが見つかりません。")
            sys.exit(1)
        is_single_file = False
        output_folder = 'pimages'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    else:
        print(f"入力パス '{input_path}' が存在しません。")
        sys.exit(1)

    for image_path in eimage_files:
        print("loading..." + image_path)
        eimage = cv2.imread(image_path)
        ewidth, eheight = eimage.shape[1], eimage.shape[0]

        pwidth = int(2 * math.tan((w_angle * np.pi/180.0)/2) * (ewidth/(2*np.pi)))
        pheight = int(2 * math.tan((h_angle * np.pi/180.0)/2) * (eheight/np.pi))
        angle_param_u = pwidth * np.pi / ewidth
        angle_param_v = 0.5 * pheight * np.pi / eheight

        e2p = E2P(ewidth, eheight, pwidth, pheight)
        R = e2p.culc_rotation_matrix(theta, phi, psi)
        e2p.generate_map(angle_param_u, angle_param_v, R, scale=1.0)
        pimage = e2p.generate_image(eimage)

        basename = os.path.basename(image_path)
        name, ext = os.path.splitext(basename)

        if is_single_file:
            # 入力がファイルの場合はそのファイル名を元に出力
            output_path = f"{name}_{int(theta)}{ext}"
        else:
            output_path = os.path.join(output_folder, basename)

        cv2.imwrite(output_path, pimage)
        print(f"保存しました: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='透視投影画像の画角と入力指定')
    parser.add_argument('w_angle', type=int, help='画角W')
    parser.add_argument('h_angle', type=int, help='画角H')
    parser.add_argument('--theta', type=float, default=0.0, help='視線方向のヨー角（左右）')
    parser.add_argument('--phi', type=float, default=0.0, help='視線方向のピッチ角（上下）')
    parser.add_argument('--psi', type=float, default=0.0, help='視線方向のロール角（傾き）')
    parser.add_argument('--input', type=str, default='eimages', help='入力画像ファイルまたはフォルダのパス')

    args = parser.parse_args()

    if args.w_angle <= 0 or args.h_angle <= 0:
        print("画角は正の整数で指定してください。")
        sys.exit(1)

    main(args.w_angle, args.h_angle, args.theta, args.phi, args.psi, args.input)