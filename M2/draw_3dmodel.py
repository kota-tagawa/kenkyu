import numpy as np
import cv2
import argparse
import os

def draw_3dmodel(img, R, t, K, points):
    # 指定された点を繋ぐ線(ファイル入力に対応させる)
    connections = [
        (0, 1), (1, 2),  # 1,2,3行目を繋ぐ線
        (3, 4),           # 4,5行目を繋ぐ線
        (5, 6), (6, 7), (7, 8),  # 6,7,8行目を繋ぐ線
        (8, 9)            # 9,10行目を繋ぐ線
    ]

    # Rt行列の作成
    Rt = np.array([[R[0][0], R[0][1], R[0][2], t[0]],
                   [R[1][0], R[1][1], R[1][2], t[1]],
                   [R[2][0], R[2][1], R[2][2], t[2]]])

    # 同次座標に変換
    points_hom = np.hstack((points, np.ones((points.shape[0], 1)))) 

    # カメラ座標系に変換し、画像座標へ投影
    img_points_3d = np.dot(Rt, points_hom.T)
    valid_indices = img_points_3d[2] > 0  # Zが正の点のみ選択
    img_points_3d = img_points_3d[:, valid_indices]

    img_points = np.dot(K, img_points_3d)
    img_points /= img_points[2]  # Z=1 に正規化
    img_points = img_points[:2].T.astype(int)
    
    for start, end in connections:
        # 2D画像上の点を取得
        pt1 = tuple(img_points[start])
        pt2 = tuple(img_points[end])

        # 線を描画
        cv2.line(img, pt1, pt2, (255, 0, 0), 2)  # 赤色で線を描画


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Project 3D coordinates onto the input image based on the camera parameters")
    parser.add_argument("image", help="Path to background image")
    parser.add_argument("points", help="Path to 3D point file")
    parser.add_argument("param", help="Path to External matrix file (R, t)")
    parser.add_argument("K", help="Path to Intrinsic matrix file (K)")
    parser.add_argument("--conection", help="Path to Connection file")

    args = parser.parse_args()

    # 背景画像の読み込み
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Image not found: {args.image}")
    
    # 点群ファイル(3d座標)の読み込み
    if not os.path.exists(args.point_file):
        raise FileNotFoundError(f"Point file not found: {args.point_file}")
    
    with open(args.point_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    points = np.array([list(map(float, line.split())) for line in lines]) # loadtxtで代用可能(エラーになる？)
    
    # 外部パラメータ R,t 読み込み
    Rt = np.loadtxt(args.param)  
    R = Rt[:, :3]
    t = Rt[:, 3]
    # 内部パラメータ K 読み込み
    K = np.loadtxt(args.K)

    draw_3dmodel(R, t, K, points)








