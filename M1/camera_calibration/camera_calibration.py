# camera_calibration.py
# editor : tagawa kota
# last edited : 2025/4/17
# Calibrating the camera using a checkerboard.
# referance：https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html

import argparse
import numpy as np
import cv2
import glob
from datetime import datetime
import os
import sys

def main(num_cols, num_rows, square_size_cm):
    # 終了条件
    termination_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # チェッカーボード上の3次元実座標を生成
    # 例 (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((num_rows*num_cols,3), np.float32)
    objp[:,:2] = np.mgrid[0:num_rows,0:num_cols].T.reshape(-1,2)
    objp *= square_size_cm

    # 実空間（3D）と画像空間（2D）の対応点を格納するリスト
    objpoints = [] 
    imgpoints = [] 

    # 結果保存用フォルダを作成
    output_folder = datetime.now().strftime('%Y%m%d')
    os.makedirs(output_folder, exist_ok=True)

    # キャリブレーション用の画像を読み込み
    image_files = glob.glob('pimages/*.jpg')

    for image_path in image_files:
        # 画像読み込みとグレースケール変換
        print("loading..." + image_path)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # チェッカーボードのコーナー検出
        found, corners = cv2.findChessboardCorners(gray, (num_rows, num_cols), None)

        if found:
            # 検出されたコーナーをより高精度に補正
            refined_corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), termination_criteria)

            # 対応点を保存
            objpoints.append(objp)
            imgpoints.append(refined_corners)

            # コーナーを描画して保存
            image_with_corners = cv2.drawChessboardCorners(image, (num_rows, num_cols), refined_corners, ret)
            image_name = os.path.basename(image_path)
            save_path = os.path.join(output_folder, image_name)
            cv2.imwrite(save_path, image_with_corners)

        else:
            print("チェッカーボードが検出できませんでした。")
            sys.exit(1)

    # 内部パラメータを計算
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    tvecs = np.array(tvecs)
    # 計算結果を表示
    print(f"RMS 誤差: {ret}")
    print(f"カメラ行列 (mtx):\n{mtx}")
    print(f"歪み係数 (dist): {dist.ravel()}")

    # カメラの世界座標系における位置を推定
    for i in range(len(rvecs)):
        rvec = np.array(rvecs[i])
        tvec = np.array(tvecs[i])
        R, _ =cv2.Rodrigues(rvec)
        C = -np.dot(R.T,tvec)

        print(f"カメラ {i+1} の回転行列:\n{R}")
        print(f"カメラ {i+1} の位置ベクトル:\n{C}")

    # 計算結果を保存
    np.savetxt(os.path.join(output_folder, "ret.txt"), np.array([ret]), delimiter =',',fmt="%0.14f")
    np.savetxt(os.path.join(output_folder, "mtx.txt"), mtx, delimiter =',',fmt="%0.14f")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='チェッカーボードパターンの情報を指定してください。'
    )
    parser.add_argument('cols', type=int, default=10, help='チェッカーボードの横方向の交点の数')
    parser.add_argument('rows', type=int, default=7, help='チェッカーボードの縦方向の交点の数')
    parser.add_argument('size', type=float, default=5.0, help='マスの大きさ（cm）')

    args = parser.parse_args()
    main(args.cols, args.rows, args.size)