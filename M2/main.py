import numpy as np
import cv2
import argparse
import os
import glob
from PIL import Image

# 3dモデルロード
from mqoloader.loadmqo import LoadMQO

# resnet特徴量の事前計算
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.models import resnet50

# 特徴点マッチング
import match_feature as mf
#カメラデータ生成
from pose_estimation import generate_camdata as gc
# 自己位置推定
from pose_estimation import pnpl
# 特徴点を画像上に記述
import draw_3dmodel as dm




# 事前学習済みResNetをロード
resnet = models.resnet50(pretrained=True)
resnet.eval()  # 評価モードに設定
# 特徴量抽出用に最終分類層を除外
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])

def load_3Dmodel(model_path):
    # 3Dモデルの読み込み
    print('Loading %s ...' % model_path)
    use_normal = False
    model = LoadMQO(model_path, 0.0, use_normal)
    print('Complete Loading !')
    return model

def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size 
    
def texture_to_world(points_2D, db_width, db_height, model, number):
    face = model.f[number]
    v0 = np.array([model.x[face[0]], model.y[face[0]], model.z[face[0]]])
    v1 = np.array([model.x[face[1]], model.y[face[1]], model.z[face[1]]])
    v3 = np.array([model.x[face[3]], model.y[face[3]], model.z[face[3]]])

    # 世界座標系の基底ベクトル
    vec1_world = v1 - v0  # 横方向ベクトル
    vec2_world = v3 - v0  # 縦方向ベクトル

    # 基底ベクトル方向に沿った割合
    u = points_2D[:, 0] / db_width
    v = points_2D[:, 1] / db_height

    # 基底ベクトルの線形結合によりベクトル空間の座標を導出
    points_3D = v0 + np.outer(u, vec1_world) + np.outer(v, vec2_world)

    return points_3D
    
def main(target_image_path, model_path, texture_paths):

    # 3dモデル読み込み
    model = load_3Dmodel(model_path)

    # 特徴点の読み込み
    print("Extracting features from database images...")
    database_features = []

    # 特徴量の事前計算（ResNetを使って各データベース画像から抽出）
    for db_path in texture_paths:
        features = mf.extract_features(db_path, resnet)
        database_features.append(features)
    database_features = np.array(database_features).astype('float32')

    # coarse-to-fineで画像検索
    dbbest_image_path, target, db = mf.ResNettoAKAZE(
        target_image_path, 
        texture_paths, 
        database_features, 
        resnet)

    # target, db の画像サイズを求める
    target_size = get_image_size(target_image_path)
    db_size = get_image_size(dbbest_image_path)

    # targetの座標群を画像中心が原点になるようにシフト
    point_2D = target - np.array([target_size[0]/2, target_size[1]/2])

    # dbの座標を3D平面上の座標に変換
    filename = os.path.basename(dbbest_image_path)
    number = int(filename.split('_')[-1].split('.')[0])
    point_3D = texture_to_world(db, db_size[0], db_size[1], model, number)
    
    # カメラデータ生成
    save_folder = os.path.join("pose_estimation", "result")
    np.savetxt(os.path.join(save_folder,"2Ddata.dat"), point_2D)
    np.savetxt(os.path.join(save_folder,"3Ddata.dat"), point_3D)
    cammat = np.loadtxt("pose_estimation/mtx.txt",delimiter=",")
    focus = (cammat[0][0] + cammat[1][1])*0.5
    R_proj = np.array([[1,0,0],[0,1,0],[0,0,1]],dtype=np.float64)
    datafolder = gc.generate_camdata(focus, [save_folder], [R_proj])
    if datafolder == -1:
        raise ValueError("2D-3D対応付け失敗")

    # pnpl実行
    true_data_filename = "pose_estimation/true_data.txt"
    try:
        true_data = np.loadtxt(true_data_filename)
    except FileNotFoundError:
        print(f"{true_data_filename} が見つかりません。測定値を使用せずに実行します。")
        true_data = None
    flag, R, t = pnpl.PnPL_noError(datafolder, true_data)
    if flag == 1:
        raise ValueError("カメラパラメータ計算失敗")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Matching input images with 3D models and automatically estimating self-position")
    parser.add_argument("image", help="Path to input image")
    parser.add_argument("model", help="Path to 3D model directory")

    args = parser.parse_args()

    # 画像パスの読み込み
    image_path = args.image
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"Image file not found: {args.image}")
    
    # モデルディレクトリ存在チェック
    model_dir = args.model
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"3D model directory not found: {model_dir}")
    
    # .mqoファイルの存在チェック
    mqo_paths = [f for f in os.listdir(model_dir) if f.lower().endswith(".mqo")]
    if not mqo_paths:
        raise FileNotFoundError(f"No .mqo files found in directory: {model_dir}")
    mqo_path = os.path.join(model_dir,mqo_paths[0])
    
    # テクスチャ画像パスの読み込み
    texture_dir = os.path.join(model_dir, "texture", "4v")
    texture_paths = sorted(glob.glob(os.path.join(texture_dir, "*.png")))
    if len(texture_paths) == 0:
        raise FileNotFoundError(f"No database images found in {texture_paths}")

    main(image_path, mqo_path, texture_paths)

