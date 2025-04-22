# main.py
# editor : tagawa kota, sugano yasuyuki
# last edited : 2025/3/4 
# Input 360-degree-image and 3dD model and run main loop of app.

# アプリケーションの本体
import application
# 汎用ライブラリ(以下に追加)
import os
import argparse
import cv2
import numpy as np
import sys

#================================================
# メイン関数
# main(eimages, model)
# 
# 引数:
#   eimages : 全方位画像（equirectangular image）のファイルまたは、画像を含むディレクトリのパス
#   model   : 3Dモデルのファイルパス
#
# 処理概要:
#   - 指定された全方位画像と3Dモデルを入力として受け取る
#   - 画像が単一ファイルの場合は、その画像を処理し、テクスチャを生成する
#   - 画像が格納されたディレクトリの場合は、対象画像を順番に処理し、最終的にテクスチャを生成する
#================================================    
def main(eimage_path, model_path):
        
    if eimage_path and model_path:
        # Applicationクラスインスタンス生成
        app = application.Application(model_path)
        
        # 画像ファイルならリスト化、ディレクトリなら全方位画像のリストを作成
        if os.path.isfile(eimage_path):
            eimage_path_list = [eimage_path]
        
        elif os.path.isdir(eimage_path):
            files = os.listdir(eimage_path)
            # ディレクトリに全方位画像が含まれるかどうか確認
            eimage_path_list = sorted(os.path.join(eimage_path, file) for file in files if file.startswith("eimage"))
            if not eimage_path_list:
                print(f"ディレクトリ{eimage_path}に全方位画像が含まれていません")
                return
        
        # 画像処理(全方位画像読み込み、カメラパラメータ推定、テクスチャ生成)
        for eimage_path in eimage_path_list:
            app.mainloop(eimage_path)
        
        # テクスチャを編集
        app.generate_texture(eimage_path_list)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Assign texture from a 360-degree image to a 3D model.")
    parser.add_argument("eimage", help="Path to an equirectangular image file or directory")
    parser.add_argument("model", help="Path to a 3D model file")
    args = parser.parse_args()

    # 画像ディレクトリ存在チェック
    eimages_path = args.eimage
    if not os.path.isdir(eimages_path):
        raise FileNotFoundError(f"eimage directory not found: {eimages_path}")

    # モデルディレクトリ存在チェック
    model_dir = args.model
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"3D model directory not found: {model_dir}")
    
    # .mqoファイルの存在チェック
    mqo_paths = [f for f in os.listdir(model_dir) if f.lower().endswith(".mqo")]
    if not mqo_paths:
        raise FileNotFoundError(f"No .mqo files found in directory: {model_dir}")
    print(mqo_paths)
    model_path = os.path.join(model_dir,mqo_paths[1])
    
    main(eimages_path, model_path)