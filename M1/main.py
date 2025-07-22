# main.py
# editor : tagawa kota, sugano yasuyuki
# last edited : 2025/3/4 
# Input 360-degree-image and 3D model and run main loop of app.

# アプリケーションの本体
import application
# 汎用ライブラリ(以下に追加)
import os
import argparse


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
def main(eimage_path, model_path, config):
        
    # Applicationクラスインスタンス生成
    app = application.Application(model_path, config)
    
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
    
    # 必須項目
    parser.add_argument("eimage", help="Path to an equirectangular image file or directory")
    parser.add_argument("model", help="Path to a .mqo 3D model file")

    # オプション
    parser.add_argument('--estimate_camparam', action='store_true', help='Run in a mode that only estimates camera parameters without generating textures')

    args = parser.parse_args()

    # eimage の存在チェック
    eimages_path = args.eimage
    if not os.path.exists(eimages_path):
        raise FileNotFoundError(f"eimage directory or file not found: {eimages_path}")

    # model ファイルの存在と拡張子チェック
    model_path = args.model
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"3D model file not found: {model_path}")
    if not model_path.lower().endswith(".mqo"):
        raise ValueError(f"The specified model file is not a .mqo file: {model_path}")

    # オプション引数を config 辞書に抽出
    required_keys = ['eimage', 'model']
    config = {k: v for k, v in vars(args).items() if k not in required_keys}

    main(eimages_path, model_path, config)