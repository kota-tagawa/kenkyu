## 概要
全方位カメラの位置と姿勢を推定し、推定結果をもとに3次元モデルにテクスチャを割り当てる

## 動作環境
- OS : Ubuntu 20.04 (LTS)
- アーキテクチャ : x86_64 (64bit)
- Python : 3.8.10

## 実行方法
- $python3 main.py -e (全方位画像or全方位画像ディレクトリ) -m (3次元モデル)
## 入力
- 全方位画像ディレクトリ：input
- モデル：model/C5F.mqo

## 出力
- モデル：result/C5F_edited.mqo
- テクスチャ：result/texture

## モード
- Application.use_truedata
    True：カメラパラメータをパラメータファイルから読み込み
    False：カメラパラメータを2次元座標、3次元座標対応から推定

- Application.use_adjust_texture_brightness
    True：平均輝度を用いた明るさ調整を実行(実行時間がかかる)
    False：明るさ調整を実行しない