## 概要
全方位カメラのキャリブレーションを行う

## 動作環境
- OS : Ubuntu 20.04 (LTS)
- アーキテクチャ : x86_64 (64bit)
- Python : 3.8.10

## 実行方法(e2p.py)
- $python e2p.py 画角Width 画角Height (オプション回転角--theta --phi --psi)
- $python camera_calibration.py チェッカーボードの横方向の交点の数 チェッカーボードの縦方向の交点の数 マスの大きさ（cm）

## フォルダ
- eimages : 入力 全方位画像
- pimages : 出力 透視投影画像
- datetime.now().strftime('%Y%m%d') : チェッカーボード結果保存用

## 変更点
20250412 degree→radianに変換する処理を追加し、任意の角度の回転行列を正しく計算できるように変更