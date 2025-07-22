# 全方位カメラ位置姿勢推定＆3次元モデルテクスチャ割り当て

## 概要

本プロジェクトは、全方位画像からカメラの位置および姿勢を推定し、その推定結果をもとに3次元モデルへテクスチャを割り当てるツールです。

---

## 動作環境

- OS：Ubuntu 20.04 LTS
- アーキテクチャ：x86_64（64bit）
- Python：3.8.10

---

## 必要ライブラリ

以下のパッケージをインストールしてください（必要に応じてrequirements.txtをご用意ください）：
pip install -r requirements.txt

---

## 実行方法

以下のコマンドで処理を実行できます：
python3 main.py [全方位画像 or 画像ディレクトリ] [3次元モデルファイル] [オプション]

### 実行例：

python3 main.py input model/C5F.mqo

---

## 入力

- 全方位画像または画像ディレクトリ：input/
- 3次元モデルファイル：model/C5F.mqo

---

## 出力

- テクスチャが割り当てられた3次元モデル：result/C5F_edited.mqo
- 生成されたテクスチャ画像群：result/texture/

---

## オプション

- --estimate_camparam: テクスチャの取得を行わず、カメラパラメータ推定のみのモードで実行します

---

## ディレクトリ構成

project_root
    ├─input                     # 全方位画像の入力ディレクトリ
    │
    ├─model                     
    │  ├─C5F.mqo                # 使用する3Dモデル
    │  ├─texture_exclusion.dat  # テクスチャ除外用ファイル
    │  └─texture                # テクスチャ保存用
    │
    ├─mqoloader                 # 3Dモデル読み込み用プログラム
    │
    ├─pose_estimation           # 自己位置推定用プログラム
    │  ├─mtx.txt                # 仮想カメラの内部パラメータ
    │  └─camera_data            # カメラ別データ
    │     ├─eimage_0
    │        ├─use_R_index.txt  # 仮想カメラの角度設定(application.R_proj_optionのインデックス)
    │        ├─cam_0
    │           ├─3Ddata.dat    # 対応付けする3次元座標
    │           └─2Ddata.dat    # 対応付けする2次元座標(透視投影画像から選択する)
    │
    ├─result                    # 結果保存用
    │
    ├─application.py            # アプリのメインループ。360度画像から取得した情報をもとに、3Dモデルにテクスチャを適用する。
    ├─main.py                   # 360度画像と3Dモデルを入力し、アプリのメインループを実行する。
    ├─e2p.py                    # 全方位画像から、任意の視点方向に対する透視投影画像を生成する。
    ├─create_mqo.py             # 3DモデルファイルにUV座標とテクスチャ画像を割り当てます。
    └─README.md

---

## 著者

田川 幸汰