import numpy as np

#
# 3次元点を行列に変換する関数
# 
# @input  P : 3次元点のリスト
# @output A : 3次元点を行列として表現したリスト 
#
def calc_A (P):
    A = []
    for p in P:
        A_ = np.array([
            [p[0], p[1], p[2], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],    
            [0.0, 0.0, 0.0, p[0], p[1], p[2], 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, p[0], p[1], p[2]]
        ], dtype=np.float64)
        A.append (A_)
    return A

#
# ベクトルの平均を計算する関数
#
# @input  P        : ベクトルのリスト
# @output centroid : ベクトルの平均  
#
def vector_mean (P):
    centroid = np.zeros(3)
    for p in P:
        centroid += p
    centroid /= len(P)
    return centroid

#
# 並進ベクトルを計算するための行列を計算する関数
#
# @input  P       : 3次元点のリスト
# @input  V       : 射影行列のリスト
# @output t_param : 並進ベクトルを計算するための行列
#
def calc_t_param (A, V):
    I_V = np.zeros((3, 3))
    I_V_A = np.zeros((3, 9))
    I = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    for a, v in zip (A, V):
        I_V += (I - v)
        I_V_A += (I - v) @ a
    t_param = -np.linalg.inv(I_V) @ I_V_A
    return t_param

#
# 並進ベクトルを計算する関数
#
# @input  P : 3次元点のリスト
# @input  R : 回転行列
# @output t : 並進ベクトル
#
def calc_t (t_param, r):
    t = t_param @ r
    return t

#
# 目的関数の値を計算する関数
#
# @input  P : 3次元点のリスト
# @input  V : 射影行列のリスト
# @input  R : 回転行列
# @input  t : 並進ベクトル
# @output J : 目的関数の値
#
def calc_J (P, V, R, t):
    J = 0.0
    I = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    for p, v in zip(P, V):
        val = np.linalg.norm ((I - v) @ (R @ p + t))
        J += val * val
    return J / len(P)

#
# カメラパラメータを計算する関数
#
# @input  P   : 3次元点のリスト
# @input  V   : 射影行列のリスト
# @input  R   : 回転行列の初期値
# @input  eps : 収束判定のしきい値
# @output J   : 目的関数の値
# @output R   : 回転行列
# @output t   : 並進ベクトル
#
def OIPnPL (P, V, R, eps):

    # 計算の簡単化のために変数を変換
    A = calc_A (P)
    r = R.reshape (9)

    # 入力3次元点をその重心が原点に位置するように平行移動
    Pc = vector_mean (P)
    P_ = P - Pc

    # 回転行列から並進ベクトルを計算
    t_param = calc_t_param (A, V)
    t = calc_t (t_param, r)

    # 目的関数の値を計算
    J = calc_J (P, V, R, t)

    # 反復回数の初期化
    count = 0

    # 反復計算
    while True:

        # 入力3次元点をカメラ座標系に変換して射影行列を用いて射影
        Q = []
        for p, v in zip(P, V):
            Q.append (v @ (R @ p + t))

        # 3次元点Qをその重心が原点に位置するように平行移動
        Qc = vector_mean (Q)
        Q_ = Q - Qc

        # 共分散行列を特異値分解して回転行列を計算
        M = np.zeros((3, 3))
        for p, q in zip(P_, Q_):
            M += np.outer (q, p)
        U, _, Vt = np.linalg.svd (M, full_matrices=True)
        S = np.array ([[1, 0, 0], [0, 1, 0], [0, 0, np.linalg.det(U @ Vt)]],
                      dtype=np.float64)
        R = U @ S @ Vt
        r = R.reshape (9)

        # 収束判定
        t = calc_t (t_param, r)
        J_ = calc_J (P, V, R, t)
        if np.abs (J_ - J) < eps:
            break
        count += 1
        J = J_
        if count > 1000:
            count = 0
            eps *= 10.0

    return J, R, t
