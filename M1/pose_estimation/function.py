#-----------------------------------------------
# PnPLで使用する関数
#-----------------------------------------------

import numpy as np
import numpy.linalg as LA
import random

I = np.eye(3,dtype=np.float64)

#==========================================================
# ランダムな点群を生成する関数
# generate_points(num, x_range, y_range, z_range, seed)
# num: 生成する3次元点の数
# x_range,y_range,z_range: 点群の生成範囲
# seed: 乱数のシード値
# 
# return: num個の点の座標
#==========================================================
def generate_points(num, x_range, y_range, z_range):
    
    x = np.random.uniform(x_range[0], x_range[1], num)
    y = np.random.uniform(y_range[0], y_range[1], num)
    z = np.random.uniform(z_range[0], z_range[1], num)

    # 3次元点をまとめる
    p = np.column_stack((x, y, z))
    p = p.reshape((num, 3, 1))

    return p

#====================================================
# 半径rの円周上の点の座標(x-y平面上の点)を取得
# calc_circle_coordinates(deg, r)
# deg: deg刻みの点の座標を計算
# r: 円の半径
#
# return: 円周上の点
# 
# (1,0,0)の点は必ず含まれる
# deg=45の場合8点
#====================================================
def calc_circle_coordinates(deg, r):
    circle_coordinates = []
    c_num = int(360/deg)
    for i in range(c_num):
        # ラジアンに変換
        theta_deg = deg * i
        theta_rad = np.deg2rad(theta_deg)

        # x, y 平面上の座標を計算
        x = r * np.cos(theta_rad)
        y = r * np.sin(theta_rad)
        coordinate = np.array([[x],[y],[0]],dtype=np.float64)
        circle_coordinates.append(coordinate)

    return circle_coordinates

#===========================================
# カメラ位置から点の方向を向く回転行列を計算
# calc_R_to_point(coordinate, c)
# coordinate: カメラが向く3次元点の座標
# c: 世界座標系でのカメラの位置
#
# return: 回転行列
#===========================================
def calc_R_to_point(coordinate, c):
    Zc = (coordinate-c)/np.linalg.norm(coordinate-c, ord=2)
    P = calc_K_alpha(Zc)
    yc = np.array([[0],[-3.0],[3.0]],dtype=np.float64)
    Yc = P.dot(yc-c)
    Yc = Yc/np.linalg.norm(Yc, ord=2)

    Zc = Zc.T[0]
    Yc = Yc.T[0]
    Xc = np.cross(Yc,Zc)
    R = np.array([[Xc[0],Yc[0],Zc[0]],
                    [Xc[1],Yc[1],Zc[1]],
                    [Xc[2],Yc[2],Zc[2]]],
                    dtype=np.float64)

    return R.T

#================================================================
# 世界座標系の３次元点 p をカメラ座標系に変換する関数
# calc_cp(ps, R, t)
# ps: 世界座標系の3次元点の座標
# R: カメラの回転行列
# t: カメラの並進ベクトル
#
# return: p_alphaのカメラ座標系での座標
#
# ここで扱う3次元点は推定に使用しない点も含まれる
#================================================================
def calc_cp(ps, R, t):
    cp = np.zeros((ps.shape[0],3,1),dtype=np.float64)
    for i,ps in enumerate(ps):
        cp[i] = R.dot(ps) + t
    return cp

#==============================================================
# カメラ座標系の３次元点 cp を画像面に投影する関数
# calc_v_alpha(cp, K)
# cp: カメラ座標系の3次元点
# K: カメラの内部パラメータ
# 
# return: cpが画像面に投影された座標
#==============================================================
def calc_v_alpha(cp, K):
    v_alpha = np.zeros((cp.shape[0],3,1),dtype=np.float64)
    for i,p in enumerate(cp):
        cp[i] = cp[i]/p[2]
        v_alpha[i] = K.dot(cp[i])
    return v_alpha

#==============================================================
# 画像面に投影される点を選択する関数
# calc_points_in_range(height, width, ps, v_alpha)
# height, width: 画像面のサイズ
# ps: generate_pointsで生成した3次元点群
# v_alpha: calc_v_alphaで計算した画像面上の点
#
#return: 画像面に投影される点（推定に使用する点）
#==============================================================
def calc_points_in_range(height, width, ps, v_alpha):
    filtered_p_list = []
    filtered_v_list = []
    center_h = height // 2
    center_w = width // 2
    
    for p, v in zip(ps, v_alpha):
        x = int(v[0][0]) + center_w
        y = int(v[1][0]) + center_h
        if 0 <= x <= width and 0 <= y <= height:
            filtered_p_list.append(p)
            filtered_v_list.append(v)
    
    # リストをnumpy配列に変換して返す
    filtered_p = np.array(filtered_p_list)
    filtered_v = np.array(filtered_v_list)

    return filtered_p.reshape(-1, 3, 1), filtered_v.reshape(-1, 3, 1)


#==============================================================
# 画像上の直線のパラメータn_alphaを求める関数
# calc_n_alpha(v_alpha)
# v_alpha: 推定に使用する画像面上の点の座標
# (推定には2点をつないだ線分を使用するため, 線特徴数5の場合10点)
# 
# return: 画像上の直線のパラメータ
#==============================================================
def calc_n_alpha(v_alpha):
    n_alpha = np.zeros((round(v_alpha.shape[0]/2),3),dtype=np.float64)
    j = 0
    v_alpha = np.reshape(v_alpha,[v_alpha.shape[0],3])
    for i in range(0,v_alpha.shape[0],2):
        n_alpha[j] = np.cross(v_alpha[i],v_alpha[i+1])
        n_alpha[j] = n_alpha[j]/np.linalg.norm(n_alpha[j], ord=2)
        j += 1
    n_alpha = np.reshape(n_alpha,[round(v_alpha.shape[0]/2),3,1])
    return n_alpha

#==================================================================
# 世界座標系で表現された直線 L_alpha の方向ベクトルを求める関数
# calc_d_alpha(p_alpha)
# p_alpha: 推定に使用する世界座標系の3次元点の座標
# (推定には2点をつないだ線分を使用するため, 線特徴数5の場合10点)
# 
# return: 3次元直線のベクトル
#==================================================================
def calc_d_alpha(p_alpha):
    d_alpha = np.zeros((round(p_alpha.shape[0]/2),3,1),dtype=np.float64)
    j = 0
    for i in range(0,p_alpha.shape[0],2):
        d_alpha[j] = p_alpha[i+1] - p_alpha[i]
        d_alpha[j] = d_alpha[j]/np.linalg.norm(d_alpha[j], ord=2)
        j += 1
    return d_alpha

#=================================================================
# 射影行列 V_alpha を計算する関数(仮想カメラを考慮)
# calc_V_alpha_kappa(v_alpha, Rk0)
# v_alpha: 推定に使用する画像面上の点の座標
# Rkappa0: 仮想カメラから基準カメラへの回転行列
#
# return: 射影行列
#=================================================================
def calc_V_alpha_kappa(v_alpha, Rkappa0):
    v_alpha_kappa = np.dot(Rkappa0, v_alpha)
    V_alpha = np.dot(v_alpha_kappa,v_alpha_kappa.T)/np.dot(v_alpha_kappa.T,v_alpha_kappa)
    return V_alpha

#=================================================================
# 射影行列 K_alpha を計算する関数(仮想カメラを考慮)
# calc_K_alpha_kappa(n_alpha, Rk0)
# n_alpha: 推定に使用する画像上の直線のパラメータ
# Rkappa0: 仮想カメラから基準カメラへの回転行列
# 
# return: 射影行列
#=================================================================
def calc_K_alpha_kappa(n_alpha, Rkappa0):
    I = np.eye(3,dtype=np.float64)
    n_alpha_kappa = np.dot(Rkappa0, n_alpha)
    K_alpha = I - np.dot(n_alpha_kappa,n_alpha_kappa.T)/np.dot(n_alpha_kappa.T,n_alpha_kappa)
    return K_alpha

#=================================================
# 射影行列 K_alpha を計算する関数()
# calc_K_alpha(n_alpha)
# n_alpha: 推定に使用する画像上の直線のパラメータ
#
# return: 射影行列
#=================================================
def calc_K_alpha(n_alpha):
    I = np.eye(3,dtype=np.float64)
    K_alpha = I - np.dot(n_alpha,n_alpha.T)
    return K_alpha

#============================================
# 重心を計算する関数
# calc_mean(points)
# points: 重心を計算する対象の3次元点群
#
# mean: 重心の3次元座標
#============================================
def calc_mean(points):
    sum = np.zeros((3,1),dtype=np.float64)
    for point in points:
        sum = sum + point
    mean = sum / points.shape[0]
    return mean

#============================================================================
# p_alpha, r_alphaを重心を原点とする座標系で表現したデータに変換する(式(30),(31))
# calc_pr_prime(p_alpha, r_alpha)
# p_alpha, r_alpha: 変換前の3次元点の座標
#
# return: 変換後の3次元点の座標
#============================================================================
def calc_pr_prime(p_alpha, r_alpha):
    p_prime = np.zeros((p_alpha.shape[0],3,1),dtype=np.float64)
    r_prime = np.zeros((r_alpha.shape[0],3,1),dtype=np.float64)

    p_mean = calc_mean(p_alpha)
    r_mean = calc_mean(r_alpha)
    for i,p in enumerate(p_alpha):
        p_prime[i] = p - p_mean
    for i,r in enumerate(r_alpha):
        r_prime[i] = r - r_mean
    return p_prime, r_prime

#======================================================================================
# q_alpha, s_alphaを重心を原点とする座標系で表現したデータに変換する(式(33)~(35))
# calc_qs_prime(p_alpha, r_alpha, V_alpha, K_alpha, R_k, t)
# p_alpha: 点特徴として使用する3次元点
# r_alpha: 線特徴として使用する3次元直線上の点
# V_alpha, K_alpha: 射影行列
# R_k: 特異値分解で計算される回転行列(式(38))
# t: R_kを用いて計算される並進ベクトル(式(32))
#
# return: 変換後の3次元点の座標
#======================================================================================
def calc_qs_prime(p_alpha, r_alpha, V_alpha, K_alpha, R_k, t):
    q_alpha = np.zeros((p_alpha.shape[0],3,1),dtype=np.float64)
    q_prime = np.zeros((p_alpha.shape[0],3,1),dtype=np.float64)
    s_alpha = np.zeros((r_alpha.shape[0],3,1),dtype=np.float64)
    s_prime = np.zeros((r_alpha.shape[0],3,1),dtype=np.float64)

    for i,(V,p) in enumerate(zip(V_alpha,p_alpha)):
        q_alpha[i] = calc_q_alpha(V, R_k, p, t)

    for i,(K,r) in enumerate(zip(K_alpha, r_alpha)):
        s_alpha[i] = calc_s_alpha(K, R_k, r, t)

    q_mean = calc_mean(q_alpha)
    s_mean = calc_mean(s_alpha)
    for i,q in enumerate(q_alpha):
        q_prime[i] = q - q_mean
    for i,s in enumerate(s_alpha):
        s_prime[i] = s - s_mean
    return q_prime, s_prime

#=============================================================
# calc_qs_primeで使用する q_alpha を計算する関数(式(33))
# calc_q_alpha(V_alpha, R_k, p_alpha, t)
# V_alpha: 射影行列
# R_k: 特異値分解で計算される回転行列(式(38))
# p_alpha: 点特徴として使用する3次元点
# t: R_kを用いて計算される並進ベクトル(式(32))
#
# return: p_alpha
#=============================================================
def calc_q_alpha(V_alpha, R_k, p_alpha, t):
    q_alpha = np.dot(V_alpha , (np.dot(R_k, p_alpha) + t))
    return q_alpha

#=============================================================
# calc_qs_primeで使用する s_alpha を計算する関数(式(35))
# calc_s_alpha(K_alpha, R_k, p_alpha, t)
# K_alpha: 射影行列
# R_k: 特異値分解で計算される回転行列(式(38))
# r_alpha: 点特徴として使用する3次元点
# t: R_kを用いて計算される並進ベクトル(式(32))
#
# return: s_alpha
#=============================================================
def calc_s_alpha(K_alpha, R_k, r_alpha, t):
    s_alpha = np.dot(K_alpha , (np.dot(R_k, r_alpha) + t))
    return s_alpha

#=============================================================
# 回転行列R_kを用いて並進ベクトルを計算する関数(式(32))
# calc_tPL(cam_list, R)
# cam_list: GenerateDataの入っているリスト
# R: 特異値分解で計算される回転行列R_k
#
# return: 並進ベクトル
#=============================================================
def calc_tPL(cam_list, R):
    I_V = np.zeros((3,3),dtype=np.float64)
    I_K = np.zeros((3,3),dtype=np.float64)
    V_I_R_p = np.zeros((3,1),dtype=np.float64)
    K_I_R_r = np.zeros((3,1),dtype=np.float64)
    for vcam in cam_list:
        for V in vcam.V_alpha:
            I_V = I_V + (I - V)
        for K in vcam.K_alpha:
            I_K = I_K + (I - K)
        for (V,p) in zip(vcam.V_alpha, vcam.p_alpha):
            V_I_R_p = V_I_R_p + np.dot(np.dot((V - I),R),p)
        for (K,r) in zip(vcam.K_alpha, vcam.r_alpha):
            K_I_R_r = K_I_R_r + np.dot(np.dot((K - I),R),r)

    t = np.dot(np.linalg.inv((I_V + I_K)) , (V_I_R_p + K_I_R_r))
    return t

#=============================================================
# 共分散行列を計算する関数(式(37))
# calc_MPL(R, cam_list)
# R: 特異値分解で計算される回転行列R_k
# cam_list: GenerateDataの入っているリスト
#
# return: 共分散行列
#=============================================================
def calc_MPL(R, cam_list):
    q_R_p = np.zeros((3,3),dtype=np.float64)
    s_R_r = np.zeros((3,3),dtype=np.float64)
    #K_R_d = np.zeros((3,3),dtype=np.float64)
    for cam in cam_list:
        for (q,p) in zip(cam.q_prime,cam.p_prime):
            q_R_p = q_R_p + np.dot(q,p.T)
        for (s,r) in zip(cam.s_prime,cam.r_prime):
            s_R_r = s_R_r + np.dot(s,r.T)
        # for (K,d) in zip(cam.K_alpha,cam.d_alpha):
        #     K_R_d = K_R_d + np.dot((np.dot(np.dot(K,R),d)),d.T)
    M = q_R_p + s_R_r# + K_R_d
    return M

#==============================================================
# 行列 M を特異値分解して、回転行列 R を計算する関数(式(38))
# calc_R(M)
# M: calc_MPLで計算した共分散行列
#
# return: 回転行列
#==============================================================
def calc_R(M):
    U, S, Vt = LA.svd(M, full_matrices = True)
    R = U.dot(np.diag([1,1,np.linalg.det(U.dot(Vt))]).dot(Vt))
    return R

#==============================================================
# 目的関数を計算する関数
# calc_objfunc(cam_list, R, t)
# cam_list: カメラデータリスト
# R: 回転行列
# t: 並進ベクトル
#
# return: 目的関数
#==============================================================
def calc_objfunc(cam_list, R, t):
    I_V_R_p_t = 0
    I_K_R_r_t = 0
    #I_K_R_d = 0
    for vcam in cam_list:
        for (V,p) in zip(vcam.V_alpha, vcam.p_alpha):
            I_V_R_p_t = I_V_R_p_t + np.linalg.norm(np.dot((I - V), (np.dot(R, p) + t)))**2
        for (K,r) in zip(vcam.K_alpha, vcam.r_alpha):
            I_K_R_r_t = I_K_R_r_t + np.linalg.norm(np.dot((I - K), (np.dot(R, r) + t)))**2
        # for (K,d) in zip(vcam.K_alpha, vcam.d_alpha):
        #     I_K_R_d = I_K_R_d + np.linalg.norm(np.dot((I - K), np.dot(R, d)))**2 

    E = I_V_R_p_t + I_K_R_r_t# + I_K_R_d
    return E

#==============================================================
# 答えが収束するまで計算する関数
# PnPL(thd, cam_list, R_0)
# thd: 収束判定の閾値
# cam_list: GenerateDataの入っているリスト
# R_0: 回転行列の初期値
#
# return: 回転行列, 並進ベクトル, 最終的な目的関数の値, 反復回数, フラグ
#==============================================================
def PnPL(thd, cam_list, R_0):
    flag = 0

    # k = 0 と初期化する
    k = 0

    R_k = R_0

    while True:
        # 回転行列 R_0 を用いて並進ベクトル t を式(32)より計算する
        t = calc_tPL(cam_list, R_k)

        for cam in cam_list:
            cam.q_prime, cam.s_prime = calc_qs_prime(cam.p_alpha, cam.r_alpha, cam.V_alpha, cam.K_alpha, R_k, t)

        # 式(37)を用いて共分散行列 M を計算する
        M = calc_MPL(R_k, cam_list)

        # 行列 M を特異値分解して、回転行列 R を計算する
        R = calc_R(M)

        # 値が収束していれば終了する
        if (np.linalg.norm(R - R_k, ord = 2)) <= thd:
            break
        else:
            R_k = R
        k = k + 1

        if k > 10000:
            flag = 1
            break
       
    # 回転行列 R を用いて並進ベクトル t を式(32)より計算する
    t = calc_tPL(cam_list, R)

    E = calc_objfunc(cam_list, R, t)
    
    return R,t,E,k,flag

#========================================================
# 誤差を加えたデータを再計算する関数
# recalc_data(cam, xp_errors, yp_errors, xl_errors, yl_errors)
# cam: データを再計算するカメラオブジェクト
# xp_errors, yp_errors: 点特徴用の点に付加する誤差
# xl_errors, yl_errors: 線特徴用の点に付加する誤差
#
#========================================================
def recalc_data(cam, xp_errors, yp_errors, xl_errors, yl_errors, npoints, nlines):

    # 誤差を加えたデータ用の配列を用意
    cam.v_in_range_P_error = cam.v_in_range_P.copy()
    cam.v_in_range_L_error = cam.v_in_range_L.copy()

    # 画面に投影された点に誤差を付加
    for n in range(cam.v_in_range_P.shape[0]):
        cam.v_in_range_P_error[n, 0, 0] += xp_errors[n]
        cam.v_in_range_P_error[n, 1, 0] += yp_errors[n]
    for n in range(cam.v_in_range_L.shape[0]):
        cam.v_in_range_L_error[n, 0, 0] += xl_errors[n]
        cam.v_in_range_L_error[n, 1, 0] += yl_errors[n]

    # 点特徴と線特徴に使用する点を選択
    select_points_list = random.sample(range(cam.p_in_range_P.shape[0]), npoints)
    select_lines_list = random.sample(range(cam.p_in_range_L.shape[0]), nlines*2)

    cam.p_alpha = cam.p_in_range_P[select_points_list, :, :]
    cam.v_alpha = cam.v_in_range_P_error[select_points_list, :, :]
    cam.r_alpha = cam.p_in_range_L[select_lines_list[0::2], :, :]
    cam.d_alpha = calc_d_alpha(cam.p_in_range_L[select_lines_list, :, :])
    cam.n_alpha = calc_n_alpha(cam.v_in_range_L_error[select_lines_list, :, :])

    # 射影行列を計算
    cam.V_alpha = np.zeros((cam.v_alpha.shape[0],3,3),dtype=np.float64)
    cam.K_alpha = np.zeros((cam.n_alpha.shape[0],3,3),dtype=np.float64)
    for j, v in enumerate(cam.v_alpha):
        cam.V_alpha[j] = calc_V_alpha_kappa(v, cam.R_k0)
    for j, n in enumerate(cam.n_alpha):
        cam.K_alpha[j] = calc_K_alpha_kappa(n, cam.R_k0)

    cam.p_prime, cam.r_prime = calc_pr_prime(cam.p_alpha, cam.r_alpha)

    return select_points_list, select_lines_list