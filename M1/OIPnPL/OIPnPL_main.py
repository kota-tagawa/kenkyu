import numpy as np
import argparse
import OIPnPL

#
# 入力データ（点と射影行列のペアデータ）を読み込む関数
#
# @input  filename   : ファイル名
# @input  _delimiter : 区切り文字 
# @output P          : 3次元点データのリスト
# @output V          : 射影行列のリスト
#
def read_input_data (filename, _delimiter):
    data = np.genfromtxt (filename, delimiter=_delimiter, dtype=np.float64, encoding=None)
    ndata, _ = data.shape
    P = []
    V = []
    for n in range(0, ndata, 4):
        P.append (data[n])
        V.append (data[n+1:n+4:, :])  
    
    return P, V

#
# 回転行列の初期値を設定する関数
#
# @output Rs : 回転行列のリスト
#
def setup_init_Rs():
    Rs = []
    Rs.append(np.array ([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64))
    Rs.append(np.array ([[0, 1, 0], [0, 0, -1], [-1, 0, 0]], dtype=np.float64))
    Rs.append(np.array ([[-1, 0, 0], [0, 0, -1], [0, -1, 0]], dtype=np.float64))
    Rs.append(np.array ([[0, -1, 0], [0, 0, -1], [1, 0, 0]], dtype=np.float64))
    Rs.append(np.array ([[1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64))
    Rs.append(np.array ([[0, 1, 0], [-1, 0, 0], [0, 0, -1]], dtype=np.float64))
    Rs.append(np.array ([[-1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64))
    Rs.append(np.array ([[0, -1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float64))
    Rs.append(np.array ([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64))
    Rs.append(np.array ([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64))
    Rs.append(np.array ([[-1, 0, 0], [0, -1, 0], [0, 0, 1]], dtype=np.float64))
    Rs.append(np.array ([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64))
    return Rs

#
# 推定結果を出力する関数
#
def output_result (R, t):
    print(f'{R[0][0]:.16e}', f'{R[0][1]:.16e}', f'{R[0][2]:.16e}')
    print(f'{R[1][0]:.16e}', f'{R[1][1]:.16e}', f'{R[1][2]:.16e}')
    print(f'{R[2][0]:.16e}', f'{R[2][1]:.16e}', f'{R[2][2]:.16e}')
    print(f'{t[0]:.16e}', f'{t[1]:.16e}', f'{t[2]:.16e}')

#
# メイン関数
#
def main():
    parser = argparse.ArgumentParser (formatter_class=argparse.ArgumentDefaultsHelpFormatter, 
                                      description='Compute camera param from point and line pairs')
    parser.add_argument('-i', required=True, 
                        help='Input data')
    parser.add_argument('--eps', required=False, 
                        help='convergence parameter. Default = 1.0e-8',
                        type=float,
                        default=1.0e-8)
    
    # 引数解析
    args = parser.parse_args()

    # 入力データの読み込み
    P, V = read_input_data (args.i, ' ')

    # 初期値の設定
    Rs = setup_init_Rs ()
    
    # カメラパラメータの推定
    J_best = 1.0e+10
    R_best = None
    t_best = None
    for R in Rs:
        J, R_est, t_est = OIPnPL.OIPnPL (P, V, R, args.eps)
        if J < J_best:
            J_best = J
            R_best = R_est
            t_best = t_est

    # 結果の出力
    output_result (R_best, t_best)

    
if __name__ == '__main__':
    main()
