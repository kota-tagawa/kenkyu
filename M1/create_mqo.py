# create_mqo.py
# editor : tagawa kota, sugano yasuyuki
# last edited : 2025/3/12
# Assign uv-coords and texture images to 3D model files

# 汎用ライブラリ(以下に追加)
import numpy as np

class CreateMQO:
    
    #================================================
    # CreateMQOクラスのコンストラクタ
    # __init__(input_path, output_path, FNUM)
    #
    # 引数
    #  - input_path : 3次元モデルの読み込み先
    #  - output_path : 3次元モデルの書き込み先
    #  - FNUM : モデルに割り当てるテクスチャの数
    #================================================    
    
    def __init__(self, input_path, output_path, FNUM):
        self.file_path_r = input_path                   # 読み込み先
        self.file_path_w = output_path                  # 書き込み先
        self.FNUM = FNUM                                # 面の数
        self.dist_list = np.full(FNUM, 5)               # 距離情報
        self.area_list = np.full(FNUM, 0)               # 面積情報
        self.brightness_list = np.zeros((FNUM, 3))      # 明るさ情報
        self.lines = self.read_mqo()                      

    # MQOファイルの読み込み
    def read_mqo(self):
        with open(self.file_path_r, 'r', encoding='utf-8') as file:
            return file.readlines()

    # MQOファイルの書き込み
    def write_mqo(self):
        with open(self.file_path_w, 'w', encoding='utf-8') as file:
            file.writelines(self.lines)


    #================================================
    # material(テクスチャ情報)とface(uv座標)のセクションを解析
    # parse_material_and_face_sections(self)
    #
    # 返り値
    # - material_section : テクスチャ情報のセクション
    # - face_section : uv情報のセクション
    # 
    # 概要
    # - MQOファイル全体から、テクスチャ情報の部分とuv情報の部分を抜き出す
    #================================================
    
    def parse_material_and_face_sections(self):
        material_section, face_section = [], []
        inside_material, inside_face = False, False

        for line in self.lines:
            if "cam0" in line:  # カメラオブジェクト内のmaterial、faceは無視
                break
            if "}" in line:     # section終了
                inside_material = inside_face = False

            if inside_material:
                material_section.append(line)
            if inside_face:
                face_section.append(line)

            if "Material " in line:     # Material section開始
                inside_material = True
            elif "\tface " in line:     # face section開始
                inside_face = True
                inside_material = False

        return material_section, face_section

    #================================================
    # material(テクスチャ情報)のセクションを編集
    # edit_material_section(self, material_section, texture, index):
    #
    # 引数
    # - material_section : テクスチャ情報のセクションのリスト
    # - texture : 割り当てるテクスチャのパス
    # - index : テクスチャのインデックス
    #
    # 返り値
    # - edited_material : 編集済みのテクスチャ情報
    # 
    # 概要
    # - テクスチャ情報の行を参照して、indexと一致したら新たなテクスチャのパスを割り当てる
    #================================================
    def edit_material_section(self, material_section, texture, index):
        edited_material = []
    
        for i, lines in enumerate(material_section):
            if i == index:
                line = lines.split(' ')
                line[11] = f'tex("{texture}")\n'
                edited_material.append(" ".join(line))
            else:
                edited_material.append(lines)
            
        return edited_material

    #================================================
    # face(uv情報)のセクションを編集（3v、4v）
    # edit_face_section_3v(self, face_section, uv, index):
    #
    # 引数
    # - face_section : uv情報のセクションのリスト
    # - uv : 割り当てるuv座標
    # - index : uv座標のインデックス
    #
    # 返り値
    # - edited_face : 編集済みのuv情報
    # 
    # 概要
    # - uv情報の行を参照して、indexと一致したら新たなuv座標を割り当てる
    #================================================
    def edit_face_section_3v(self, face_section, uv, index):
        edited_face = []

        for i, lines in enumerate(face_section):
            if i == index:
                uv = uv.tolist()
                line = lines.split(' ')
                if line[0] == "\t\t3":
                    line[5] = f'UV({str(uv[0][0])}'
                    line[6] = str(uv[0][1])
                    line[7] = str(uv[1][0])
                    line[8] = str(uv[1][1])
                    line[9] = str(uv[2][0])
                    line[10] = f'{str(uv[2][1])})\n'
                edited_face.append(" ".join(line))
            else:    
                edited_face.append(lines)

        return edited_face
    
    def edit_face_section_4v(self, face_section, uv, index):
        edited_face = []

        for i, lines in enumerate(face_section):
            if i == index:
                uv = uv.tolist()
                line = lines.split(' ')
                if line[0] == "\t\t4":
                    line[6] = f'UV({str(uv[0][0])}'
                    line[7] = str(uv[0][1])
                    line[8] = str(uv[1][0])
                    line[9] = str(uv[1][1])
                    line[10] = str(uv[2][0])
                    line[11] = str(uv[2][1])
                    line[12] = str(uv[3][0])
                    line[13] = f'{str(uv[3][1])})\n'
                edited_face.append(" ".join(line))
            else:    
                edited_face.append(lines)

        return edited_face
    

    #================================================
    # MQOファイルを更新
    # update_mqo(self, uv, texture, index):
    #
    # 引数
    # - texture : 割り当てるテクスチャのパス
    # - uv : 割り当てるuv座標
    # - index : 割り当てるインデックス
    # 
    # 概要
    # - MQOファイルを解析して、テクスチャ情報とuv情報のセクションを読み込む
    # - それぞれのセクションを編集
    # - 該当する部分に挿入して、MQOファイルの内容を更新する
    #================================================
    def update_mqo(self, uv, texture, index):
        # material(テクスチャ情報)とface(uv座標)のセクションを解析
        material, face = self.parse_material_and_face_sections()
        # Materialの個数(カメラ用テクスチャ+1)
        material_num = self.FNUM + 1
        # テクスチャ情報を編集
        if material and texture is not None:
            edited_material = self.edit_material_section(material, texture, index)
            self.lines = [line for line in self.lines if line not in material]
            insertion_point = self.lines.index('Material %s {\n' % str(material_num))
            self.lines[insertion_point+1:insertion_point+1] = edited_material

        # uv情報を編集
        if face and uv is not None:
            if len(uv) == 3:
                edited_face = self.edit_face_section_3v(face, uv, index)
            else:
                edited_face = self.edit_face_section_4v(face, uv, index)    
            self.lines = [line for line in self.lines if line not in face]
            insertion_point = self.lines.index('\tface %s {\n' % str(self.FNUM))
            self.lines[insertion_point+1:insertion_point+1] = edited_face
        
            
    #================================================
    # カメラ情報を書き込み
    # write_campos(self, R, t, cam_num):
    #
    # 引数
    # - R : カメラの回転行列
    # - t : カメラの並進ベクトル
    # - cam_num : 全方位カメラ番号
    # 
    # 概要
    # - 全方位カメラの位置姿勢を求めて、MQOファイルに全方位カメラを書き込む
    #================================================
    def write_campos(self, R, t, cam_num):
        # 三角錐を作成
        V0 = np.array([0, 0, 0])
        # 底面の3頂点 (V1, V2, V3)
        V1 = np.array([-0.1, 0.2, 0.1])
        V2 = np.array([0.1, 0.2, 0.1])
        V3 = np.array([0.1, 0.2, -0.1])
        V4 = np.array([-0.1, 0.2, -0.1])
        vertices = np.array([V0, V1, V2, V3, V4])
        # 平行移動
        C = -np.dot(R.T, t.reshape(-1))
        vertices = vertices + C
        # テクスチャ生成
        texture_num = self.FNUM + 1
        # カメラオブジェクトを追加
        cam_object = []
        cam_object.append('Object "cam%s" {\n' % cam_num)
        cam_object.append('\tdepth 0\n')
        cam_object.append('\tfolding 0\n')
        cam_object.append('\tscale 1 1 1\n')
        cam_object.append('\trotation 0 0 0\n')
        cam_object.append('\tvisible 15\n')
        cam_object.append('\tlocking 0\n')
        cam_object.append('\tshading 1\n')
        cam_object.append('\tfacet 59.5\n')
        cam_object.append('\tnormal_weight 1\n')
        cam_object.append('\tcolor 1 0 0\n')
        cam_object.append('\tcolor_type 0\n')
        cam_object.append('\tvertex 5 {\n')
        cam_object.append('\t\t%s %s %s\n' %(vertices[0][0], vertices[0][1], vertices[0][2]))
        cam_object.append('\t\t%s %s %s\n' %(vertices[1][0], vertices[1][1], vertices[1][2]))
        cam_object.append('\t\t%s %s %s\n' %(vertices[2][0], vertices[2][1], vertices[2][2]))
        cam_object.append('\t\t%s %s %s\n' %(vertices[3][0], vertices[3][1], vertices[3][2]))
        cam_object.append('\t\t%s %s %s\n' %(vertices[4][0], vertices[4][1], vertices[4][2]))
        cam_object.append('\t}\n')
        cam_object.append('\tface 6 {\n')
        cam_object.append('\t\t3 V(0 1 2) M(%s) UV(0 0 1 0 1 1)\n' % texture_num)
        cam_object.append('\t\t3 V(0 2 3) M(%s) UV(0 0 1 0 1 1)\n' % texture_num)
        cam_object.append('\t\t3 V(0 3 4) M(%s) UV(0 0 1 0 1 1)\n' % texture_num)
        cam_object.append('\t\t3 V(0 4 1) M(%s) UV(0 0 1 0 1 1)\n' % texture_num)
        cam_object.append('\t\t3 V(1 2 4) M(%s) UV(0 0 1 0 1 1)\n' % texture_num)
        cam_object.append('\t\t3 V(2 3 4) M(%s) UV(0 0 1 0 1 1)\n' % texture_num)
        cam_object.append('\t}\n')
        cam_object.append('}\n')
        # # cam_objectを文字列に変換
        cam_object_str = ''.join(cam_object)
        # "Eof" 行の前に新しいオブジェクトを挿入
        for i, line in enumerate(self.lines):
            if line.strip() == 'Eof':
                insert_index = i  # Eofの位置を取得
                self.lines[insert_index:insert_index] = cam_object_str  # その位置に行として挿入
                break
