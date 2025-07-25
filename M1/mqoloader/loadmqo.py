from mqoloader.mesh3d import Mesh3D
from mqoloader.material import Material
from mqoloader.vector3d import Vector3D
from mqoloader.model3d import Model3D
from mqoloader.face3d import Face3D
from OpenGL.GL import *
from OpenGL.GLU import *
# OpenGL初期化（glfw）
import glfw
import os

class LoadMQO(Model3D):

    def __init__(self,filename, scale, normal_flag):
        # OpenGL初期化(Windowsの場合必要)
        self.create_OpneGLcontext()

        Model3D.__init__(self)

        self.scalex = scale
        self.scaley = scale
        self.scalez = scale
        
        ### 追加
        self.sizex = 1.0
        self.sizey = 1.0
        self.sizez = 1.0
        self.shiftx = 0.0
        self.shifty = 0.0
        self.shiftz = 0.0
        
        # パスの解析
        parts = filename.split(os.sep)
        if len(parts) == 1:
            self.path = "."
        else:
            self.path = os.sep.join(parts[:-1])

        with open(filename) as file:
            lines = file.read()

        lines = lines.split('\n')
        for i in range(len(lines)):
            line = lines[i].split(' ')
            if line[0] == "Material":
                self.add_material(lines,i)
            elif line[0] == "Object":
                if line[1].startswith('"cam'):
                    print("カメラオブジェクトは読み込みません。")
                    break
                self.meshes.append(Mesh3D(line[1].replace('"','')))
            elif line[0] == "\tfacet":
                self.set_facet(lines,i,self.meshes[-1])
            elif line[0] == "\tvertex":
                self.add_vertex(lines,i,self.meshes[-1])
            elif line[0] == "\tface":
                self.add_face(lines,i,self.meshes[-1].faces)

        if normal_flag:
            self.calc_normals(flat=False)
    
    def create_OpneGLcontext(self):
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW3")
        # ウィンドウ生成
        self.window = glfw.create_window(800, 600, "OpenGL Window", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create a window")
        # OpenGL コンテキストを作成
        glfw.make_context_current(self.window)

    def set_facet(self,lines,index,mesh):
        line = lines[index].split(' ')
        facet = float(line[1])
        mesh.set_facet(facet)

    def add_material(self,lines,index):
        line = lines[index].split(' ')
        num = int(line[1])
        textureIDs = (GLuint * (num + 1)) ()
        glGenTextures(num, textureIDs)
        for i in range(num + 1):
            textureIDs[i] = i
            tex_count = 1
            
        for i in range(index+1,num+index+1):
            line = lines[i].split(' ')
            name = line[0].replace('\t','').replace('"','')
            index = 1
            if 'shader' in line[1]:
                index += 1
            if 'vcol' in line[2]:
                index += 1
            if 'dbls' in line[2]:
                index += 1
            if 'dbls' in line[3]:
                print('in')
                index += 1
                
            one = float(line[index].replace('col(',''))
            two = float(line[index+1])
            three = float(line[index+2])
            four = float(line[index+3].replace(')',''))
            dif = float(line[index+4].replace('dif(','').replace(')',''))
            amb = float(line[index+5].replace('amb(','').replace(')',''))
            emi = float(line[index+6].replace('emi(','').replace(')',''))
            spc = float(line[index+7].replace('spc(','').replace(')',''))
            power = float(line[index+8].replace('power(','').replace(')',''))
            tex = None

            if len(line) == index+10:
                tex = line[index+9].replace('tex("','').replace('")','')
                tex = os.path.join(self.path, tex)
                # テクスチャファイル存在確認
                if not os.path.exists(tex):
                    raise FileNotFoundError(f"TextureFile does not exist: {tex}")
                
            self.materials.append(Material(name,
                                           [one, two, three, four],
                                           dif, amb, emi, spc, power,
                                           textureIDs[tex_count],
                                           tex))
            tex_count += 1

    def add_vertex(self, lines, index, mesh):
        line = lines[index].split(' ')
        num = int(line[1])
        ### 追加
        x = []
        y = []
        z = []
        v = []
        for i in range(index+1, num+index+1):
            line = lines[i].split(' ')
            mesh.vertices.append(Vector3D(float(line[0]),
                                          float(line[1]),
                                          float(line[2])))
            ### 追加
            x.append(float(line[0]))
            y.append(float(line[1]))
            z.append(float(line[2]))
            v.append([float(line[0]),float(line[1]),float(line[2])])
        self.set_modelvertex(x,y,z,v)

    def add_face(self, lines, index, faces):
        line = lines[index].split(' ')
        num = int(line[1])
        ### 追加
        f = []
        uv = []
        for i in range(index+1,num+index+1):
            line = lines[i].split(' ')
            if line[0] == "\t\t3":
                one   = int(line[1].replace('V(',''))
                two   = int(line[2])
                three = int(line[3].replace(')',''))
                material = line[4].replace('M(','')
                material = int(material.replace(')',''))
                u1 = v1 = u2 = v2 = u3 = v3 = 0
                if len(line) >= 6:
                    u1 = float(line[5].replace('UV(',''))
                    v1 = float(line[6])
                    u2 = float(line[7])
                    v2 = float(line[8])
                    u3 = float(line[9])
                    v3 = float(line[10].replace(')',''))
                    faces.append(Face3D([three,two,one],material,[u3,v3,u2,v2,u1,v1]))
                f.append([one, two, three])
                uv.append([[u1, v1], [u2, v2], [u3, v3]])
            elif line[0] == "\t\t4":
                one   = int(line[1].replace('V(',''))
                two   = int(line[2])
                three = int(line[3])
                four  = int(line[4].replace(')',''))
                material = line[5].replace('M(','')
                material = int(material.replace(')',''))
                u1 = v1 = u2 = v2 = u3 = v3 = u4 = v4 = 0
                if len(line) >= 7:
                    u1 = float(line[6].replace('UV(',''))
                    v1 = float(line[7])
                    u2 = float(line[8])
                    v2 = float(line[9])
                    u3 = float(line[10])
                    v3 = float(line[11])
                    u4 = float(line[12])
                    v4 = float(line[13].replace(')',''))
                    faces.append(Face3D([three, two, one],material,[u3, v3, u2, v2, u1, v1]))
                    faces.append(Face3D([one, four, three],material,[u1, v1, u4, v4, u3, v3]))
                f.append([one, two, three, four])
                uv.append([[u1, v1], [u2, v2], [u3, v3], [u4, v4]])
            self.set_modelface(f)
            self.set_modeluv(uv)
    
    def set_modelvertex(self,x,y,z,v):
        self.x = x
        self.y = y
        self.z = z
        self.v = v
    
    def set_modelface(self,f):
        self.f = f

    def set_modeluv(self,uv):
        self.uv = uv
