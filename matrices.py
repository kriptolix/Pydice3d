import numpy as np
import math
import pybullet as pb

def mat4_persp(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0,0] = f / aspect
    m[1,1] = f
    m[2,2] = (far + near) / (near - far)
    m[2,3] = (2 * far * near) / (near - far)
    m[3,2] = -1.0
    return m


def mat4_lookat(eye, center, up):
    f = np.array(center) - np.array(eye)
    f /= np.linalg.norm(f)
    r = np.cross(f, np.array(up)); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    m = np.eye(4, dtype=np.float32)
    m[0,:3] = r;  m[0,3] = -np.dot(r, eye)
    m[1,:3] = u;  m[1,3] = -np.dot(u, eye)
    m[2,:3] = -f; m[2,3] =  np.dot(f, eye)
    return m


def mat4_from_bullet(pos, orn):
    """Converte posição+quaternion do Bullet para matriz 4x4 column-major→row-major."""
    # Bullet retorna quaternion (x,y,z,w)
    rm = pb.getMatrixFromQuaternion(orn)  # lista de 9 floats, row-major
    m = np.eye(4, dtype=np.float32)
    m[0,0]=rm[0]; m[0,1]=rm[1]; m[0,2]=rm[2]; m[0,3]=pos[0]
    m[1,0]=rm[3]; m[1,1]=rm[4]; m[1,2]=rm[5]; m[1,3]=pos[1]
    m[2,0]=rm[6]; m[2,1]=rm[7]; m[2,2]=rm[8]; m[2,3]=pos[2]
    return m