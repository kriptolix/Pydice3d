import numpy as np
import pybullet as pb

def get_top_face(body_id, face_normals, client):
    pos, orn = pb.getBasePositionAndOrientation(body_id, physicsClientId=client)
    rot_matrix = np.array(pb.getMatrixFromQuaternion(orn)).reshape(3, 3)

    world_up = np.array([0, 1, 0])

    best_face = None
    best_dot = -1

    for face, normal in face_normals.items():
        local_n = np.array(normal)
        world_n = rot_matrix @ local_n

        dot = np.dot(world_n, world_up)

        if dot > best_dot:
            best_dot = dot
            best_face = face

    return best_face


## Quando o dado parar, você faz:
for body in self.physics.dice_ids:
    face = get_top_face(body, FACE_NORMALS_D6, self.physics.client)
    print("Resultado:", face)

def load_obj(path):
    vertices = []
    normals = []
    faces = []

    with open(path, "r") as f:
        for line in f:
            if line.startswith("v "):
                vertices.append(list(map(float, line.split()[1:])))
            elif line.startswith("vn "):
                normals.append(list(map(float, line.split()[1:])))
            elif line.startswith("f "):
                faces.append(line.split()[1:])

    return vertices, normals, faces