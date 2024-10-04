import numpy as np

# # ASSUMPTION: MANIFOLD TRIANGLE MESHES

# # 1. position on input to position on unit sphere (Gauss map; using normals)
# # 2. position on unit sphere to analogy normal

# def nearest_normal(input_normals, analogy_normals):
#   # given a position on the unit sphere, what's the analogous normal?
#   nvertices, _ = input_normals.shape

#   for i in range(nvertices):
#     # map vertex in input to point on unit sphere (Gauss map)
#     position = input_normals[i] / np.linalg.norm(input_normals[i])

#     # map point on unit sphere to analogy normal


# def spherical_parametrization():
#   pass


# def calc_target_normal(input_normals, analogy_normals):
#   # input object normal -> unit sphere normal -> analogy object normal
#   nvertices, _ = input_normals
#   for

#   target_normal = None
#   min_distance = float("inf")
#   for analogy_normal in analogy_normals:
#     distance = np.abs(np.dot(normal, analogy_normal))
#     if distance < min_distance:
#       min_distance = distance
#       target_normal = analogy_normal

#   return target_normal


class Optimizer:
  def __init__(self, analogy_path, input_path, output_path):
    self._analogy_path = analogy_path
    self._input_path = input_path
    self._output_path = output_path

    self._analogy_verts, self._analogy_faces = self._load_obj(analogy_path)
    self._input_verts, self._input_faces = self._load_obj(input_path)
    self._output_verts, self._output_faces = np.copy(self._input_verts), np.copy(self._input_faces)

  def run(self, factor):
    input_nverts, _ = self._input_verts.shape

    # NOTE: face normals for analogy shape, vertex normals for input shape
    analogy_face_normals = self._calc_face_normals(self._analogy_verts, self._analogy_faces)
    input_vert_normals = self._calc_vert_normals(self._input_verts, self._input_faces)

    # is thee sphere right-handed????
    # edges are verticallllll

    # "target" vert normals
    output_vert_normals = self._calc_output_vert_normals(analogy_face_normals, input_vert_normals)

    input_vert_areas = self._calc_vert_areas(self._input_verts, self._input_faces)

    input_vert_neighborhoods, input_vert_neighborhood_weights = self._calc_vert_neighborhoods(
      self._input_verts, self._input_faces
    )

    Q = np.zeros((input_nverts, input_nverts))
    for k in range(input_nverts):
      Q += (
        input_vert_neighborhoods[k]
        @ np.diag(input_vert_neighborhood_weights[k])
        @ input_vert_neighborhoods[k].T
      )
    print("Cot Laplace")
    print(Q)  # how can this be negative? this is scaled by some factor?
    Q_inv = np.linalg.inv(Q)

    K = np.zeros((input_nverts * 3, input_nverts))
    for k in range(input_nverts):
      K[3 * k : 3 * (k + 1), :] = (
        self._input_verts.T
        @ input_vert_neighborhoods[k]
        @ np.diag(input_vert_neighborhood_weights[k])
        @ input_vert_neighborhoods[k].T
      )

    rotations = np.hstack([np.eye(3)] * input_nverts)

    for _ in range(100):
      # local-step
      for k in range(input_nverts):
        input_edges = np.hstack(
          [
            self._input_verts.T @ input_vert_neighborhoods[k],
            input_vert_normals[k, :].reshape((3, 1)),
          ]
        )
        output_edges = np.hstack(
          [
            self._output_verts.T @ input_vert_neighborhoods[k],
            output_vert_normals[k, :].reshape((3, 1)),
          ]
        )
        weights = np.diag(
          np.append(input_vert_neighborhood_weights[k], factor * input_vert_areas[k])
        )

        U, _, Vt = np.linalg.svd(input_edges @ weights @ output_edges.T)
        rotation = Vt.T @ U.T
        if np.linalg.det(rotation) < 0:
          Vt[0, :] *= -1
          rotation = Vt.T @ U.T
        rotations[:, 3 * k : 3 * (k + 1)] = rotation

      # global step
      self._output_verts = Q_inv @ K.T @ rotations.T

      energy = np.trace(self._output_verts.T @ Q @ self._output_verts) - 2 * np.trace(
        rotations @ K @ self._output_verts
      )  # why is this neg? -- maybe because there's no constant
      print(energy)

      # when factor = 0, vertices shouldn't move
      # use vn to fix order of verts?

    self._save_obj(self._output_path, self._output_verts, self._output_faces, output_vert_normals)

  def _calc_output_vert_normals(self, analogy_vert_normals, input_vert_normals):
    input_nverts, _ = input_vert_normals.shape

    output_vert_normals = np.zeros_like(input_vert_normals)

    for i in range(input_nverts):
      j = np.argmax(np.dot(analogy_vert_normals, input_vert_normals[i, :]))
      output_vert_normals[i, :] = analogy_vert_normals[j, :]

    return output_vert_normals

  def _load_obj(self, path):
    verts = []
    faces = []

    with open(path, "r") as file:
      for line in file:
        if line.startswith("v "):
          parts = line.split()
          x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
          vert = [x, y, z]
          verts.append(vert)

        elif line.startswith("f "):
          parts = line.split()
          i, j, k = (
            int(parts[1].split("/")[0]) - 1,
            int(parts[2].split("/")[0]) - 1,
            int(parts[3].split("/")[0]) - 1,
          )
          face = [i, j, k]
          faces.append(face)

    return np.array(verts), np.array(faces)

  # ACTUALLY QUESTION -- IS THERE ANYTHING THAT NEEDS THIS TO BE A WATERTIGHT MESH???

  def _save_obj(self, path, verts, faces, vert_normals=[]):
    lines = []

    for vert in verts:
      lines.append(f"v {vert[0]} {vert[1]} {vert[2]}")

    for vert_normal in vert_normals:
      lines.append(f"vn {vert_normal[0]} {vert_normal[1]} {vert_normal[2]}")

    for face in faces:
      lines.append(
        f"f {face[0] + 1}//{face[0] + 1} {face[1] + 1}//{face[1] + 1} {face[2] + 1}//{face[2] + 1}"
      )

    with open(path, "w") as file:
      file.write("\n".join(lines))

  # CORRECT DON'T TOUCH
  def _calc_face_areas(self, verts, faces):
    nfaces, _ = faces.shape

    face_areas = np.zeros((nfaces,))

    for f in range(nfaces):
      i, j, k = faces[f, :]
      vert_i, vert_j, vert_k = verts[i, :], verts[j, :], verts[k, :]

      edge_ij = vert_j - vert_i
      edge_jk = vert_k - vert_j

      face_area = np.linalg.norm(np.cross(edge_ij, edge_jk)) / 2
      face_areas[f] = face_area

    return face_areas

  # CORRECT DON'T TOUCH (but TODO: mixed Voronoi area)
  def _calc_vert_areas(self, verts, faces):
    nverts, _ = verts.shape
    nfaces, _ = faces.shape

    face_areas = self._calc_face_areas(verts, faces)

    vert_areas = np.zeros((nverts,))
    for k in range(nverts):
      vert_area = 0
      for f in range(nfaces):
        if k in faces[f, :]:
          vert_area += face_areas[f] / 3
      vert_areas[k] = vert_area
    # mixed Voronoi area: acute/right -> circumcenters; obtuse -> barycenter
    return vert_areas

  # CORRECT DON'T TOUCH
  def _calc_face_normals(self, verts, faces):
    nfaces, _ = faces.shape

    face_normals = np.zeros((nfaces, 3))

    for f in range(nfaces):
      i, j, k = faces[f, :]
      vert_i, vert_j, vert_k = verts[i, :], verts[j, :], verts[k, :]

      edge_ij = vert_j - vert_i
      edge_jk = vert_k - vert_j

      face_normal = np.cross(edge_ij, edge_jk)
      face_normal /= np.linalg.norm(face_normal)
      face_normals[f, :] = face_normal

    return face_normals

  # CORRECT DON'T TOUCH
  def _calc_vert_normals(self, verts, faces):
    nverts, _ = verts.shape
    nfaces, _ = faces.shape

    vert_normals = np.zeros((nverts, 3))
    face_normals = self._calc_face_normals(verts, faces)
    face_areas = self._calc_face_areas(verts, faces)

    for v in range(nverts):
      vert_normal = np.zeros((3,))

      for f in range(nfaces):
        if v in faces[f, :]:
          vert_normal += face_areas[f] * face_normals[f, :]

      vert_normal /= np.linalg.norm(vert_normal)
      vert_normals[v, :] = vert_normal

    return vert_normals

  # CORRECT DON'T TOUCH
  def _calc_vert_neighborhoods(self, verts, faces):
    nverts, _ = verts.shape

    vert_cots = self._calc_vert_cots(verts, faces)

    # 1-ring neighborhood
    vert_neighborhood = []
    vert_neighborhood_weights = []

    for v in range(nverts):
      edges = []
      weights = []
      for face in faces:
        if v in face:
          i, j, k = face

          edge_ij = np.zeros((nverts,))
          edge_ij[i] = -1
          edge_ij[j] = 1
          edges.append(edge_ij)
          weights.append((vert_cots[i, j] + vert_cots[j, i]) / 2)

          edge_jk = np.zeros((nverts,))
          edge_jk[j] = -1
          edge_jk[k] = 1
          edges.append(edge_jk)
          weights.append((vert_cots[j, k] + vert_cots[k, j]) / 2)

          edge_ki = np.zeros((nverts,))
          edge_ki[k] = -1
          edge_ki[i] = 1
          edges.append(edge_ki)
          weights.append((vert_cots[k, i] + vert_cots[i, k]) / 2)

      vert_neighborhood.append(np.array(edges).T)
      vert_neighborhood_weights.append(np.array(weights))

    return vert_neighborhood, vert_neighborhood_weights

  # CORRECT DON'T TOUCH
  def _calc_vert_cots(self, verts, faces):
    nverts, _ = verts.shape

    vert_cots = np.zeros((nverts, nverts))

    for face in faces:
      i, j, k = face
      vert_i, vert_j, vert_k = verts[i, :], verts[j, :], verts[k, :]

      edge_ij = vert_j - vert_i
      edge_jk = vert_k - vert_j
      edge_ki = vert_i - vert_k

      vert_cots[i, j] = np.dot(edge_ki, -edge_jk) / np.linalg.norm(np.cross(edge_ki, -edge_jk))
      vert_cots[j, k] = np.dot(edge_ij, -edge_ki) / np.linalg.norm(np.cross(edge_ij, -edge_ki))
      vert_cots[k, i] = np.dot(edge_jk, -edge_ij) / np.linalg.norm(np.cross(edge_jk, -edge_ij))

    return vert_cots


if __name__ == "__main__":
  optim = Optimizer("./assets/tetrahedron.obj", "./assets/cube.obj", "./out.obj")
  optim.run(0)
