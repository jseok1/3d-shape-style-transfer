import numpy as np
from scipy.sparse import csc_array
from sksparse.cholmod import cholesky

# ASSUMPTION: MANIFOLD TRIANGLE MESHES

# # 1. position on input to position on unit sphere (Gauss map; using normals)
# # 2. position on unit sphere to analogy normal


class Stylizer:
  def run(self, analogy_path, input_path, output_path, strength):
    analogy_verts, analogy_faces = self._load_obj(analogy_path)
    input_verts, input_faces = self._load_obj(input_path)
    output_verts = np.copy(input_verts)

    input_nverts, _ = input_verts.shape

    analogy_face_normals = self._calc_face_normals(analogy_verts, analogy_faces)
    input_face_normals = self._calc_face_normals(input_verts, input_faces)
    input_face_areas = self._calc_face_areas(input_verts, input_faces)
    input_vert_normals = self._calc_vert_normals(
      input_verts, input_faces, input_face_normals, input_face_areas
    )
    input_vert_areas = self._calc_vert_areas(input_verts, input_faces, input_face_areas)
    input_vert_neighbors, input_vert_neighbor_weights = self._calc_vert_neighbors(
      input_verts, input_faces
    )
    target_vert_normals = self._calc_target_vert_normals(analogy_face_normals, input_vert_normals)

    # kinda slow: sparse matrix here
    Q = np.zeros((input_nverts, input_nverts))
    for k in range(input_nverts):
      Q += (
        input_vert_neighbors[k]
        @ np.diag(input_vert_neighbor_weights[k])
        @ input_vert_neighbors[k].T
      )

    # assert np.all(np.allclose(Q, Q.T))  # Q is a |V|-by-|V| symmetric matrix
    # assert positive eigenvalues

    factor = cholesky(csc_array(Q), ordering_method="amd")

    # K is a |9V|-by-|3V| matrix stacking the constant terms
    K = np.zeros((input_nverts * 3, input_nverts))
    for k in range(input_nverts):
      # repeated computation here - can speed up
      K[3 * k : 3 * (k + 1), :] = (
        input_verts.T
        @ input_vert_neighbors[k]
        @ np.diag(input_vert_neighbor_weights[k])
        @ input_vert_neighbors[k].T
      )

    # 3-by-|3V|
    rotations = np.hstack([np.eye(3)] * input_nverts)

    for __ in range(100):
      # local step
      self._calc_optimal_rotations(
        rotations,
        input_verts,
        output_verts,
        input_vert_normals,
        target_vert_normals,
        input_vert_neighbors,
        input_vert_neighbor_weights,
        input_vert_areas,
        strength,
      )

      # global step
      self._calc_optimal_output_verts(output_verts, rotations, K, factor)

      # energy = np.trace(self._output_verts.T @ Q @ self._output_verts) - 2 * np.trace(
      #   R @ K @ self._output_verts
      # )  # why is this neg? -- maybe because there's no constant
      # print(energy)

    self._save_obj(output_path, output_verts, input_faces)

  def _calc_target_vert_normals(self, analogy_face_normals, input_vert_normals):
    input_nverts, _ = input_vert_normals.shape

    target_vert_normals = np.zeros((input_nverts, 3))

    for i in range(input_nverts):
      j = np.argmax(np.dot(analogy_face_normals, input_vert_normals[i, :]))
      target_vert_normals[i, :] = analogy_face_normals[j, :]

    return target_vert_normals

  def _calc_optimal_rotations(
    self,
    rotations,
    input_verts,
    output_verts,
    input_vert_normals,
    target_vert_normals,
    input_vert_neighbors,
    input_vert_neighbor_weights,
    input_vert_areas,
    strength,
  ):
    input_nverts, _ = input_verts.shape

    # potential sparse optim, also do in parallel (?)
    for k in range(input_nverts):
      input_edges = np.hstack(
        [
          input_verts.T @ input_vert_neighbors[k],
          input_vert_normals[k, :].reshape((3, 1)),
        ]
      )
      output_edges = np.hstack(
        [
          output_verts.T @ input_vert_neighbors[k],
          target_vert_normals[k, :].reshape((3, 1)),
        ]
      )

      U, _, Vt = np.linalg.svd(
        input_edges
        @ np.diag(np.append(input_vert_neighbor_weights[k], strength * input_vert_areas[k]))
        @ output_edges.T
      )
      rotation = Vt.T @ U.T
      if np.linalg.det(rotation) < 0:
        Vt[0, :] *= -1
        rotation = Vt.T @ U.T
      rotations[:, 3 * k : 3 * (k + 1)] = rotation

  def _calc_optimal_output_verts(self, output_verts, rotations, K, factor):
    output_verts[:, :] = factor(K.T @ rotations.T)

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

  def _save_obj(self, path, verts, faces):
    lines = []

    for vert in verts:
      lines.append(f"v {vert[0]} {vert[1]} {vert[2]}")

    for face in faces:
      lines.append(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}")

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
  def _calc_vert_areas(self, verts, faces, face_areas):
    nverts, _ = verts.shape
    nfaces, _ = faces.shape

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
  def _calc_vert_normals(self, verts, faces, face_normals, face_areas):
    nverts, _ = verts.shape
    nfaces, _ = faces.shape

    vert_normals = np.zeros((nverts, 3))

    for v in range(nverts):
      vert_normal = np.zeros((3,))

      for f in range(nfaces):
        if v in faces[f, :]:
          vert_normal += face_areas[f] * face_normals[f, :]

      vert_normal /= np.linalg.norm(vert_normal)
      vert_normals[v, :] = vert_normal

    return vert_normals

  # CORRECT DON'T TOUCH
  def _calc_vert_neighbors(self, verts, faces):
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
  # maybe there should be a preprocessor that makes meshes Delaunay
  stylizer = Stylizer()
  stylizer.run("./assets/tetrahedron.obj", "./assets/sphere.obj", "./out.obj", 1000)

  # maybe a constraint should be locking in a single vertex?

  # a square matrix is PSD iff it's symmetric and its eigenvalues are non-negative

  #
  # L encodes the connectivity and angles of the mesh geometry
  # x.T L x describes a scalar smoothness measure of the function x (e.g. position) across the mesh
  # If x is position, then x.TLx measures how much the vertices' positions deviate from their neighbors.
  # In Laplacian smoothing, x.T L x is minimized.

  # x.TLx = sum_{i,j} w_ij(x_i - x_j)^2
