import numpy as np
from scipy.sparse import csc_array
from sksparse.cholmod import cholesky
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
    nverts, _ = self._input_verts.shape

    # NOTE: face normals for analogy shape, vertex normals for input shape
    analogy_face_normals = self._calc_face_normals(self._analogy_verts, self._analogy_faces)
    input_vert_normals = self._calc_vert_normals(self._input_verts, self._input_faces)

    # "target" vert normals
    output_vert_normals = self._calc_output_vert_normals(analogy_face_normals, input_vert_normals)

    input_vert_areas = self._calc_vert_areas(self._input_verts, self._input_faces)

    # T = {tk} is a set of target normals for each vertex k
    # V is a |V|-by-3 matrix of rest vertices
    # F is a |F|-by-3 matrix of face lists
    # V' is a |V|-by-3 matrix of deformed vertices
    #
    
    A, W = self._calc_vert_neighborhoods(self._input_verts, self._input_faces)
    # A[k] is a |V|-by-|Nk| directed incidence matrix such that Nk is V.T @ A[k]

    # there might be error here?
    Q = np.zeros((nverts, nverts))
    for k in range(nverts):
      Q += A[k] @ np.diag(W[k]) @ A[k].T


    factor_ = cholesky(csc_array(Q), ordering_method='amd')
    L = factor_.L()
    L = L.toarray()

    # L = np.linalg.cholesky(Q)

    C = self._calc_vert_cots(self._input_verts, self._input_faces)
    # self._calc_edge_cots
    # self._calc_edge_weights

    # assert np.all(np.allclose(Q, Q.T))  # Q is a |V|-by-|V| symmetric matrix
    # still not sure why negative

    K = np.zeros((nverts * 3, nverts))  # K is a |9V|-by-|3V| matrix stacking the constant terms
    for k in range(nverts):
      K[3 * k : 3 * (k + 1), :] = self._input_verts.T @ A[k] @ np.diag(W[k]) @ A[k].T

    R = np.hstack([np.eye(3)] * nverts)  # R is a 3-by-|3V| matrix concatenating the rotations

    for __ in range(100):
      # local-step
      for k in range(nverts):
        input_edges = np.hstack(
          [
            self._input_verts.T @ A[k],
            input_vert_normals[k, :].reshape((3, 1)),
          ]
        )
        output_edges = np.hstack(
          [
            self._output_verts.T @ A[k],
            output_vert_normals[k, :].reshape((3, 1)),
          ]
        )
        weights = np.diag(np.append(W[k], factor * input_vert_areas[k]))

        U, _, Vt = np.linalg.svd(input_edges @ weights @ output_edges.T)
        rotation = Vt.T @ U.T
        if np.linalg.det(rotation) < 0:
          Vt[0, :] *= -1
          rotation = Vt.T @ U.T
        R[:, 3 * k : 3 * (k + 1)] = rotation

        # assert np.all(np.isclose(rotation, np.eye(3)))

      # debugging global step -- factor is 0, so just minimize ARAP
      # ideal rotation should be no rotation (identity matrix)

      # global step
      # self._output_verts = np.linalg.solve(L.T, np.linalg.solve(L, K.T @ R.T))
      self._output_verts = factor_(K.T @ R.T)
      # assert np.all(np.isclose(Q @ self._output_verts, K.T @ R.T))

      # print(self._output_verts)
      # print(self._input_verts)

      # solver solution is really similar to the null solution -- numerical problems likely
      # constant is -288.0 <-- minimum achievable
      # but solver finds a different solution which also produces -288.0
      # there are multiple solutions here...
      # det(Q) is really close to 0, which is causing instability
      # So is there a mistake in Q or is it just the input shape?
      # cotan Laplacian -- why is this different?

      # energy = np.trace(self._output_verts.T @ Q @ self._output_verts) - 2 * np.trace(
      #   R @ K @ self._output_verts
      # )  # why is this neg? -- maybe because there's no constant
      # print(energy)

      energy = 0
      for k in range(nverts):
        E = self._input_verts.T @ A[k]
        E_prime = self._output_verts.T @ A[k]
        for _ in range(A[k].shape[1]):
          energy += W[k][_] * np.linalg.norm(
            R[:, 3 * k : 3 * (k + 1)] @ E[:, _] - E_prime[:, _]
          ) ** 2 - W[k][_] * (R[:, 3 * k : 3 * (k + 1)] @ E[:, _]).T @ (
            R[:, 3 * k : 3 * (k + 1)] @ E[:, _]
          )
      # print(energy)

    self._save_obj(self._output_path, self._output_verts, self._output_faces)

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
  # maybe there should be a preprocessor that makes meshes Delaunay
  optim = Optimizer("./assets/tetrahedron.obj", "./assets/sphere.obj", "./out.obj")
  optim.run(1000)

  # maybe a constraint should be locking in a single vertex?


  # a square matrix is PSD iff it's symmetric and its eigenvalues are non-negative
  
  # 
  # L encodes the connectivity and angles of the mesh geometry
  # x.T L x describes a scalar smoothness measure of the function x (e.g. position) across the mesh
  # If x is position, then x.TLx measures how much the vertices' positions deviate from their neighbors.
  # In Laplacian smoothing, x.T L x is minimized.

  # x.TLx = sum_{i,j} w_ij(x_i - x_j)^2