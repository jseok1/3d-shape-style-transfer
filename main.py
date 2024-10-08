import numpy as np
from scipy.sparse import csr_array, csc_array, lil_array, diags, hstack, vstack
from sksparse.cholmod import cholesky
from tqdm import tqdm

import timeit

# ASSUMPTION: MANIFOLD TRIANGLE MESHES

# # 1. position on input to position on unit sphere (Gauss map; using normals)
# # 2. position on unit sphere to analogy normal


class Stylizer:
  def run(self, analogy_path, input_path, output_path, strength):
    analogy_verts, analogy_faces = self._load_obj(analogy_path)
    analogy_face_normals = self._calc_face_normals(analogy_verts, analogy_faces)

    input_verts, input_faces = self._load_obj(input_path)
    vert_neighbor_face_masks = self._calc_vert_neighbor_face_masks(input_verts, input_faces)
    input_vert_neighbors, weights = self._calc_vert_neighbor_edge_masks(
      input_verts, input_faces, vert_neighbor_face_masks
    )
    input_face_normals = self._calc_face_normals(input_verts, input_faces)
    input_face_areas = self._calc_face_areas(input_verts, input_faces)
    input_vert_normals = self._calc_vert_normals(
      input_verts, input_faces, vert_neighbor_face_masks, input_face_normals, input_face_areas
    )
    input_vert_areas = self._calc_vert_areas(
      input_verts, input_faces, vert_neighbor_face_masks, input_face_areas
    )

    output_verts = np.copy(input_verts)
    target_vert_normals = self._calc_target_vert_normals(analogy_face_normals, input_vert_normals)

    input_nverts, _ = input_verts.shape

    input_cot_laplacian = csr_array((input_nverts, input_nverts))
    for k in range(input_nverts):
      input_cot_laplacian += input_vert_neighbors[k] @ diags(weights[k]) @ input_vert_neighbors[k].T
    factor = cholesky(input_cot_laplacian, ordering_method="amd")

    # K is a |9V|-by-|3V| matrix stacking the constant terms
    K = vstack(
      [
        csc_array(
          input_verts.T @ input_vert_neighbors[k] @ diags(weights[k]) @ input_vert_neighbors[k].T
        )
        for k in range(input_nverts)
      ]
    )

    # 3-by-|3V|
    rotations = np.hstack([np.eye(3)] * input_nverts)

    for _ in tqdm(range(100)):
      # local step
      self._local_step(
        rotations,
        input_verts,
        output_verts,
        input_vert_normals,
        target_vert_normals,
        input_vert_neighbors,
        weights,
        input_vert_areas,
        strength,
      )

      # global step
      self._global_step(output_verts, rotations, K, factor)

      # energy = np.trace(self._output_verts.T @ cot_laplacian @ self._output_verts) - 2 * np.trace(
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

  def _local_step(
    self,
    rotations,
    input_verts,
    output_verts,
    input_vert_normals,
    target_vert_normals,
    input_vert_neighbors,
    weights,
    input_vert_areas,
    strength,
  ):
    input_nverts, _ = input_verts.shape

    for k in range(input_nverts):
      input_vert_neighbor_edges = np.hstack(
        [
          input_verts.T @ input_vert_neighbors[k],
          input_vert_normals[k, :].reshape((3, 1)),
        ]
      )
      output_vert_neighbor_edges = np.hstack(
        [
          output_verts.T @ input_vert_neighbors[k],
          target_vert_normals[k, :].reshape((3, 1)),
        ]
      )

      U, _, Vt = np.linalg.svd(
        input_vert_neighbor_edges
        @ diags(np.append(weights[k], strength * input_vert_areas[k]))
        @ output_vert_neighbor_edges.T
      )
      rotation = Vt.T @ U.T
      if np.linalg.det(rotation) < 0:
        Vt[0, :] *= -1
        rotation = Vt.T @ U.T
      rotations[:, 3 * k : 3 * (k + 1)] = rotation

  def _global_step(self, output_verts, rotations, K, factor):
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

  def _save_obj(self, path, verts, faces):
    lines = []

    for vert in verts:
      lines.append(f"v {vert[0]} {vert[1]} {vert[2]}")

    for face in faces:
      lines.append(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}")

    with open(path, "w") as file:
      file.write("\n".join(lines))

  def _calc_face_areas(self, verts, faces):
    verts_ijk = verts[faces]
    verts_i = verts_ijk[:, 0]
    verts_j = verts_ijk[:, 1]
    verts_k = verts_ijk[:, 2]

    face_areas = np.linalg.norm(np.cross(verts_j - verts_i, verts_k - verts_j), axis=1) / 2

    return face_areas

  # TODO: mixed Voronoi area
  def _calc_vert_areas(self, verts, faces, vert_neighbor_face_masks, face_areas):
    nverts, _ = verts.shape

    vert_areas = np.zeros((nverts,))

    for k in range(nverts):
      vert_neighbor_face_areas = vert_neighbor_face_masks[k].T @ face_areas
      vert_areas[k] = np.sum(vert_neighbor_face_areas / 3, axis=0)

    return vert_areas

  def _calc_face_normals(self, verts, faces):
    verts_ijk = verts[faces]
    verts_i = verts_ijk[:, 0]
    verts_j = verts_ijk[:, 1]
    verts_k = verts_ijk[:, 2]

    face_normals = np.cross(verts_i - verts_j, verts_k - verts_j)
    face_normals /= np.linalg.norm(face_normals, axis=1).reshape((-1, 1))

    return face_normals

  def _calc_vert_normals(self, verts, faces, vert_neighbor_face_masks, face_normals, face_areas):
    nverts, _ = verts.shape

    vert_normals = np.zeros((nverts, 3))

    for k in range(nverts):
      vert_neighbor_face_areas = vert_neighbor_face_masks[k].T @ face_areas
      vert_neighbor_face_normals = vert_neighbor_face_masks[k].T @ face_normals

      vert_normals[k, :] = np.sum(
        vert_neighbor_face_areas.reshape((-1, 1)) * vert_neighbor_face_normals,
        axis=0,
      )
      vert_normals[k, :] /= np.linalg.norm(vert_normals[k, :])

    return vert_normals

  def _calc_vert_neighbor_face_masks(self, verts, faces):
    nverts, _ = verts.shape
    nfaces, _ = faces.shape

    vert_neighbor_face_masks = []
    for k in range(nverts):
      y, _ = np.where(k == faces)
      (vert_nfaces,) = y.shape

      x = np.arange(vert_nfaces)
      entries = [1] * vert_nfaces

      vert_neighbor_face_masks.append(csr_array((entries, (y, x)), shape=(nfaces, vert_nfaces)))

    # vert_neighbor_face_masks = np.any(np.arange(nverts).reshape((-1, 1, 1)) == faces, axis=2)
    return vert_neighbor_face_masks

  def _calc_vert_neighbor_edge_masks(self, verts, faces, vert_neighbor_face_masks):
    nverts, _ = verts.shape

    vert_neighbor_edge_masks = []
    weights = []

    edge_cots = self._calc_edge_cots(verts, faces)

    for k in range(nverts):
      vert_neighbor_faces = vert_neighbor_face_masks[k].T @ faces

      tips = vert_neighbor_faces.flatten()
      tails = vert_neighbor_faces[:, [1, 2, 0]].flatten()

      y = np.vstack([tails, tips]).T.flatten()
      (vert_nedges,) = y.shape
      vert_nedges //= 2

      x = np.repeat(np.arange(vert_nedges), 2)
      entries = np.vstack([[-1] * vert_nedges, [1] * vert_nedges]).T.flatten()

      vert_neighbor_edge_masks.append(csr_array((entries, (y, x)), shape=(nverts, vert_nedges)))
      weights.append((edge_cots[tips, tails] + edge_cots[tails, tips]) / 2)

    return vert_neighbor_edge_masks, weights

  def _calc_edge_cots(self, verts, faces):
    nverts, _ = verts.shape

    edge_cots = np.zeros((nverts, nverts))

    for face in faces:
      i, j, k = face
      vert_i, vert_j, vert_k = verts[i, :], verts[j, :], verts[k, :]

      edge_ij = vert_j - vert_i
      edge_jk = vert_k - vert_j
      edge_ki = vert_i - vert_k

      edge_cots[i, j] = np.dot(edge_ki, -edge_jk) / np.linalg.norm(np.cross(edge_ki, -edge_jk))
      edge_cots[j, k] = np.dot(edge_ij, -edge_ki) / np.linalg.norm(np.cross(edge_ij, -edge_ki))
      edge_cots[k, i] = np.dot(edge_jk, -edge_ij) / np.linalg.norm(np.cross(edge_jk, -edge_ij))

    return edge_cots

    nverts, _ = verts.shape

    vert_cots = np.zeros((nverts, nverts))

    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    vert_cots[i, j] = np.dot(verts[i, :] - verts[k, :], verts[j, :] - verts[k, :]) / np.linalg.norm(
      np.cross(verts[i, :] - verts[k, :], verts[j, :] - verts[k, :])
    )
    vert_cots[j, k] = np.dot(verts[j, :] - verts[i, :], verts[k, :] - verts[i, :]) / np.linalg.norm(
      np.cross(verts[j, :] - verts[i, :], verts[k, :] - verts[i, :])
    )
    vert_cots[k, i] = np.dot(verts[k, :] - verts[j, :], verts[i, :] - verts[j, :]) / np.linalg.norm(
      np.cross(verts[k, :] - verts[j, :], verts[i, :] - verts[j, :])
    )


if __name__ == "__main__":
  # maybe there should be a preprocessor that makes meshes Delaunay
  stylizer = Stylizer()
  # stylizer.run("./assets/tetrahedron.obj", "./assets/spot/spot_triangulated.obj", "./spot-tetrahedron.obj", 10)
  stylizer.run("./assets/cube.obj", "./assets/sphere.obj", "./out.obj", 100)

  # maybe a constraint should be locking in a single vertex?

  # a square matrix is PSD iff it's symmetric and its eigenvalues are non-negative

  #
  # L encodes the connectivity and angles of the mesh geometry
  # x.T L x describes a scalar smoothness measure of the function x (e.g. position) across the mesh
  # If x is position, then x.TLx measures how much the vertices' positions deviate from their neighbors.
  # In Laplacian smoothing, x.T L x is minimized.

  # x.TLx = sum_{i,j} w_ij(x_i - x_j)^2

  # csr_array A --> A v
  # csc_array A --> v A or A.T v
