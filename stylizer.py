import numpy as np

from scipy.sparse import csc_array, diags, vstack
from sksparse.cholmod import cholesky
from tqdm import tqdm


class Stylizer:
  def run(self, reference_path, input_path, output_path, gamma):
    reference_verts, reference_faces, _, _ = self._load_obj(reference_path)
    reference_face_normals = self._calc_face_normals(reference_verts, reference_faces)

    input_verts, input_faces, input_texture_verts, input_texture_faces = self._load_obj(input_path)
    input_face_areas = self._calc_face_areas(input_verts, input_faces)
    input_face_normals = self._calc_face_normals(input_verts, input_faces)
    input_edge_cots = self._calc_edge_cots(input_verts, input_faces)
    input_vert_neighbor_face_masks = self._calc_vert_neighbor_face_masks(input_verts, input_faces)
    input_vert_neighbor_edge_masks, weights = self._calc_vert_neighbor_edge_masks(
      input_verts, input_faces, input_vert_neighbor_face_masks, input_edge_cots
    )
    input_vert_areas = self._calc_vert_areas(
      input_verts, input_faces, input_vert_neighbor_face_masks, input_face_areas
    )
    input_vert_normals = self._calc_vert_normals(
      input_verts, input_faces, input_vert_neighbor_face_masks, input_face_areas, input_face_normals
    )

    output_verts = np.copy(input_verts)
    target_vert_normals = self._calc_target_vert_normals(reference_face_normals, input_vert_normals)

    input_nverts, _ = input_verts.shape

    # TODO: repeated calculation: input_vert_neighbor_edge_masks[k] @ diags(weights[k]) @ input_vert_neighbor_edge_masks[k].T
    input_cot_laplacian = csc_array((input_nverts, input_nverts))
    for k in range(input_nverts):
      input_cot_laplacian += (
        input_vert_neighbor_edge_masks[k] @ diags(weights[k]) @ input_vert_neighbor_edge_masks[k].T
      )
    factor = cholesky(input_cot_laplacian, ordering_method="amd")

    K = vstack(
      [
        csc_array(
          input_verts.T
          @ input_vert_neighbor_edge_masks[k]
          @ diags(weights[k])
          @ input_vert_neighbor_edge_masks[k].T
        )
        for k in range(input_nverts)
      ]
    )

    rotations = np.hstack([np.eye(3)] * input_nverts)

    for _ in tqdm(range(20)):
      # local step
      self._local_step(
        rotations,
        input_verts,
        output_verts,
        input_vert_normals,
        target_vert_normals,
        input_vert_neighbor_edge_masks,
        weights,
        input_vert_areas,
        gamma,
      )

      # global step
      self._global_step(output_verts, rotations, K, factor)

      # energy = np.trace(output_verts.T @ input_cot_laplacian @ output_verts) - 2 * np.trace(
      #   rotations @ K @ output_verts
      # )

    self._save_obj(output_path, output_verts, input_faces, input_texture_verts, input_texture_faces)

  def _calc_target_vert_normals(self, reference_face_normals, input_vert_normals):
    input_nverts, _ = input_vert_normals.shape

    target_vert_normals = np.zeros((input_nverts, 3))

    for i in range(input_nverts):
      j = np.argmax(np.dot(reference_face_normals, input_vert_normals[i, :]))
      target_vert_normals[i, :] = reference_face_normals[j, :]

    return target_vert_normals

  def _local_step(
    self,
    rotations,
    input_verts,
    output_verts,
    input_vert_normals,
    target_vert_normals,
    input_vert_neighbor_edge_masks,
    weights,
    input_vert_areas,
    gamma,
  ):
    input_nverts, _ = input_verts.shape

    for k in range(input_nverts):
      input_vert_neighbor_edges = np.hstack(
        [
          input_verts.T @ input_vert_neighbor_edge_masks[k],
          input_vert_normals[k, :].reshape((3, 1)),
        ]
      )
      output_vert_neighbor_edges = np.hstack(
        [
          output_verts.T @ input_vert_neighbor_edge_masks[k],
          target_vert_normals[k, :].reshape((3, 1)),
        ]
      )

      U, _, Vh = np.linalg.svd(
        input_vert_neighbor_edges
        @ diags(np.append(weights[k], gamma * input_vert_areas[k]))
        @ output_vert_neighbor_edges.T
      )
      V = Vh.T
      rotation = V @ U.T
      if np.linalg.det(rotation) < 0:
        V[:, 0] *= -1
        rotation = V @ U.T
      rotations[:, 3 * k : 3 * (k + 1)] = rotation

  def _global_step(self, output_verts, rotations, K, factor):
    output_verts[:, :] = factor(K.T @ rotations.T)

  def _load_obj(self, path):
    verts = []
    faces = []
    texture_verts = []
    texture_faces = []

    with open(path, "r") as file:
      lines = file.readlines()
      for line in lines:
        if line.startswith("v "):
          tokens = line.split()
          verts.append(tokens[1:4])

        elif line.startswith("vt "):
          tokens = line.split()
          texture_verts.append(tokens[1:3])

        elif line.startswith("f "):
          tokens = [token.split("/") for token in line.split()]
          faces.append([token[0] for token in tokens[1:4]])

          if (
            len(tokens[1]) > 1
            and tokens[1][1]
            and len(tokens[2]) > 1
            and tokens[2][1]
            and len(tokens[3]) > 1
            and tokens[3][1]
          ):
            texture_faces.append([token[1] for token in tokens[1:4]])

    verts = np.array(verts, dtype=np.float64)
    faces = np.array(faces, dtype=np.int64)
    texture_verts = np.array(texture_verts, dtype=np.float64)
    texture_faces = np.array(texture_faces, dtype=np.int64)

    faces -= 1
    texture_faces -= 1

    return verts, faces, texture_verts, texture_faces

  def _save_obj(self, path, verts, faces, texture_verts, texture_faces):
    nverts, _ = verts.shape
    nfaces, _ = faces.shape
    texture_nverts, _ = texture_verts.shape
    texture_nfaces, _ = texture_faces.shape

    faces += 1
    texture_faces += 1

    lines = []

    for k in range(nverts):
      line = f"v {verts[k, 0]} {verts[k, 1]} {verts[k, 2]}\n"
      lines.append(line)

    for k in range(texture_nverts):
      line = f"vt {texture_verts[k, 0]} {texture_verts[k, 1]}\n"
      lines.append(line)

    for k in range(nfaces):
      if texture_nfaces:
        line = f"f {faces[k, 0]}/{texture_faces[k, 0]} {faces[k, 1]}/{texture_faces[k, 1]} {faces[k, 2]}/{texture_faces[k, 2]}\n"
      else:
        line = f"f {faces[k, 0]} {faces[k, 1]} {faces[k, 2]}\n"
      lines.append(line)

    with open(path, "w") as file:
      file.writelines(lines)

  def _calc_face_areas(self, verts, faces):
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    face_areas = (
      np.linalg.norm(np.cross(verts[i, :] - verts[j, :], verts[k, :] - verts[j, :]), axis=1) / 2
    )

    return face_areas

  def _calc_face_normals(self, verts, faces):
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    face_normals = np.cross(verts[i, :] - verts[j, :], verts[k, :] - verts[j, :])
    face_normals /= np.linalg.norm(face_normals, axis=1).reshape((-1, 1))

    return face_normals

  # TODO: mixed Voronoi area
  def _calc_vert_areas(self, verts, faces, vert_neighbor_face_masks, face_areas):
    nverts, _ = verts.shape

    vert_areas = np.zeros((nverts,))

    for k in range(nverts):
      vert_neighbor_face_areas = vert_neighbor_face_masks[k].T @ face_areas
      vert_areas[k] = np.sum(vert_neighbor_face_areas, axis=0) / 3

    return vert_areas

  def _calc_vert_normals(self, verts, faces, vert_neighbor_face_masks, face_areas, face_normals):
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

      vert_neighbor_face_masks.append(csc_array((entries, (y, x)), shape=(nfaces, vert_nfaces)))

    return vert_neighbor_face_masks

  def _calc_vert_neighbor_edge_masks(self, verts, faces, vert_neighbor_face_masks, edge_cots):
    nverts, _ = verts.shape

    vert_neighbor_edge_masks = []
    weights = []

    for k in range(nverts):
      vert_neighbor_faces = vert_neighbor_face_masks[k].T @ faces

      tips = vert_neighbor_faces.flatten()
      tails = vert_neighbor_faces[:, [1, 2, 0]].flatten()

      y = np.vstack([tails, tips]).T.flatten()
      (vert_nedges,) = y.shape
      vert_nedges //= 2

      x = np.repeat(np.arange(vert_nedges), 2)
      entries = np.vstack([[-1] * vert_nedges, [1] * vert_nedges]).T.flatten()

      edge_cots_alpha = edge_cots[tips, tails]
      edge_cots_beta = edge_cots[tails, tips]

      vert_neighbor_edge_masks.append(csc_array((entries, (y, x)), shape=(nverts, vert_nedges)))
      weights.append((edge_cots_alpha + edge_cots_beta) / 2)

    return vert_neighbor_edge_masks, weights

  def _calc_edge_cots(self, verts, faces):
    nverts, _ = verts.shape

    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]

    edge_cots = np.zeros((nverts, nverts))

    edge_cots[i, j] = np.sum(
      (verts[i, :] - verts[k, :]) * (verts[j, :] - verts[k, :]), axis=1
    ) / np.linalg.norm(np.cross(verts[i, :] - verts[k, :], verts[j, :] - verts[k, :]), axis=1)
    edge_cots[j, k] = np.sum(
      (verts[j, :] - verts[i, :]) * (verts[k, :] - verts[i, :]), axis=1
    ) / np.linalg.norm(np.cross(verts[j, :] - verts[i, :], verts[k, :] - verts[i, :]), axis=1)
    edge_cots[k, i] = np.sum(
      (verts[k, :] - verts[j, :]) * (verts[i, :] - verts[j, :]), axis=1
    ) / np.linalg.norm(np.cross(verts[k, :] - verts[j, :], verts[i, :] - verts[j, :]), axis=1)

    return edge_cots
