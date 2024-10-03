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


class Object:
  def __init__(self, vertices, faces, vertex_normals, face_normals, face_areas, vertex_neighbors):
    self.vertices = vertices
    self.normals = vertex_normals
    self.faces = faces
    self.neighbors = []

    self._face_normals = face_normals
    self._face_areas = face_areas


def load_obj(path):
  vertices = []  # |V| x 3
  faces = []  # |F| x 3 (intended for triangular meshes)
  vertex_normals = []  # |V| x 3 (vertex normals computed as area-weighted face normals)
  face_normals = []  # |F| x 3
  face_areas = []  # |F| x 1                                     MAYBE use as crude Voronoi area approx
  face_angles = []  # |F| x 3 (interior angle at vertex i, j, k of this face)
  vertex_neighbor_edges = []  # |V| x |V| x |Nk|
  weights = []  # |V| x |Nk| x |Nk|

  # some of these are one-time computations, which should be separated into other methods?
  # you load faces and vertices, which are the minimum info needed to define a mesh
  # then everything you calculate dynamically

  with open(path, "r") as file:
    for line in file:
      if line.startswith("v "):
        parts = line.split()
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        vertex = [x, y, z]
        vertices.append(vertex)

      elif line.startswith("f "):
        parts = line.split()
        pi, pj, pk = (
          int(parts[1].split("/")[0]) - 1,
          int(parts[2].split("/")[0]) - 1,
          int(parts[3].split("/")[0]) - 1,
        )
        face = [pi, pj, pk]
        faces.append(face)

    vertices = np.array(vertices)
    faces = np.array(faces)

    nvertices, _ = vertices.shape
    nfaces, _ = faces.shape

    for i in range(nfaces):
      face = faces[i, :]
      pi, pj, pk = vertices[face[0]], vertices[face[1]], vertices[face[2]]

      e_ij = pj - pi
      e_jk = pk - pj
      e_ki = pi - pk

      cross_prod = np.cross(-e_ij, e_jk)
      face_normal = cross_prod / np.linalg.norm(cross_prod)
      face_normals.append(face_normal)

      face_area = np.linalg.norm(cross_prod) / 2
      face_areas.append(face_area)

      alpha_k = np.arccos(np.dot(-e_jk, e_ki) / (np.linalg.norm(e_jk) * np.linalg.norm(e_ki)))
      alpha_i = np.arccos(np.dot(-e_ki, e_ij) / (np.linalg.norm(e_ki) * np.linalg.norm(e_ij)))
      alpha_j = np.arccos(np.dot(-e_ij, e_jk) / (np.linalg.norm(e_ij) * np.linalg.norm(e_jk)))

      face_angles.append([alpha_i, alpha_j, alpha_k])

    face_normals = np.array(face_normals)
    face_areas = np.array(face_areas)
    face_angles = np.array(face_angles)

    g_weights = []
    for i in range(nvertices):
      vertex_normal = np.zeros((3,))

      # maybe also Voronoi area here
      edges = []
      weights = []
      for j in range(nfaces):
        face = faces[j, :]
        angles = face_angles[j, :]
        voronoi = 0
        if i in face:
          vertex_normal += face_normals[j] * face_areas[j]

          # DIRECTED EDGES (shared edges between faces counted twice)
          # tip - tail
          _i, _j, _k = face
          alpha_i, alpha_j, alpha_k = angles

          edge_ij = np.zeros((nvertices,))
          edge_ij[_j] = 1
          edge_ij[_i] = -1
          edges.append(edge_ij)
          alpha_ij = alpha_k
          beta_ij = None
          for k in range(nfaces):
            if _i == faces[k, 0] and _j == faces[k, 2]:
              beta_ij = face_angles[k, 1]
              break
            elif _i == faces[k, 1] and _j == faces[k, 0]:
              beta_ij = face_angles[k, 2]
              break
            elif _i == faces[k, 2] and _j == faces[k, 1]:
              beta_ij = face_angles[k, 0]
              break
          weight_ij = (1 / np.tan(alpha_ij) + 1 / np.tan(beta_ij)) / 2
          weights.append(weight_ij)

          edge_jk = np.zeros((nvertices,))
          edge_jk[_k] = 1
          edge_jk[_j] = -1
          edges.append(edge_jk)
          alpha_jk = alpha_i
          beta_jk = None
          for k in range(nfaces):
            if _j == faces[k, 2] and _k == faces[k, 1]:
              beta_jk = face_angles[k, 0]
              break
            elif _j == faces[k, 0] and _k == faces[k, 2]:
              beta_jk = face_angles[k, 1]
              break
            elif _j == faces[k, 1] and _k == faces[k, 0]:
              beta_jk = face_angles[k, 2]
              break
          weight_jk = (1 / np.tan(alpha_jk) + 1 / np.tan(beta_jk)) / 2
          weights.append(weight_jk)

          edge_ki = np.zeros((nvertices,))
          edge_ki[_i] = 1
          edge_ki[_k] = -1
          edges.append(edge_ki)
          alpha_ki = alpha_j
          beta_ki = None
          for k in range(nfaces):
            if _k == faces[k, 2] and _i == faces[k, 1]:
              beta_ki = face_angles[k, 0]
              break
            elif _k == faces[k, 0] and _i == faces[k, 2]:
              beta_ki = face_angles[k, 1]
              break
            elif _k == faces[k, 1] and _i == faces[k, 0]:
              beta_ki = face_angles[k, 2]
              break
          weight_ki = (1 / np.tan(alpha_ki) + 1 / np.tan(beta_ki)) / 2
          weights.append(weight_ki)

          voronoi 

      weights = np.diag(weights)
      g_weights.append(weights)

      vertex_normal /= np.linalg.norm(vertex_normal)
      vertex_normals.append(vertex_normal)

      vertex_neighbor_edges.append(np.array(edges).T)

    vertex_normals = np.array(vertex_normals)

    # cotan weights can also be calculated here
    # so can Voronoi area

    # the only thing that needs to be recalculated is E'

    return (
      vertices,
      faces,
      vertex_normals,
      face_normals,
      face_areas,
      vertex_neighbor_edges,
      face_angles,
      g_weights,
    )


def save_obj(path, vertices, faces, normals, neighbors):
  pass


# idea - |V| x |V| incidence matrix that picks out |Nk| for each vert --> actually no this doesn't work as maybe |Nk| > |V|
# Ak - |V| x |Nk|
# (V^T)Ak - |3| x |Nk| - edge vectors


vertices, faces, vertex_normals, face_normals, face_areas, neighbors, face_angles, weights = (
  load_obj("./assets/cube.obj")
)
# input_vertices, input_faces, input_normals, input_neighbors = load_obj("./assets/cube.obj")
# V_prime = np.copy(V)
print(vertices)
print(faces)
# print(face_normals)
# print(vertex_normals)

example_incidence = vertices.T @ neighbors[0]
print(neighbors[0].shape)

# [[-1.  0.  1. -1.  0.  1. -1.  0.  1. -1.  0.  1. -1.  0.  1. -1.  0.  1.]
#  [ 0.  1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1. -1.  0.  0.  0.  0.]
#  [ 1. -1.  0.  0.  1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
#  [ 0.  0.  0.  1. -1.  0.  0.  0.  0.  0.  1. -1.  0.  0.  0.  0.  0.  0.]
#  [ 0.  0.  0.  0.  0.  0.  1. -1.  0.  0.  0.  0.  0.  0.  0.  0.  1. -1.]
#  [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  1. -1.  1. -1.  0.]
#  [ 0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
#  [ 0.  0.  0.  0.  0.  0.  0.  1. -1.  1. -1.  0.  0.  0.  0.  0.  0.  0.]]
#    1  [2]  3   4  [5]  1   6  [7]  8   8  [9]  4   3 [10] 11  11 [12]  6
# 6 triangles incident to this vertex
# 18 edges -> 12 repeated (but in different directions), 6 not repeated

print(weights[0].shape)

# weights w_ij = 1 / 2 * (cot alpha + cot beta)
#


# step 1: map unit sphere normal to analogy shape FACE normal
# step 2: map input shape VERTEX normal to unit sphere normal (trivial using Gauss map)
# step 3:


# MIXED VORONOI CELL AREA
# 1-ring neighborhood
# acute -> circumcenter (perpendicular bisectors intersect at circumcenter inside the triangle)
# ...
# obtuse -> centroid (circumcetner outside the triangle, so take area of barycenter, which is 1/3 area?)
#
# barycenter = (i + j + k)/ 3
# np.linalg.norm(np.cross(e_jk, barycenter - j)) / 2

# Voronoi area of a triangular mesh can be calculated with:
# for each triangle T:
#   if acute:
#     area += 1 / 8 * (||v1 - v2||^2 cot(alpha) + ||v2 - v||^2 cot(beta) + ||v - v1||^2 cot(gamma))
#   if obtuse:
#     the vertex opposite the obtuse area gets the entire area (perpendicular bisectors meet outside)
# where alpha, beta, gamma are angles opposite to v, v1, v2 in triangle T
# the idea is to use perpendicular bisectors as voroni areas (???)
#
# optimization is min_V' sum_{k in V} E(vk, vk') + lambda * voronoi_k * ||normal_k - target_normal_k||^2
#
# ARAP: sum_{i,j in spokes+rims} w_ij ||Rk * eij - eij'||^2
#
# cotangent weighting factor is w_ij = 1 / 2 * (cot alpha + cot beta)

# solving is local/global steps
# local is orthogonal procrustes problem: SVD solution for R_k
# global is linear
#
#
# What do you need:
# vertex positions
# incidence matrix for each vertex (spokes and rims) --> half edge
# cotangent weights -- need angles
#

# Need V' and R_k
#
# V matrix (easy)
# both of these require neighborhood info


# for each vertex k:
# E_k and E_k' matrices: 3 x |N_k| of edge vectors around vertex k
# W_k matrix: |N_k| x |N_k| of cotangent weights for edge incidence around vertex k
# t_k vector: (unit) target normal vector for vertex k
# n_k vector: (unit) normal vector for vertex k
# a_k scalar: Voronoi area for vertex k (need to get angles -- expensive)

# Taking the SVD is an independent process so just do it in parallel.

# A_k matrix:


# compute N~_A'
# compute T from n^(V) <-- vertex normal of input computed via area-weighted average of face normals
# n^ = n^(V)
# Q, K = precompute(V, F)
# V' = V
# while not converge do
#     R = local-step(V', n^, T, lambda)
#     V' = global-step(R, Q, K)
