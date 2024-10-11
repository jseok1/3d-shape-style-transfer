# 3D Shape Style Transfer

[INSERT IMAGE]

## Introduction

This is a tool that stylizes 3D shapes based on the techniques introduced in the paper _Normal-Driven Spherical Shape Analogies_ by Liu and Jacobson. The idea behind their approach is to use analogies of the form "$A$ _is to_ $A^\prime$ _as_ $B$ _is to_ $B^\prime$" to stylize an input shape based on a reference shape. In this framework, $A$ is a unit sphere, $A^\prime$ is the reference shape, $B$ is the input shape, and $B^\prime$ is the output shape. The fundamental insight to this approach is that surface normals capture the style of a shape. If two shapes have the same surface normals "up to analogy", then those two shapes have the same style. Let $N_A$, $N_{A^\prime}$, $N_B$, and $N_{B^\prime}$ be the surface normals of $A$, $A^\prime$, $B$, and $B^\prime$, respectively. The aim of this approach is to construct $B^\prime$ such that $N_{B^\prime}$ is the same as $N_{A^\prime}$ up to analogy.

Let $T$ be the target surface normals of $B^\prime$. That is, $T$ is the optimal $N_{B^\prime}$, since it may not be possible to construct $B^\prime$ such that $N_{B^\prime}$ is the same as $N_{A^\prime}$ up to analogy. Also, assume there is a mapping $\tilde{N}_{A^\prime} : A \rightarrow N_{A^\prime}$.

Then, $T$ can be constructed as follows.
1. Map each point on $B$ to a point on $A$ using the Gauss map.
2. Map those points on $A$ to surface normals on $A^\prime$ using $\tilde{N}_{A^\prime}$.
3. Assign those surface normals on $A^\prime$ as the target surface normals on $B^\prime$.

<!-- The choice of $\tilde{N}_{A^\prime}$ is important. -->

Then, $B^\prime$ can be constructed by deforming $B$ such that $B^\prime$ satisfies the following two conditions.
1. $N_{B^\prime}$ approximates $T$.
2. $B^\prime$ approximates the surface details of $B$. 

This is achieved by optimizing a carefully constructed energy. To make this concrete, assume $B$ and $B^\prime$ are manifold triangle meshes with vertices $V$ and $V^\prime$, respectively. The aim is to find the optimal $V^\prime$.

The first condition can be achieved by minimizing the squared Euclidean distance between the surface normals and the corresponding target surface normals. Let $\textbf{v}_k$ and $\textbf{v}_k^\prime$ be the $k$-th vertices in $V$ and $V^\prime$, respectively. Let $\hat{\textbf{n}}_k^\prime$ be the vertex normal of $\textbf{v}_k^\prime$ (i.e., the area-weighted average of the face normals in the one-ring neighborhood of $\textbf{v}_k^\prime$). Let $\textbf{t}_k$ be the target vertex normal of $\textbf{v}_k^\prime$. Finally, let $a_k$ be the Voronoi area of $\textbf{v}_k$. Then, this condition can be concretely formulated as follows.

```math
\argmin_{V^\prime} \sum_{k = 1}^{|V|} a_k\|\hat{\textbf{n}}_k^\prime - \textbf{t}_k\|^2
```

The second condition can be achieved by minimizing the as-rigid-as-possible (ARAP) energy, which regularizes the deformation by penalizing non-rigid transformations (i.e., transformations other than rotations and translations). The insight here is that rigid transformations preserve surface details. Again, let $\textbf{v}_k$ and $\textbf{v}_k^\prime$ be the $k$-th vertices in $V$ and $V^\prime$, respectively. Let $\textbf{v}_i$ and $\textbf{v}_j$ be a pair of adjacent vertices in the one-ring neighborhood of $\textbf{v}_k$, and let $\textbf{v}_i^\prime$ and $\textbf{v}_j^\prime$ be the corresponding pair of adjacent vertices in the one-ring neighborhood of $\textbf{v}_k^\prime$. Let $\textbf{e}_{ij} := \textbf{v}_j - \textbf{v}_i$ be the edge between $\textbf{v}_i$ and $\textbf{v}_j$, and let $\textbf{e}_{ij}^\prime := \textbf{v}_j^\prime - \textbf{v}_i^\prime$ be the edge between $\textbf{v}_i^\prime$ and $\textbf{v}_j^\prime$. Let $w_{ij} := \frac{1}{2}(\cot \alpha_{ij} + \cot \beta_{ij})$ be the cotangent weight of $\textbf{e}_{ij}$, where $\alpha_{ij}$ and $\beta_{ij}$ are the angles opposite to $\textbf{e}_{ij}$ in the triangle mesh. Finally, let $\textbf{R}_k$ be the matrix that approximates the rotation by which the one-ring neighborhood of $\textbf{v}_k$ deforms to the one-ring neighborhood of $\textbf{v}_k^\prime$. Then, this condition can be concretely formulated as follows.

```math
\underset{V^\prime, R}{\text{argmin}} \sum_{k = 1}^{|V|}\sum_{i,j \in \mathcal{N}(k)} w_{ij} \|\textbf{R}_k\textbf{e}_{ij} - \textbf{e}_{ij}^\prime\|^2
```

In addition to $V^\prime$, $R := \{\textbf{R}_0, \textbf{R}_1, \ldots, \textbf{R}_{|V|}\}$ must be also optimized.

Together, these conditions yield the following energy, where $\lambda$ is a parameter that controls the balance between the input mesh's style and the reference mesh's style in the output mesh.

```math
\begin{align*}
&\underset{V^\prime, R}{\text{argmin}} \sum_{k = 1}^{|V|}\sum_{i,j \in \mathcal{N}(k)} w_{ij} \|\textbf{R}_k\textbf{e}_{ij} - \textbf{e}_{ij}^\prime\|^2 + \lambda a_k \|\hat{\textbf{n}}_k^\prime - \textbf{t}_k\|^2 \\
\approx\:&\underset{V^\prime, R}{\text{argmin}} \sum_{k = 1}^{|V|}\sum_{i,j \in \mathcal{N}(k)} w_{ij} \|\textbf{R}_k\textbf{e}_{ij} - \textbf{e}_{ij}^\prime\|^2 + \lambda a_k \|\textbf{R}_k \hat{\textbf{n}}_k - \textbf{t}_k\|^2
\end{align*}
```

Because $\hat{\textbf{n}}_k^\prime$ makes optimizing this energy challenging, $\hat{\textbf{n}}_k^\prime$ is approximated as $\textbf{R}_k \hat{\textbf{n}}_k$. 

## Usage

Install the dependencies in `requirements.txt`.

```bash
$ pip install -r requirements.txt
```

Then, run the program with the following command. **The reference mesh, input mesh, and output mesh must be manifold triangle meshes in .obj format.**

```bash
$ main.py [-h] -r REFERENCE_PATH -i INPUT_PATH -o OUTPUT_PATH [-s STRENGTH]

options:
  -h, --help            show this help message and exit
  -r REFERENCE_PATH, --reference-path REFERENCE_PATH
                        reference .obj path
  -i INPUT_PATH, --input-path INPUT_PATH
                        input .obj path
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        output .obj path
  -s STRENGTH, --strength STRENGTH
                        strength of reference .obj style in output .obj
```

## References

Hsueh-Ti Derek Liu and Alec Jacobson. Normal-Driven Spherical Shape Analogies. _Computer Graphics
Forum_, 40(5):45–55, 2021.

Olga Sorkine and Marc Alexa. As-Rigid-As-Possible Surface Modeling. In _Proceedings of EUROGRAPH-
ICS/ACM SIGGRAPH Symposium on Geometry Processing_, pages 109–116, 2007.

<!--
TODO:
* better Voronoi area approximation
* mean curvature flow
* transfer texture coords
* lock a single vertex
* standardize $\lambda$?
-->
