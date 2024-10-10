# 3D Object Style Transfer

[INSERT IMAGE]

## Introduction

This is a tool that stylizes 3D shapes based on the techniques introduced in the paper _Normal-Driven Spherical Shape Analogies_ by Liu and Jacobson. The idea behind their approach is to use analogies of the form "$A$ _is to_ $A^\prime$ _as_ $B$ _is to_ $B^\prime$" to stylize an input shape based on a reference shape. In this framework, $A$ is a unit sphere, $A^\prime$ is the reference shape, $B$ is the input shape, and $B^\prime$ is the output shape. The fundamental insight to this approach is that surface normals capture the style of a shape. If two shapes have the same surface normals "up to analogy", then those two shapes have the same style. Let $N_A$, $N_{A^\prime}$, $N_B$, and $N_{B^\prime}$ be the surface normals of $A$, $A^\prime$, $B$, and $B^\prime$, respectively. The aim of this approach is to construct $B^\prime$ such that $N_{B^\prime}$ is the same as $N_{A^\prime}$ up to analogy.

Let $T$ be the target surface normals of $B^\prime$. That is, $T$ is the optimal $N_{B^\prime}$, since it may not be possible to construct $B^\prime$ such that $N_{B^\prime}$ is the same as $N_{A^\prime}$ up to analogy in practice. Also, assume there is a mapping $\tilde{N}_{A^\prime} : A \rightarrow N_{A^\prime}$.

Then, $T$ can be constructed as follows:
1. Map each point on $B$ to a point on $A$ using the Gauss map.
2. Map those points on $A$ to surface normals on $A^\prime$ using $\tilde{N}_{A^\prime}$.
3. Assign those surface normals on $A^\prime$ as the target surface normals on $B^\prime$.

<!-- The choice of $\tilde{N}_{A^\prime}$ is important. -->

Then, $B^\prime$ can be constructed by deforming $B$ in a way such that $N_{B^\prime}$ approximates $T$ and $B^\prime$ preserves the surface details of $B$. 

<!-- vertex and face nromals -->
<!-- Concretely, assume $B$ and $B^\prime$ are manifold triangle meshes with vertices $V$ and $V^\prime$. Then, we minimize $\sum_{k \in V}\|\hat{\textbf{n}}_k^\prime - \textbf{t}_k\|^2$

These constraints are captured by the following energy:

```math
\underset{\textbf{V}^\prime, R}{\text{min}} \sum_{k \in V}\sum_{i,j \in N_k} w_{ij} \|\textbf{R}_k\textbf{e}_{ij} - \textbf{e}_{ij}^\prime\|^2 + \lambda a_k \|\textbf{R}_k \hat{\textbf{n}}_k - \textbf{t}_k\|^2
```

Where...
* $\textbf{V}$ is the set of vertices of $B$
* $\textbf{V}^\prime$ is the set of vertices of $B^\prime$
* $\textbf{R}$ is a 3-by-3 rotation matrix on vertex $k$
* $N_k$ is the one-ring neighborhood of vertices of vertex $k$
* $\textbf{e}_{ij}$ is the edge from vertex $i$ to vertex $j$ in $B$
* $\textbf{e}_{ij}$ is the edge from vertex $i$ to vertex $j$ in $B^\prime$
* $w_{ij}$ is the cotangent weight of edge $(i, j)$ in $B$, calculated as $\frac{1}{2}(\cot \alpha_{ij} + \cot \beta_{ij})$
* $\lambda$ is a parameter that controls the strength of ARAP regularization
* $a_k$ is the Voronoi area of vertex $k$ in $B$
* $\textbf{n}_k$ is the vertex normal at vertex $k$ in $B$ calculated as the area-weighted average of face normals
* $\textbf{t}_k$ is the target vertex normal at vertex $k$
  
This energy uses as-rigid-as-possible (ARAP) regularization to penalize non-rigid transformations (i.e., rotations and translations), hence preserving the details of $B$.  -->

## Usage

Install the dependencies in `requirements.txt`.

```bash
$ pip install -r requirements.txt
```

Then, run the program with the following command. **The reference object, input object, and output object must be manifold triangle meshes in .obj format.**

```bash
$ main.py [-h] -r REFERENCE_PATH -i INPUT_PATH -o OUTPUT_PATH

options:
  -h, --help            show this help message and exit
  -r REFERENCE_PATH, --reference-path REFERENCE_PATH
                        reference .obj path
  -i INPUT_PATH, --input-path INPUT_PATH
                        input .obj path
  -o OUTPUT_PATH, --output-path OUTPUT_PATH
                        output .obj path
```

## References

Hsueh-Ti Derek Liu and Alec Jacobson. Normal-driven spherical shape analogies. _Computer Graphics
Forum_, 40(5):45–55, 2021.

Olga Sorkine and Marc Alexa. As-Rigid-As-Possible Surface Modeling. In _Proceedings of EUROGRAPH-
ICS/ACM SIGGRAPH Symposium on Geometry Processing_, pages 109–116, 2007.

## Backlog

TODO: Voronoi area, mean curvature flow, transfer texture coords
maybe a constraint should be locking in a single vertex?
