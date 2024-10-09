# Shape Analogies

[INSERT IMAGE]

## Introduction

This is a tool that stylizes 3D objects based on the techniques introduced in the paper _Normal-Driven Spherical Shape Analogies_ by Liu and Jacobson. The idea behind their approach is to use analogies of the form " $A$ _is to_ $A^\prime$ _as_ $B$ _is to_ $B^\prime$ " to capture the style of $A^\prime$ and analogously apply that style to $B$ to construct $B^\prime$. This is achieved by constructing surface normals for $B^\prime$ such that the mapping between the surface normals of $B$ and $B^\prime$ is analogous to the mapping between the surface normals of $A$ and $A^\prime$, where $A$ is a unit sphere.

Concretely, this is achieved by mapping the points on $B$ to points on $A$ using the Gauss map. This equivalently maps the surface normals on $B$ to surface normals on $A$ using the trivial map, which provides a way for the mapping between the surface normals of $A$ and $A^\prime$ to be transferred to the surface normals of $B$. The surface normals for $B^\prime$ can then be analogously computed.

Once the target normals are determined, the optimal $B^\prime$ is constructed by deforming $B$ such that the following energy is minimized. 

```math
\underset{V^\prime, R}{\text{min}} \sum_{k \in V}\sum_{i,j \in N_k} w_{ij} \|\textbf{R}_k\textbf{e}_{ij} - \textbf{e}_{ij}^\prime\|^2 + \lambda a_k \|\textbf{R}_k \hat{\textbf{n}}_k - \textbf{t}_k\|^2
```

Where...
* $\textbf{V}$ is a |V|-by-3 matrix containing the vertices of $B$
* $\textbf{V}^\prime$ is a |V|-by-3 matrix containing the vertices of $B^\prime$
* $\textbf{R}$ is the 3-by-3 rotation matrix on vertex $k$
* $N_k$ is the one-ring neighborhood of vertex $k$
* $\textbf{e}_{ij}$ is the edge from vertex $i$ to vertex $j$ in $B$
* $\textbf{e}_{ij}$ is the edge from vertex $i$ to vertex $j$ in $B^\prime$
* $w_{ij}$ is the cotangent weight of edge $(i, j)$ in $B$, calculated as $\frac{1}{2}(\cot \alpha_{ij} + \cot \beta_{ij})$
* $\lambda$ is a parameter that controls the strength of ARAP regularization
* $a_k$ is the Voronoi area of vertex $k$ in $B$
* $\textbf{n}_k$ is the vertex normal at vertex $k$ in $B$ calculated as the area-weighted average of face normals
* $\textbf{t}_k$ is the target vertex normal at vertex $k$
  
This energy uses as-rigid-as-possible (ARAP) regularization to penalize non-rigid transformations (i.e., rotations and translations) and preserve the features of $B$. 

## Usage

Install the dependencies in `requirements.txt`.

```bash
$ pip install -r requirements.txt
```

Then, run the program with the following command.

```bash
$ python3 main.py [-h] -a ANALOGY_PATH -i INPUT_PATH -o OUTPUT_PATH

options:
  -h, --help            show this help message and exit
  -a ANALOGY_PATH, --analogy-path ANALOGY_PATH
                        analogy .obj path (.obj with target style)
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
