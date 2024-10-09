# Shape Analogies

[INSERT IMAGE]

## Introduction

This is a tool that stylizes 3D objects based on the techniques introduced in the paper _Normal-Driven Spherical Shape Analogies_ by Liu and Jacobson. The idea behind their approach is to use analogies of the form "$A$ is to $A^\prime$ as $B$ is to $B^\prime$", where $A$, $A^\prime$, $B$, and $B^\prime$ are objects, to capture the surface normal "style" of $A^\prime$ and analogously apply that surface normal style to $B$ to form $B^\prime$.

Here, $A$ is a unit sphere. Assume $A$ is deformed into $A^\prime$. Then, the map $\phi : N_{A} \rightarrow N_{A^\prime}$ 



If _A_ is deformed into _A'_, then the relationship between the surface normals of _A_ and _A'_ can analogously be used to relate the surface normals of _B_ and _B'_. Then by extension, _B_ can be analogously deformed into _B'_



. _A'_ and _B'_ will share similar surface normals and hence have the same geometric style.

The algorithm proceeds in three steps.
1. 









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

ARAP encourages rigid transformations (translation and rotation only) in a least squares formulation.

## References

Hsueh-Ti Derek Liu and Alec Jacobson. Normal-driven spherical shape analogies. _Computer Graphics
Forum_, 40(5):45–55, 2021.

Olga Sorkine and Marc Alexa. As-rigid-as-possible surface modeling. In _Proceedings of EUROGRAPH-
ICS/ACM SIGGRAPH Symposium on Geometry Processing_, pages 109–116, 2007.

Normal-Driven Spherical Shape Analogies
As-Rigid-As-Possible Surface Modeling

Sparse matrices:
CSR -> Av or v^T A^T
CSC -> v^TA or A^Tv

--> another project idea could be ARAP deformations in real-time

https://mobile.rodolphe-vaillant.fr/entry/101/definition-laplacian-matrix-for-triangle-meshes

    # Good to add in README
    # Since Q (L in the other paper) is symmetric positive-definite, ve definite, the sparse
    # Cholesky factorization with fill-reducing reordering is an efficient choice
    # Step 1: Perform Cholesky factorization: A = L L^T
    # L = np.linalg.cholesky(A)
    # Step 2: Solve Ly = b using forward substitution
    # y = np.linalg.solve(L, b)
    # Step 3: Solve L^T x = y using back substitution
    # x = np.linalg.solve(L.T, y)
    # Factorization is more numerically stable and computationally efficient than solving system
    # via x = A^-1 b

TODO: Voronoi area, mean curvature flow, transfer texture coords

# maybe a constraint should be locking in a single vertex?
