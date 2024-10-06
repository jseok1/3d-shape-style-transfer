
ARAP encourages rigid transformations (translation and rotation only) in a least squares formulation. 

## Papers

Normal-Driven Spherical Shape Analogies
As-Rigid-As-Possible Surface Modeling

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


TODO: Voronoi area, mean curvature flow, cupy, transfer texture coords