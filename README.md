# Point-cloud-generation-from-2D-image

Python implementation of our upcoming paper on point cloud generation from 2D image technique:
----------------------------------------------------------------------------------------------

For citation quote the paper as:

Hafiz, A.M., Bhat, R.U.A., Parah, S.A. et al. SE-MD: a single-encoder multiple-decoder deep network for point cloud reconstruction from 2D images. 
Pattern Analysis and Applications, Springer (2023). https://doi.org/10.1007/s10044-023-01155-x

Abstract:

3D model reconstruction from single 2D RGB images is a challenging and actively researched computer vision task. Several techniques based on conventional network architectures have been proposed for the same. However, the body of research work is limited and there are some issues like using inefficient 3D representation formats, weak 3D model reconstruction backbones, inability to reconstruct dense point clouds, dependence of post-processing for reconstruction of dense point clouds and dependence on silhouettes in RGB images. In this paper, a new 2D RGB image to point cloud conversion technique is proposed, which improves the state-of-the-art in the field due to its efficient, robust and simple model by using the concept of parallelization in network architecture. It not only uses efficient and rich 3D representation of point clouds, but also uses a new robust point cloud reconstruction backbone to address the prevalent issues. This involves using a single-encoder multiple-decoder deep network architecture wherein each decoder reconstructs certain fixed viewpoints. This is followed by fusing all the viewpoints to reconstruct a dense point cloud. Various experiments are conducted to evaluate the proposed technique and to compare its performance with those of the state-of-the-arts and impressive gains in performance are demonstrated.
-------------------------------------------------------------------------------------------

This project is based on the work (See their readme):

https://github.com/chenhsuanlin/3D-point-cloud-generation.git
