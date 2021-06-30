# Point-cloud-generation-from-2D-image

Python implementation of our upcoming paper on point cloud generation from 2D image technique:
----------------------------------------------------------------------------------------------

arXiv Paper:

arXiv:2106.15325v1 [cs.CV]

Title:

"SE-MD: A Single-encoder multiple-decoder deep network for point cloud generation from 2D images"

Authors:

Abdul Mueed Hafiz, PhD, MTech, BTech
Rouf Ul Alam Bhat, PhD, MTech, BTech
Shabir Ahmad Parah, PhD, MPhil, MSc
M. Hassaballah, PhD

Abstract:

3D model generation from single 2D RGB images is a challenging and actively
researched computer vision task. Various techniques using conventional network
architectures have been proposed for the same. However, the body of research work is
limited and there are various issues like using inefficient 3D representation formats,
weak 3D model generation backbones, inability to generate dense point clouds,
dependence of post-processing for generation of dense point clouds, and dependence
on silhouttes in RGB images. In this paper, a novel 2D RGB image to point cloud
conversion technique is proposed, which improves the state of art in the field due to its
efficient, robust and simple model by using the concept of parallelization in network
architecture. It not only uses the efficient and rich 3D representation of point clouds,
but also uses a novel and robust point cloud generation backbone in order to address
the prevalent issues. This involves using a single-encoder multiple-decoder deep
network architecture wherein each decoder generates certain fixed viewpoints. This is
followed by fusing all the viewpoints to generate a dense point cloud. Various
experiments are conducted on the technique and its performance is compared with
those of other state of the art techniques and impressive gains in performance are
demonstrated. Code is available at https://github.com/mueedhafiz1982/
-------------------------------------------------------------------------------------------

This project is based on the work (See their readme):

https://github.com/chenhsuanlin/3D-point-cloud-generation.git
