# Interactive Multi-label Segmentation

This repository is based on an article written by Claudia Nieuwenhuis and Daniel Cremers (https://vision.in.tum.de/_media/spezial/bib/nieuwenhuis-cremers-pami12.pdf). This article published in 2012 proposes a method for multi-label segmentation thanks to a statistical framework that takes into account the spatial variation of color distribution. 

My implementation of the method is available in the "Interactive image segmentation" file. In this one you can find a \textit{Image\_seg\_main.py} file containing the python code and a notebook allowing to test the algorithm on an input image with given scribbles. 

## Interactive scribbling :
To draw scribbles on the image, you have to run \textit{Get\_scribbles.py}. Before that, you have to choose the input image and the number of regions. The script will open a window with the input image. By clicking in two places, we can then draw a line allowing to label the pixels of this line in the first region. We can repeat this maneuver as many times as we want. Then to go to the next region we close the current window, and we draw new lines for the second region.
The positions of the clicks for each region are stored in the dictionary \textit{Scribbles}. Then the positions of all pixels on lines are obtained with the function \textit{get\_entire\_scribbles} that uses the Bresenham's Line Algorithm (https://gist.github.com/Siyeong-Lee/fdacece959c6973603dbda955e45637b).

## Compute f and g :
Now that we have labeled some pixels of the image we will be able to compute $g(x)$ and $f_i(x)$ for each region $i$. 
This is done with the functions \textit{compute\_f} and \textit{fct\_g}. The values for $f$ are stored in a Numpy array of dimension 3 $(n \times H \times W)$ with n the number of regions and $(H \times W)$ the image size. First, we have to compute the values of the space kernel width for each position and each region. Then we have to compute the values of the Gaussian kernels for space and color. My implementation is not very efficient, especially when the size of scribbles increases. It would probably be possible to speed up the algorithm by doing the calculations in parallel. Concerning g, we have to compute the image gradient. This is done by forward differences as detailed in https://vision.in.tum.de/_media/spezial/bib/nieuwenhuis-et-al-ijcv13.pdf. 

## Primal Dual optimization :
The main part of the code is the \textit{primal\_dual\_optimization} which takes as argument the values of $g(x)$ and $f_i(x)$ and returns the values of $\theta_i$ after convergence as well as the values of the energies and the primal-dual gap at each iteration. For the initialization we start with values of $\theta_i$ randomly taken between $0$ and $1$. For step 1. (projected gradient ascent in the dual variables) we need to compute the gradient of $\theta_i$. This is done with the same method as for calculating the gradient of the image.
Concerning the gradient descent in the primal variables we obtain the divergence of $\xi_i$ by computing backward differences as explained in https://vision.in.tum.de/_media/spezial/bib/nieuwenhuis-et-al-ijcv13.pdf.

## Projection on $\kappa_g$ and $\tilde{\mathcal{B}}$} : 
The projection on $\kappa_g$ is just done by identifying positions where $|\xi_i(x)|$ is greater than $\frac{g(x)}{2}$ and replace $\xi_i(x)$ by $\frac{g(x)}{2}\frac{\xi_i}{|\xi_i|}$ so that the norm will be exactly equal to $\frac{g(x)}{2}$.

Concerning $\tilde{\mathcal{B}}$, I implemented the algorithm proposed in the article https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf to do the projection onto a simplex in $\mathbb{R}^n$.