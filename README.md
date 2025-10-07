# Clustering Van Allen Radiation Belt Particles 
A data-driven, reproducible machine learning technique has been developed to successfully classify energetic electron pitch angle distributions in the radiation belts. In this technique, a two-stage dimensionality reduction is employed by applying an autoencoder and principal component analysis to create a 3D representation of multidimensional electron data for use in clustering algorithms. A meanshift algorithm is applied to predict the number of classifications in the 3D data, before a k-means algorithm is applied to cluster the data into this number of classifications. The method used in this project was adapted from Bakrania et al (2020).

By using this method on Van Allen Probe REPT electron data (electrons with energies 1MeV+), we have identified eight different energy-dependent pitch angle distributions (PADs), six of which are scientifically significant, each a common PAD shape - either butterfly (flux minimum at 90 degrees), pancake (flux peak at 90 deegrees) or flattop (flux plateau across a range of pitch angles) type distributions (Killey et al., 2023). 

Each PAD type can be associated with a different magnetospheric phenomenon based upon their spatio-temporal evolution during geomagnetic storms, for example, wave-particle interactions, radial diffusion, magnetopause shadowing and particle injections (Killey et al., 2025).

The algorithm has been tested for robustness on a subset of Van Allen Probe magEIS electron data (electrons with energies 200keV - 1MeV) - resulting in the identification of three different energy-dependent PADs, which are common with their relativistic counterparts (Killey, 2025), but acknowledge that this algorithm could easily be applied to other particle species - such as protons. This method also has the potential to be applied to other heliospheric missions, such as JUNO, to identify similar PAD features in more sparse datasets and alternative plasma environments.

***ML_all.py*** contains the code to run all the machine learning tools needed for this project, including the data pre-processing. The code has been built in such a way that it should automatically run on any pitch angle resolved particle flux dataset, with adjustable paths, save names and parameters.

# References
Bakrania, M., Rae, J., Walsh, A.~P., Verscharen, D., Smith, A.~W. (2020), Using dimensionality reduction and clustering techniques to classify space plasma regimes: electron magnetotail populations. AGU Fall Meeting Abstracts.

Killey, S., Rae, I. J., Chakraborty, S., Smith, A. W., Bentley, S. N., Bakrania, M. R.,  Wainwright, R., Watt. C. E. J., Sandhu, J. K. (2023), “Using machine learning to diagnose relativistic electron distributions in the Van Allen radiation belts”, RAS Techniques and Instruments, 16 August 2023, Volume 2, Issue 1, January 2023, Pages 548–561, https://doi.org/10.1093/rasti/rzad035

Killey, S., Rae, I. J., Smith, A. W., Bentley, S. N., Watt, C. E. J., Chakraborty, S., et al. (2025). Identifying typical relativistic electron pitch angle distributions: Evolution during geomagnetic storms. Geophysical Research Letters, 52, e2024GL112900. https://doi.org/10.1029/2024GL112900

Killey, S (2025), "Using Machine Learning to Understand Electron Pitch Angle Evolution of the Van Allen Radiation Belts", Doctoral Thesis, Northumbria University, 22 May 2025, https://researchportal.northumbria.ac.uk/en/studentTheses/using-machine-learning-to-understand-electron-pitch-angle-evoluti
