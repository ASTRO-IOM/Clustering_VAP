# Clustering Van Allen Probe REPT Data
A data-driven, reproducible machine learning technique has been developed to successfully classify energetic electron pitch angle ditributions in the radition belts. In this technique we use a 2-stage dimensionality reduction by applying an autoencoder and principal component analysis to create a 3D representation of mutli-dimensional electron data to use in clustering algorithms. A meanshift algorithm is applied to predict the number of classifcations in the 3D data, before a k-means algorithm is applied to cluster the data into this number of classifications. The method used in this project was adapted from Bakrania et al (2020).

By using this method on Van Allen Probe REPT data, we have identified 8 different energy-dependent pitch angle distributions (PADs), 6 of which are scientifically significant, whereby they are expected to be as results of magentospheric phenomena (Killey et al., 2023). The shapes of these PADs are as expected from previous literature- butterfly (flux minimum at 90 degrees), pancake (flux peak at 90 deegrees) or flattop (flux plateau across a range of pitch angles) type distributions.

This method has the potential to be applied to other heliospheric missions to identify similar PAD features in more sparse multidimensional datasets.

***ML_all.py*** contains the code to run all the machine learning tools needed for this project, including the data pre-processing. The code has been built in such a way that it should automatically run on any dataset you give it, with adjustable paths, savenames and parameters.


# References
Bakrania, M., Rae, J., Walsh, A.~P., Verscharen, D., Smith, A.~W. (2020), Using dimensionality reduction and clustering techniques to classify space plasma regimes: electron magnetotail populations. AGU Fall Meeting Abstracts.

Killey, S., Rae, I. J., Chakraborty, S., Smith, A. W., Bentley, S. N., Bakrania, M. R.,  Wainwright, R., Watt. C. E. J., Sandhu, J. K. (2023), “Using machine learning to diagnose relativistic electron distributions in the Van Allen radiation belts”, RAS Techniques and Instruments, 16 August 2023, Volume 2, Issue 1, January 2023, Pages 548–561, https://doi.org/10.1093/rasti/rzad035
