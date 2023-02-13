# Clustering Van Allen Probe REPT Data
My PhD project entails applying a combination of machine learning techniques to relativitic electron flux measurements in order to classify pitch angle ditributions in the radition belts. In this technique we use a 2-stage dimensionality reduction by applying an autoencoder and principal component analysis to create a 3D representation of Van Allen Provbe REPT data to use in clustering algorithms. A meanshift algorithm is applied to predict the number of classifcations in the 3D data, before a k-means algorithm is applied to cluster the data into this number of classifications. The method used in this project was adapted from Bakrania et al (2020).

By using this method, we have identified 6 different energy dependent pitch angle distributions (PADs), 5 of which are scientifically significant, whereby they are expected to be as results of magentospheric phenomena (Killey et al., 2023 preprint). The shapes of theses PADs are as expected from previous literature- butterfly, pancake or flattops.

This method has the potential to be applied to other heliospheric missions to identify similar PAD features in more sparse multidimensional datasets.


***ML_all.py*** contains the code to run all the machine learning tools needed for this project, including the data pre-processing. The code has been built in such a way that it should automatically run on any dataset you give it, with adjustable paths, savenames and parameters.


# References
Bakrania, M., Rae, J., Walsh, A.~P., Verscharen, D., Smith, A.~W. (2020), Using dimensionality reduction and clustering techniques to classify space plasma regimes: electron magnetotail populations. AGU Fall Meeting Abstracts.

Killey, S., Rae, I. J., Chakraborty, S., Smith, A. W., Bentley, S. N., Wainwright, R. (submitted 2023), “Using Machine Learning to diagnose relativistic electron distributions in the Van Allen Radiation Belts”, RAS Techniques and Instruments, 31 January 2023, manuscript ID: RASTI -23-006, preprint: https://essopenarchive.org/doi/full/10.22541/essoar.167591055.58532301/v1.
