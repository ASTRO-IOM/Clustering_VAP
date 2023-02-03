#!/usr/bin/env python
# coding: utf-8
# author: S. Killey

### script to perform data preprocessing, autoencoder, PCA, Mean shift and either Agglomerative or Kmeans clustering on normalised Flux data


#import packages
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from scipy import stats
from mpl_toolkits.mplot3d import axes3d
import matplotlib
import matplotlib.colors as colors
from collections import Counter
from scipy.optimize import curve_fit
from scipy import asarray as ar,exp
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.manifold import TSNE
import math
from sklearn import mixture
from sklearn import cluster
from sklearn.decomposition import PCA
from matplotlib.colors import LogNorm
from matplotlib import ticker, cm
import pandas as pd
from datetime import datetime
from pathlib import Path
import os.path
import matplotlib.patches as mpatches
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import MeanShift
from dateutil import parser
import glob
from tensorflow import keras
from tensorflow.keras import layers, Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
import cdflib
import cblind as cb
import time

# ----------- Adjustable Params -----------------
probe = ['b']

#split data
random_state = 4 # set the random seed
test_size = 0.6 # want 60% of the original data to be our test set
training_size = 0.5 #want 50% of the remaining 40% to be our training set and 50% to be validation set - 20:20:60 split

#autoencoder
input_dims = 204  #number of dimensions of the input dataset
encoded_dims = 102 #number of dimensions of the hidden, encoded layer
lrate = 0.0005 # learning rate of the optimizer
epoch = 200    # number of iterations the autoencoder will train for
batch_size = 256   # number of samples that are propagated through the network each iteration. Determined by 2^n, where n is positive. Should be as close to the # of dimensions as possible.
#validation = 1/8 # there fraction of points selected to be excluded from the training and are therefore tested on at the end of each epoch. To be used instead of a validation dataset

#PCA
PCA_dims = 3 # set to 3 for visualisation purposes

#MeanShift
bandwidth = 1.6  # size of the window function used in the mean shift calculation


#------------- Define Functions ----------------
#normalising
def normalise_90(data):
	'''function to normalise the data with respect to its 90deg PA value.
		input: flux data
		output: normalised flux data of same dimensions '''
	print('normalising')
	
	data[data==-1e31]=0
	data[data==np.nan]=0
	data[data==np.inf]=0
	data[data==-np.inf]=0
	#create duplicate of data
	copy_data = data

	data_shape = data.shape
	#create array of zeros
	data_norm = np.zeros(data_shape)

	#iterate over each observation and pitch angle
	for i in range(0, len(data_norm)):
		PA8 = data[i,8]
		#print('90deg PA:',PA8)
		for j in range(0,17):    
			
			PA = copy_data[i,j]    
			data_norm[i,j] = PA /PA8
			#print(PA, '/', PA8, '=', data_norm[i,j])
	
	#set any erroneous values to 0
	bad = np.isinf(data_norm)
	data_norm[bad==True] = 0

	bad = np.isnan(data_norm)
	data_norm[bad==True] = 0

	np.savez_compressed(os.path.join(combinedpath, data_norm_savename), data_norm = data_norm)   
	print(data_norm.max())
	return data_norm    

#normalising
def normalise_max(data):
	'''function to normalise the data with respect to its 90deg PA value.
		input: flux data
		output: normalised flux data of same dimensions '''
	print('normalising')
	
	data[data==-1e31]=0
	data[data==np.nan]=0
	data[data==np.inf]=0
	data[data==-np.inf]=0
	#create duplicate of data
	copy_data = data

	data_shape = data.shape
	#create array of zeros
	data_norm_max = np.zeros(data_shape)

	#iterate over each observation and pitch angle
	for i in range(0, len(data_norm_max)):
		data_max = data[i].max()
		#print('90deg PA:',PA8)
		for j in range(0,17):    
			
			PA = copy_data[i,j]    
			data_norm_max[i,j] = PA/ data_max
			#print(PA, '/', data_max, '=', data_norm_max[i,j])
	
	#set any erroneous values to 0
	bad = np.isinf(data_norm_max)
	data_norm_max[bad==True] = 0

	bad = np.isnan(data_norm_max)
	data_norm_max[bad==True] = 0

	np.savez_compressed(os.path.join(combinedpath, data_norm_savename), data_norm = data_norm_max)   
	print(data_norm_max.max()) # should be 1
	return data_norm_max   

          
#flattening data
def flatten_data(data_norm):
	'''function to flatten data if exists in 3D -> eg f(t,PA,E)
	input: normalised data in 3D
	output: normalised data in 2D '''
	print('flattening')
	data_flat = data_norm.reshape(len(data_norm),-1)

	#save data
	np.savez_compressed(os.path.join(combinedpath, data_flat_savename), data_flat = data_flat)

	return data_flat

def split_data(data_flat, test_size, training_size):
	''' function to split the flattened data into a training and testing set respectively 
	input: 
	data_flat: flattened data
	test_size: percentage of data to remain as a testing set (value between 0 and 1)
	output:
	train: training data set
	test: testing data set
	train_index: the original index of each observation in the training set
	test_index: the original index of each observation in the testing set '''
	# -------- Duplicate the Data ----------
	flat = data_flat
	print(flat.max())
	## ----- Split into test and training sets ----- 
	#need to keep track of the index that is being split
	index = np.arange(0, len(data_flat),1)
	#split the data into a training and testing set
	other, test, other_index, test_index = train_test_split(flat, index, test_size = test_size, random_state = 4)
	print(len(data_flat))
	train, val, train_index, val_index = train_test_split(other, other_index, test_size = training_size, random_state = 4)
	print(len(train),len(test),len(train_index),len(test_index))
	
	#save the test and training set
	np.savez_compressed(os.path.join(combinedpath,test_savename), index = test_index, test = test)
	np.savez_compressed(os.path.join(combinedpath,train_savename), index = train_index, train = train)
	np.savez_compressed(os.path.join(combinedpath,val_savename), index = val_index, val = val)
	return train, val, test, test_index, val_index, train_index


#autoencoder
def autoencode_images(train, val, test, input_dims, encoded_dims):
	'''function to create, train and test autoencoders on the data
	input:
	train: training data
	test: testing data
	input_dims: dimensions of the flat normalised data
	encoded_dims: number of dimensions of the enoded layer
    
	output:
	encoded_imgs: the applied encoding to the data
	decoded_imgs: the applied reconstruction of the data'''

	print('autoencoding') 
	print('training length:', len(train))
	print('validation length:', len(val))
	print('test length:', len(test))
	

	#----- Build the Autoencoder-----------
	    
	#1) set up an input placeholder -> has the same dimensions as the flattened data. 
	#This defines the number of neurons in the input of the encoder.
	input_img = Input(shape=(input_dims,))
    
	#2) set up the encoded representation (reduction) (hidden layer)
	#This creates the hidden layer with the # of neurons = encoding dim and connects it to input
	encoded = Dense(encoded_dims, activation='relu')(input_img)
    
	#3) set up the decoded reconstruction (output layer)
	#This is reconstructing the input from the encoded layer. The number of neurons/ dimensions is the same as the input.
	decoded = Dense(input_dims, activation='sigmoid')(encoded)
    
	#4) create the model that maps the input to its reconstruction.
	# This builds the whole autoencoder: from the input image to the latent layer to the reconstruction.
	autoencoder = Model(input_img, decoded)
    
	#5) create the model that maps the input to its encoded representation (reduction)
	# This builds the network between the input and latent layer
	encoder = Model(input_img, encoded)
    
	#6) set up an encoder input placeholder (the hidden layer)
	#This defines the number of neurons in the encoded layer of the autoencoder. Acts as an 'input' in the reconstruction.
	encoded_input = Input(shape=(encoded_dims,))
    
	#7) retrieve the last layer of the autoencoder model
	#This defines the reconstruction layer of the autoencoder.
	decoder_layer = autoencoder.layers[-1]
    
	#8) create model that reconstructs the encoded image
	# This builds the network between the encoded layer and the reconstruction.
	decoder = Model(encoded_input, decoder_layer(encoded_input))

	#9) Add any additional functions
	early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    min_delta=0,
    patience=5,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=False
    )
						
	#10) compile the autoencoder 
	#the metrics provide some statistical evaluation on the performance of the training on both the training set and validation set
	autoencoder.compile(optimizer=keras.optimizers.Adam(learning_rate = lrate), loss='mse')
    	
	autoencoder.summary() # the nuumber of params is the number of trainable weights
	#11) train the autoencoder
	#This will learn and identify patterns and relationships within the 'training' data set.
	#The epochs define the number of iterations, the batch_size is number of training samples to work with
	#Validation split sections of this amount of training data to test on the validate the results at the end of each epoch: 
	#The loss and model metrics are measured on this validation
	history = autoencoder.fit(train, train,
                epochs= epoch,
                batch_size= batch_size,
                shuffle=True,
                validation_data=(val,val), callbacks = [early_stopping]) 
                #last line helps model calculate val_loss and will stop when val_loss no longer increases/decreases											

	#12) apply the encoder to the test set AND then reconstuct
	#can apply to the full data set or the testing set
	#This is running the autoencoder
	encoded_imgs = encoder.predict(test) ## flat or test)
	decoded_imgs = decoder.predict(encoded_imgs) 
	

	#print(len(test), len(test_index),len(encoded_imgs))
    
	##13) save as a compressed file
	np.savez_compressed(os.path.join(autoencoderpath, autoencoder_savename), encoded_imgs = encoded_imgs, decoded_imgs = decoded_imgs, training_loss = history.history['loss'], validation_loss = history.history['val_loss'])
    
	#14) plot the loss curves
	colour, linestyle = cb.Colorplots().cblind(2)
	plt.figure(figsize=(9,6))
	plt.plot(history.history['loss'],linewidth=2.5, c = colour[0])
	plt.plot(history.history['val_loss'],linewidth=2.5, c = colour[1])
	plt.ylabel('Loss value',fontsize=20)
	plt.xlabel('Epoch',fontsize=20)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.legend(['Training loss', 'Validation loss'], loc='upper right',fontsize=20)
	plt.grid()
	plt.title('Van Allen Probe {}'.format(rbsp).upper())
    
	#save figure
	plt.savefig(os.path.join(losscurvepath, losscurve_figure_savename), format = 'png')
    
	plt.close()


	plt.figure(figsize=(9,6))
	plt.plot(history.history['loss'],linewidth=2.5, c = colour[0])
	plt.plot(history.history['val_loss'],linewidth=2.5, c = colour[1])
	plt.yscale('log')
	plt.ylabel('Loss value',fontsize=20)
	plt.xlabel('Epoch',fontsize=20)
	plt.xticks(fontsize=20)
	plt.yticks(fontsize=20)
	plt.legend(['Training loss', 'Validation loss'], loc='upper right',fontsize=20)
	plt.grid()
	plt.title('Van Allen Probe {}'.format(rbsp).upper())
	#save figure
	plt.savefig(os.path.join(losscurvepath, log_losscurve_figure_savename), format = 'png')
    
	plt.close()


	print('final loss:', history.history['val_loss'][-1])
	return encoded_imgs, decoded_imgs


#PCA
def pca_3d(encoded_imgs, PCA_dims):
	''' apply a 3D pca to the encoded images
	inputs:
	encoded_imgs: the encoded data from the AE
	PCA_dims: the number of PCA dimensions desired
    
	output:
	encoded_3d_pca: the applied 3D pca model to the encoded data'''
	#define evaluation method

	def evaluate_model(model, X):
		cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
		cross_scores = cross_val_score(model, X,)# scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
		return cross_scores
  
	print('PCA')  
	#1) create the 3D pca model
	model_encoder = PCA(n_components=PCA_dims) #create the model of (p1,p2,p3)

	#2) apply to data and transform into PCA_dims 
	encoded_3d_pca = model_encoder.fit_transform(encoded_imgs) 

	#3) Evaluate
	params = model_encoder.get_params()
	#print(params)
	precision = model_encoder.get_precision()
	print('precision:',precision)
	score = model_encoder.score(encoded_imgs)
	print('score length', score.shape)
	print('score:',score)
	score_sample = model_encoder.score_samples(encoded_imgs)
	print('score sample length', score_sample.shape)
	print('score samples:',score_sample)

	var_ratio = model_encoder.explained_variance_ratio_
	print(var_ratio[0:PCA_dims])
	var = model_encoder.explained_variance_
	print(var)
	noise = model_encoder.noise_variance_
	print(noise)

	cross_scores = evaluate_model(model_encoder, encoded_imgs)
	cross_mean, cross_std = np.mean(cross_scores), np.std(cross_scores)
	print('log likelihood mean: %.3f , std: %.3f' %(cross_mean, cross_std))

	#4) plot
	fig = plt.figure(figsize=(15,15))
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(encoded_3d_pca[:,0],encoded_3d_pca[:,1],encoded_3d_pca[:,2], c = 'k')
	ax.set_xlabel('PCA_0',fontsize = 26, labelpad = 20)
	ax.set_ylabel('PCA_1',fontsize = 26, labelpad = 20)
	ax.set_zlabel('PCA_2',fontsize = 26, labelpad = 20)
	ax.tick_params(labelsize= 26)  
	plt.title('Van Allen Probe {}'.format(rbsp).upper,fontsize = 20)
	plt.grid()    
	ax.view_init(20, 168)
    
	plt.savefig(os.path.join(PCA_3d_figurepath ,PCA_figure_savename), format = 'png')
    
	plt.close()

	assert len(encoded_3d_pca)==len(encoded_imgs)
    
	#save as a compressed file	
	np.savez_compressed(os.path.join(PCA_3d_path, PCA_savename), encoded_3d_pca = encoded_3d_pca, score = score, score_sample = score_sample, params = params, precision = precision, variance = var, variance_ratio = var, noise = noise, cross_scores = cross_scores, cross_mean = cross_mean, cross_std = cross_std)
	return encoded_3d_pca

 
#MeanShift
def Mean_Shift(encoded_3d_pca, bandwidth):
	'''function to perform the meanshift of the data.This will predict the number of clusters for agglomerative data.
	input:
	enocoded_3d_pca: the 3D flux data from the PCA

	output:
	ms_clustering: clustered data
	nclusters: number of clusters '''

	print('Mean Shift')
	st = time.process_time()
	# precict how many clusters
	ms_clustering = MeanShift(bandwidth=bandwidth,bin_seeding=True, n_jobs=2).fit(encoded_3d_pca) 
	c = Counter(ms_clustering.labels_)
	clusters = sorted(c)
	nclusters = len(c)
	
	print('number of clusters', nclusters)

	#sil_score = metrics.silhouette_score(encoded_3d_pca, ms_clustering.labels_, metric = 'euclidean') 
	CH_score = metrics.calinski_harabasz_score(encoded_3d_pca, ms_clustering.labels_)
	DB_index = metrics.davies_bouldin_score(encoded_3d_pca, ms_clustering.labels_)
	#print('sil_score:', sil_score)
	print('CH_score:', CH_score)
	print('DB_index:', DB_index)


	#save
	#np.savez_compressed(os.path.join(Meanshift_path, MeanShift_savename), ms_clustering = ms_clustering.labels_, nclusters = nclusters, CH_score = CH_score, DB_index = DB_index)#sil_score = sil_score
	et = time.process_time()
	res = et - st
	print('CPU Execution time:', res, 'seconds')	
	return ms_clustering, nclusters

#Agglomerativeclustering
def Agglomerative(encoded_3d_pca, nclusters):
	'''function to perform the agglomerative of the data using the predicted the number of clusters from MeanShift.
	input:
	
	enocoded_3d_pca: the 3D flux data from the PCA
	nclusters: the number of clusters determined from Mean Shift

	output:
	agglomerative: the clustering data by agglomerative clustering'''

	print('Agglomerative')
	#1)cluster using agglomerative algorithm
	ac_clustering = AgglomerativeClustering(n_clusters=nclusters,linkage='ward').fit(encoded_3d_pca)

	ac_clustering = ac_clustering.labels_
	c = Counter(ac_clustering)
	print(c)

	#2) save
	np.savez_compressed(os.path.join(agg_path, agg_savename),ac_clustering)

	#3) Colour the PCA plot
	#3.1) define colour palette and build figure
	colour, linestyle = cb.Colorplots().rainbow(nclusters)
	patch =[]
	fig = plt.figure(figsize=(15,15))
	ax = fig.add_subplot(111, projection='3d')
	ax.set_xlabel('PCA 0')
	ax.set_ylabel('PCA 1')
	ax.set_zlabel('PCA 2')

	#3.2) sort data into clusters
	for n in range(0, nclusters):
		vars()['encoded_3d_pca_{}'.format(n)] = []

		for j in range(len(ac_clustering)):   
			if ac_clustering[j] == n:
                		vars()['encoded_3d_pca_{}'.format(n)].append(encoded_3d_pca[j])

		ax.scatter(np.array(vars()['encoded_3d_pca_{}'.format(n)])[:,0],np.array(vars()['encoded_3d_pca_{}'.format(n)])[:,1],np.array(vars()['encoded_3d_pca_{}'.format(n)])[:,2],label='Cluster {}'.format(n))     
		patch.append(mpatches.Patch(color=colour[n])) 

	#3.3) add extras and save
	plt.legend()#handles = patch)
	plt.title('VAP {}'.format(rbsp).upper())
	plt.savefig(os.path.join(agg_figure_path, agg_figure_savename), format = 'png')
	plt.close()

	return ac_clustering

def Kmeans_cluster(encoded_3d_pca,nclusters):
    ''' a function to apply a k means clustering algorithm to the 3D pca modelled data:
    enocoded_3d_pca: the 3D flux data from the PCA
nclusters: the number of clusters determined from Mean Shift'''

    print('kmeans')
    st = time.process_time()


    kmeans_model = KMeans(n_clusters= nclusters, random_state=0).fit(encoded_3d_pca) # cluster the data into n unknown groups depending on their p-unit vector quantities -> aka relative positions. 
    # clusters given by the estimation 

    kmeans = kmeans_model.labels_
    inertia = kmeans_model.inertia_
    centre_coords= kmeans_model.cluster_centers_
    #sil_score = metrics.silhouette_score(encoded_3d_pca, kmeans, metric = 'euclidean') 
    CH_score = metrics.calinski_harabasz_score(encoded_3d_pca, kmeans)
    DB_index = metrics.davies_bouldin_score(encoded_3d_pca, kmeans)
    #print('sil_score:', sil_score)
    print('CH_score:', CH_score)
    print('DB_index:', DB_index)

    #save the classification
    np.savez_compressed(os.path.join(kmeanspath, kmeans_savename),kmeans = kmeans, inertia = inertia, cluster_centres = centre_coords, DB_index = DB_index, CH_score = CH_score)

    et = time.process_time()
    res = et - st
    print('CPU Execution time:', res, 'seconds')
    c = Counter(kmeans)
    clusters = sorted(c)
    print(c)
    #plot 
    #set the colour palette to be rainbow (cblind version)
    palette, linestyle = cb.Colorplots().rainbow(nclusters)
    patch =[]
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('PCA 0',fontsize = 26, labelpad = 20)
    ax.set_ylabel('PCA 1',fontsize = 26, labelpad = 20)
    ax.set_zlabel('PCA 2',fontsize = 26, labelpad = 20)
    ax.xaxis.set_tick_params(labelsize=26)
    ax.yaxis.set_tick_params(labelsize=26)
    ax.zaxis.set_tick_params(labelsize=26)
    ax.tick_params(axis='both', which='major', labelsize=26)
    ax.tick_params(axis='both', which='minor', labelsize=26)
    ax.view_init(20, 168)
    for n in clusters:

        vars()['encoded_3d_pca_{}'.format(n)] = []

        for j in range(len(kmeans)):   
                if kmeans[j] == n:
                    vars()['encoded_3d_pca_{}'.format(n)].append(encoded_3d_pca[j]) #if in kmeans group =0, assign the encoded_img_3d value to the encoded_imgs_3d_0 array
               

        #x = np.array((vars()['encoded_3d_pca_{}'.format(n)])[:,0])#[0]
        #y = np.array((vars()['encoded_3d_pca_{}'.format(n)])[:,1])
        #z = np.array((vars()['encoded_3d_pca_{}'.format(n)])[:,2])
        ax.scatter(np.array(vars()['encoded_3d_pca_{}'.format(n)])[:,0],np.array(vars()['encoded_3d_pca_{}'.format(n)])[:,1],np.array(vars()['encoded_3d_pca_{}'.format(n)])[:,2], label='Cluster {}'.format(n))
        patch.append(mpatches.Patch(color=palette[n],label = '{}'.format(n))) 

    legend = ax.legend(handles= patch, loc = 'upper right', title = 'Cluster',fontsize= 26)
    plt.setp(legend.get_title(),fontsize= 26)


    #plt.title('Van Allen Probe {}'.format(rbsp).upper(),fontsize = 18)
    plt.tight_layout()
    ax.xaxis._axinfo['label']['space_factor'] = 5.0
    #ax.yaxis._axinfo['label']['space_factor'] = 5.0
    #ax.zaxis._axinfo['label']['space_factor'] = 5.0
    plt.savefig(os.path.join(kmeans_figure_path, kmeans_figure_savename), format = 'png')
    plt.close()
    return kmeans


# ---------- RUN --------------

for rbsp in probe:

	#------------ Directories -----------------------
	#data
	motherpath = '/home/Documents'
	probedatapath = os.path.join(motherpath,'Data/RBSP_ECT/REPT/RBSP_{}'.format(rbsp))
	combinedpath = os.path.join(probedatapath,'Combined_data') 
	combinedyearpath = os.path.join(combinedpath, 'All_years') 
	autoencoderpath = os.path.join(probedatapath,'Autoencoder') 
	PCA_3d_path = os.path.join(probedatapath,'3D_PCA')
	Meanshift_path = os.path.join(probedatapath,'MeanShift')
	agg_path = os.path.join(probedatapath,'Agglomerative')
	kmeanspath = os.path.join(probedatapath,'K_means')

	#figures
	probefigurepath = os.path.join(motherpath,'Figures', 'RBSP_ECT/REPT/RBSP_{}'.format(rbsp))
	losscurvepath = os.path.join(probefigurepath,'Autoencoder_loss_curves')
	PCA_3d_figurepath = os.path.join(probefigurepath,'3D_PCA')
	agg_figure_path = os.path.join(probefigurepath,'Agglomerative')
	kmeans_figure_path = os.path.join(probefigurepath,'K_means')

	#-------------- File Savenames ---------------
	
	data_norm_savename = 'All_years/norm_max_data_rbsp_{}_All'.format(rbsp)
	data_flat_savename = 'All_years/norm_max_data_flat_rbsp_{}_All'.format(rbsp)
	test_savename = 'All_years/test_data_rbsp_{}_All'.format(rbsp)
	train_savename = 'All_years/train_data_rbsp_{}_All'.format(rbsp)
	val_savename = 'All_years/val_data_rbsp_{}_All'.format(rbsp)

	autoencoder_savename = 'autoencoder_outputs_rbsp_{}_All'.format(rbsp)
	PCA_savename = '3D_PCA_{}_All'.format(rbsp)
	MeanShift_savename = 'MeanShift_rbsp_{}_All'.format(rbsp)
	kmeans_savename ='kmeans_rbsp_{}_All'.format(rbsp)

	losscurve_figure_savename =  'Loss_curves_rbsp_{}_All.png'.format(rbsp)
	log_losscurve_figure_savename =  'Loss_curves_rbsp_{}_log_All.png'.format(rbsp)
	PCA_figure_savename = '3D_PCA_rbsp_{}_All.png'.format(rbsp)
	kmeans_figure_savename = 'kmeans_rbsp_{}_All.png'.format(rbsp)



	#-------------- DATA ---------------------------
	dataall = np.load(os.path.join(combinedpath, 'All_years/RBSP_{}_All.npz'.format(rbsp)),allow_pickle=True)['data_all'] 

	##for rerunning specific functions
	#epochall = np.load(os.path.join(combinedpath, 'All_years/RBSP_{}_All.npz'.format(rbsp)),allow_pickle=True)['epoch_all'] 
	#data_norm = np.load(os.path.join(combinedpath, 'All_years/norm_max_data_rbsp_{}_All.npz'.format(rbsp)),allow_pickle=True)['data_norm']
	#data_flat = np.load(os.path.join(combinedpath, 'All_years/norm_max_data_flat_rbsp_{}_All.npz'.format(rbsp)),allow_pickle=True)['data_flat']
	#train = np.load(os.path.join(combinedpath, 'All_years/train_data_rbsp_{}_All.npz'.format(rbsp)),allow_pickle=True)['train']
	#test = np.load(os.path.join(combinedpath, 'All_years/test_data_rbsp_{}_All.npz'.format(rbsp)),allow_pickle=True)['test']
	#val = np.load(os.path.join(combinedpath, 'All_years/val_data_rbsp_{}_All.npz'.format(rbsp)),allow_pickle=True)['val']
	#encoded_imgs = np.load(os.path.join(autoencoderpath, 'autoencoder_outputs_rbsp_{}_All.npz'.format(rbsp)), allow_pickle = True)['encoded_imgs']
	#encoded_3d_pca = np.load(os.path.join(PCA_3d_path, '3D_PCA_{}_All.npz'.format(rbsp)), allow_pickle=True)['encoded_3d_pca']
	#nclusters = np.load(os.path.join(Meanshift_path, 'MeanShift_rbsp_{}_All.npz'.format(rbsp)), allow_pickle=True)['nclusters']

	#data_norm = normalise_90(dataall)  ##either normalise at 90 for flux magnitude clustering 
	data_norm = normalise_max(dataall)  ## or normlaise at maximum for distribution shape clustering
	data_flat = flatten_data(data_norm)
	train, val, test, train_index, val_index, test_index = split_data(data_flat, test_size, training_size)
	encoded_imgs, decoded_imgs = autoencode_images(train,val, test, input_dims, encoded_dims)
	encoded_3d_pca = pca_3d(encoded_imgs, PCA_dims)
	ms_clustering, nclusters = Mean_Shift(encoded_3d_pca, bandwidth)
	#ag_clustering = Agglomerative(encoded_3d_pca, nclusters) #due to computational expense, it is not recommended to use agglomerative clustering
	kmeans = Kmeans_cluster(encoded_3d_pca, nclusters)

		
