import numpy as np
import scipy.misc,scipy.io
import time,os,sys
import threading
import util
import point_cloud_utils as pcu

print(util.toYellow("======================================================="))
print(util.toYellow("evaluate_dist.py (evaluate average distance of generated point cloud)"))
print(util.toYellow("======================================================="))

import tensorflow as tf
import data
import options

total1=[]

print(util.toMagenta("setting configurations..."))
opt = options.set(training=False)

with tf.device("/gpu:0"):
	VsPH = tf.placeholder(tf.float64,[None,3])
	VtPH = tf.placeholder(tf.float64,[None,3])
	_,minDist = util.projection(VsPH,VtPH)

# load data
print(util.toMagenta("loading dataset..."))
dataloader = data.Loader(opt,loadNovel=False,loadTest=True)
CADN = len(dataloader.CADs)

print(util.toYellow("======= EVALUATION START ======="))
timeStart = time.time()
# start session
tfConfig = tf.ConfigProto(allow_soft_placement=True)
tfConfig.gpu_options.allow_growth = True
with tf.Session(config=tfConfig) as sess:
	pred2GT_all = np.ones([CADN,opt.inputViewN])*np.inf
	GT2pred_all = np.ones([CADN,opt.inputViewN])*np.inf

	for m in range(CADN):
		CAD = dataloader.CADs[m]
		# load GT
		obj = scipy.io.loadmat("/content/data/{0}_testGT/{1}.mat".format(opt.category,CAD))
		Vgt = np.concatenate([obj["V"],obj["Vd"]],axis=0)
		VgtN = len(Vgt)
		# load prediction
		Vpred24 = scipy.io.loadmat("/content/drive/MyDrive/attprj1/3D-point-cloud-generation/results_{0}/orig-ft_it2000/{1}.mat".format(opt.group,CAD))["pointcloud"][:,0]
		assert(len(Vpred24)==opt.inputViewN)
		Vpred = Vpred24[0]
		VpredN = len(Vpred)
		# rotate CAD model to be in consistent coordinates
		Vpred[:,1],Vpred[:,2] = Vpred[:,2],-Vpred[:,1]
		# compute test error in both directions
		dist = pcu.chamfer_distance(np.float32(Vpred),np.float32(Vgt))
		print(m,dist)
		total1.append(dist)
#			pred2GT_all[m,a] = computeTestError(Vpred,Vgt,type="pred->GT")
#			GT2pred_all[m,a] = computeTestError(Vgt,Vpred,type="GT->pred")

print(util.toYellow("======= EVALUATION DONE ======="))
