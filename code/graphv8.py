import numpy as np
import tensorflow as tf
import time
import scipy.ndimage.filters

# build encoder
def encoder(opt,image): # [B,H,W,3]
	def conv2Layer(opt,feat,outDim):
		weight,bias = createVariable(opt,[3,3,int(feat.shape[-1]),outDim])
		conv = tf.nn.conv2d(feat,weight,strides=[1,2,2,1],padding="SAME")+bias
		batchnorm = batchNormalization(opt,conv,type="conv")
		relu = tf.nn.relu(batchnorm)
		return relu
	def linearLayer(opt,feat,outDim,final=False):
		weight,bias = createVariable(opt,[int(feat.shape[-1]),outDim])
		fc = tf.matmul(feat,weight)+bias
		batchnorm = batchNormalization(opt,fc,type="fc")
		relu = tf.nn.relu(batchnorm)
		return relu if not final else fc
	with tf.variable_scope("encoder"):
		feat = image
		with tf.variable_scope("conv1"): feat = conv2Layer(opt,feat,96) # 32x32
		with tf.variable_scope("conv2"): feat = conv2Layer(opt,feat,128) # 16x16
		with tf.variable_scope("conv3"): feat = conv2Layer(opt,feat,192) # 8x8
		with tf.variable_scope("conv4"): feat = conv2Layer(opt,feat,256) # 4x4
		feat = tf.reshape(feat,[opt.batchSize,-1])
		with tf.variable_scope("fc1"): feat = linearLayer(opt,feat,2048)
		with tf.variable_scope("fc2"): feat = linearLayer(opt,feat,1024)
		with tf.variable_scope("fc3"): feat = linearLayer(opt,feat,512,final=True)
		latent = feat
	return latent

# build decoder
def decoder(opt,latent):
	def linearLayer(opt,feat,outDim):
		weight,bias = createVariable(opt,[int(feat.shape[-1]),outDim])
		fc = tf.matmul(feat,weight)+bias
		batchnorm = batchNormalization(opt,fc,type="fc")
		relu = tf.nn.relu(batchnorm)
		return relu
	def deconv2Layer(opt,feat,outDim):
		H,W = int(feat.shape[1]),int(feat.shape[2])
		resize = tf.image.resize_images(feat,[H*2,W*2],method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
		weight,bias = createVariable(opt,[3,3,int(feat.shape[-1]),outDim],stddev=opt.std)
		conv = tf.nn.conv2d(resize,weight,strides=[1,1,1,1],padding="SAME")+bias
		batchnorm = batchNormalization(opt,conv,type="conv")
		relu = tf.nn.relu(batchnorm)
		return relu
	def pixelconv2Layer1(opt,feat,outDim):
		weight,bias = createVariable(opt,[1,1,int(feat.shape[-1]),outDim],gridInit=True)
		conv1 = tf.nn.conv2d(feat,weight[:,:,:,0:4],strides=[1,1,1,1],padding="SAME")+bias[:,:,:,0:4]
		conv2 = tf.nn.conv2d(feat,weight[:,:,:,4:8],strides=[1,1,1,1],padding="SAME")+bias[:,:,:,4:8]
		conv3 = tf.nn.conv2d(feat,weight[:,:,:,8:12],strides=[1,1,1,1],padding="SAME")+bias[:,:,:,8:12]
		conv4 = tf.nn.conv2d(feat,weight[:,:,:,12:16],strides=[1,1,1,1],padding="SAME")+bias[:,:,:,12:16]
		conv5 = tf.nn.conv2d(feat,weight[:,:,:,16:20],strides=[1,1,1,1],padding="SAME")+bias[:,:,:,16:20]
		conv6 = tf.nn.conv2d(feat,weight[:,:,:,20:24],strides=[1,1,1,1],padding="SAME")+bias[:,:,:,20:24]
		conv7 = tf.nn.conv2d(feat,weight[:,:,:,24:28],strides=[1,1,1,1],padding="SAME")+bias[:,:,:,24:28]
		conv8 = tf.nn.conv2d(feat,weight[:,:,:,28:32],strides=[1,1,1,1],padding="SAME")+bias[:,:,:,28:32]
		return conv1,conv2,conv3,conv4,conv5,conv6,conv7,conv8  
	with tf.variable_scope("decoder"):
		
		feato = tf.nn.relu(latent)
		
		with tf.variable_scope("fc1"): feat = linearLayer(opt,feato,1024)
		with tf.variable_scope("fc2"): feat = linearLayer(opt,feat,2048)
		with tf.variable_scope("fc3"): feat = linearLayer(opt,feat,4096)
		feat = tf.reshape(feat,[opt.batchSize,4,4,-1])		
		with tf.variable_scope("deconv1"): feat = deconv2Layer(opt,feat,192) # 8x8
		with tf.variable_scope("deconv2"): feat = deconv2Layer(opt,feat,128) # 16x16
		with tf.variable_scope("deconv3"): feat = deconv2Layer(opt,feat,96) # 32x32
		with tf.variable_scope("deconv4"): feat = deconv2Layer(opt,feat,64) # 64x64
		with tf.variable_scope("deconv5"): feat = deconv2Layer(opt,feat,48) # 128x128
		with tf.variable_scope("pixelconv1"): feat_a,tmp,tmp,tmp,tmp,tmp,tmp,tmp = pixelconv2Layer1(opt,feat,8*4) # 128x128

		with tf.variable_scope("fc1a"): feat = linearLayer(opt,feato,1024)
		with tf.variable_scope("fc2a"): feat = linearLayer(opt,feat,2048)
		with tf.variable_scope("fc3a"): feat = linearLayer(opt,feat,4096)
		feat = tf.reshape(feat,[opt.batchSize,4,4,-1])		
		with tf.variable_scope("deconv1a"): feat = deconv2Layer(opt,feat,192) # 8x8
		with tf.variable_scope("deconv2a"): feat = deconv2Layer(opt,feat,128) # 16x16
		with tf.variable_scope("deconv3a"): feat = deconv2Layer(opt,feat,96) # 32x32
		with tf.variable_scope("deconv4a"): feat = deconv2Layer(opt,feat,64) # 64x64
		with tf.variable_scope("deconv5a"): feat = deconv2Layer(opt,feat,48) # 128x128
		with tf.variable_scope("pixelconv2"): tmp,feat_b,tmp,tmp,tmp,tmp,tmp,tmp = pixelconv2Layer1(opt,feat,8*4) # 128x128

		with tf.variable_scope("fc1a1"): feat = linearLayer(opt,feato,1024)
		with tf.variable_scope("fc2a1"): feat = linearLayer(opt,feat,2048)
		with tf.variable_scope("fc3a1"): feat = linearLayer(opt,feat,4096)
		feat = tf.reshape(feat,[opt.batchSize,4,4,-1])		
		with tf.variable_scope("deconv1a1"): feat = deconv2Layer(opt,feat,192) # 8x8
		with tf.variable_scope("deconv2a1"): feat = deconv2Layer(opt,feat,128) # 16x16
		with tf.variable_scope("deconv3a1"): feat = deconv2Layer(opt,feat,96) # 32x32
		with tf.variable_scope("deconv4a1"): feat = deconv2Layer(opt,feat,64) # 64x64
		with tf.variable_scope("deconv5a1"): feat = deconv2Layer(opt,feat,48) # 128x128
		with tf.variable_scope("pixelconv3"): tmp,tmp,feat_c,tmp,tmp,tmp,tmp,tmp = pixelconv2Layer1(opt,feat,8*4) # 128x128

		with tf.variable_scope("fc1a11"): feat = linearLayer(opt,feato,1024)
		with tf.variable_scope("fc2a11"): feat = linearLayer(opt,feat,2048)
		with tf.variable_scope("fc3a11"): feat = linearLayer(opt,feat,4096)
		feat = tf.reshape(feat,[opt.batchSize,4,4,-1])		
		with tf.variable_scope("deconv1a11"): feat = deconv2Layer(opt,feat,192) # 8x8
		with tf.variable_scope("deconv2a11"): feat = deconv2Layer(opt,feat,128) # 16x16
		with tf.variable_scope("deconv3a11"): feat = deconv2Layer(opt,feat,96) # 32x32
		with tf.variable_scope("deconv4a11"): feat = deconv2Layer(opt,feat,64) # 64x64
		with tf.variable_scope("deconv5a11"): feat = deconv2Layer(opt,feat,48) # 128x128
		with tf.variable_scope("pixelconv4"): tmp,tmp,tmp,feat_d,tmp,tmp,tmp,tmp = pixelconv2Layer1(opt,feat,8*4) # 128x128

		with tf.variable_scope("fc1a111"): feat = linearLayer(opt,feato,1024)
		with tf.variable_scope("fc2a111"): feat = linearLayer(opt,feat,2048)
		with tf.variable_scope("fc3a111"): feat = linearLayer(opt,feat,4096)
		feat = tf.reshape(feat,[opt.batchSize,4,4,-1])		
		with tf.variable_scope("deconv1a111"): feat = deconv2Layer(opt,feat,192) # 8x8
		with tf.variable_scope("deconv2a111"): feat = deconv2Layer(opt,feat,128) # 16x16
		with tf.variable_scope("deconv3a111"): feat = deconv2Layer(opt,feat,96) # 32x32
		with tf.variable_scope("deconv4a111"): feat = deconv2Layer(opt,feat,64) # 64x64
		with tf.variable_scope("deconv5a111"): feat = deconv2Layer(opt,feat,48) # 128x128
		with tf.variable_scope("pixelconv41"): tmp,tmp,tmp,tmp,feat_e,tmp,tmp,tmp = pixelconv2Layer1(opt,feat,8*4) # 128x128

		with tf.variable_scope("fc1a2111"): feat = linearLayer(opt,feato,1024)
		with tf.variable_scope("fc2a2111"): feat = linearLayer(opt,feat,2048)
		with tf.variable_scope("fc3a2111"): feat = linearLayer(opt,feat,4096)
		feat = tf.reshape(feat,[opt.batchSize,4,4,-1])		
		with tf.variable_scope("deconv21a111"): feat = deconv2Layer(opt,feat,192) # 8x8
		with tf.variable_scope("deconv22a111"): feat = deconv2Layer(opt,feat,128) # 16x16
		with tf.variable_scope("deconv23a111"): feat = deconv2Layer(opt,feat,96) # 32x32
		with tf.variable_scope("deconv24a111"): feat = deconv2Layer(opt,feat,64) # 64x64
		with tf.variable_scope("deconv25a111"): feat = deconv2Layer(opt,feat,48) # 128x128
		with tf.variable_scope("pixelconv241"): tmp,tmp,tmp,tmp,tmp,feat_f,tmp,tmp = pixelconv2Layer1(opt,feat,8*4) # 128x128

		with tf.variable_scope("fc1a3111"): feat = linearLayer(opt,feato,1024)
		with tf.variable_scope("fc2a3111"): feat = linearLayer(opt,feat,2048)
		with tf.variable_scope("fc3a3111"): feat = linearLayer(opt,feat,4096)
		feat = tf.reshape(feat,[opt.batchSize,4,4,-1])		
		with tf.variable_scope("deconv31a111"): feat = deconv2Layer(opt,feat,192) # 8x8
		with tf.variable_scope("deconv32a111"): feat = deconv2Layer(opt,feat,128) # 16x16
		with tf.variable_scope("deconv33a111"): feat = deconv2Layer(opt,feat,96) # 32x32
		with tf.variable_scope("deconv34a111"): feat = deconv2Layer(opt,feat,64) # 64x64
		with tf.variable_scope("deconv35a111"): feat = deconv2Layer(opt,feat,48) # 128x128
		with tf.variable_scope("pixelconv341"): tmp,tmp,tmp,tmp,tmp,tmp,feat_g,tmp = pixelconv2Layer1(opt,feat,8*4) # 128x128

		with tf.variable_scope("fc1a4111"): feat = linearLayer(opt,feato,1024)
		with tf.variable_scope("fc2a4111"): feat = linearLayer(opt,feat,2048)
		with tf.variable_scope("fc3a4111"): feat = linearLayer(opt,feat,4096)
		feat = tf.reshape(feat,[opt.batchSize,4,4,-1])		
		with tf.variable_scope("deconv1a4111"): feat = deconv2Layer(opt,feat,192) # 8x8
		with tf.variable_scope("deconv2a4111"): feat = deconv2Layer(opt,feat,128) # 16x16
		with tf.variable_scope("deconv3a4111"): feat = deconv2Layer(opt,feat,96) # 32x32
		with tf.variable_scope("deconv4a4111"): feat = deconv2Layer(opt,feat,64) # 64x64
		with tf.variable_scope("deconv5a4111"): feat = deconv2Layer(opt,feat,48) # 128x128
		with tf.variable_scope("pixelconv441"): tmp,tmp,tmp,tmp,tmp,tmp,tmp,feat_h = pixelconv2Layer1(opt,feat,8*4) # 128x128


		feat=tf.concat([feat_a,feat_b,feat_c,feat_d,feat_e,feat_f,feat_g,feat_h], axis=-1)
		XYZ,maskLogit = tf.split(feat,[opt.outViewN*3,opt.outViewN],axis=-1) # [B,H,W,3V],[B,H,W,V]
	return XYZ, maskLogit, decoder # [B,H,W,3V],[B,H,W,V]

def createVariable1(opt,weightShape,biasShape=None,stddev=None,gridInit=False):
	if biasShape is None: biasShape = [weightShape[-1]]
	weight1 = tf.Variable(tf.random_normal(weightShape,stddev=opt.std),dtype=np.float32,name="weight1")
	if gridInit:
		X,Y = np.meshgrid(range(128),range(128),indexing="xy") # [H,W]
		X,Y = X.astype(np.float32),Y.astype(np.float32)
		initTile = np.concatenate([np.tile(X,[4,1,1]),
								   np.tile(Y,[4,1,1]),
								   np.ones([4,128,128],dtype=np.float32)*opt.renderDepth,
								   np.zeros([4,128,128],dtype=np.float32)],axis=0) # [4V,H,W]
		biasInit = np.expand_dims(np.transpose(initTile,axes=[1,2,0]),axis=0) # [1,H,W,4V]
	else:
		biasInit = tf.constant(0.0,shape=biasShape)
	bias1 = tf.Variable(biasInit,dtype=np.float32,name="bias1")
	return weight1,bias1
	
def createVariable(opt,weightShape,biasShape=None,stddev=None,gridInit=False):
	if biasShape is None: biasShape = [weightShape[-1]]
	weight = tf.Variable(tf.random_normal(weightShape,stddev=opt.std),dtype=np.float32,name="weight")
	if gridInit:
		X,Y = np.meshgrid(range(128),range(128),indexing="xy") # [H,W]
		X,Y = X.astype(np.float32),Y.astype(np.float32)
		initTile = np.concatenate([np.tile(X,[opt.outViewN,1,1]),
								   np.tile(Y,[opt.outViewN,1,1]),
								   np.ones([opt.outViewN,opt.outH,opt.outW],dtype=np.float32)*opt.renderDepth,
								   np.zeros([opt.outViewN,opt.outH,opt.outW],dtype=np.float32)],axis=0) # [4V,H,W]
		biasInit = np.expand_dims(np.transpose(initTile,axes=[1,2,0]),axis=0) # [1,H,W,4V]
	else:
		biasInit = tf.constant(0.0,shape=biasShape)
	bias = tf.Variable(biasInit,dtype=np.float32,name="bias")
	return weight,bias


# batch normalization wrapper function
def batchNormalization(opt,input,type):
	with tf.variable_scope("batchNorm"):
		globalMean = tf.get_variable("mean",shape=[input.shape[-1]],dtype=tf.float32,trainable=False,
											initializer=tf.constant_initializer(0.0))
		globalVar = tf.get_variable("var",shape=[input.shape[-1]],dtype=tf.float32,trainable=False,
										  initializer=tf.constant_initializer(1.0))
		if opt.training:
			if type=="conv": batchMean,batchVar = tf.nn.moments(input,axes=[0,1,2])
			elif type=="fc": batchMean,batchVar = tf.nn.moments(input,axes=[0])
			trainMean = tf.assign_sub(globalMean,(1-opt.BNdecay)*(globalMean-batchMean))
			trainVar = tf.assign_sub(globalVar,(1-opt.BNdecay)*(globalVar-batchVar))
			with tf.control_dependencies([trainMean,trainVar]):
				output = tf.nn.batch_normalization(input,batchMean,batchVar,None,None,opt.BNepsilon)
		else: output = tf.nn.batch_normalization(input,globalMean,globalVar,None,None,opt.BNepsilon)
	return output

# L1 loss
def l1_loss(input):
	return tf.reduce_sum(tf.abs(input))
# L1 loss (masked)
def masked_l1_loss(diff,mask):
	return l1_loss(tf.boolean_mask(diff,mask))
# sigmoid cross-entropy loss
def cross_entropy_loss(logit,label):
	return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,labels=label))
