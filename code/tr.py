print(util.toYellow("======= TRAINING START ======="))
timeStart = time.time()
# start session
tfConfig = tf.ConfigProto(allow_soft_placement=True)
tfConfig.gpu_options.allow_growth = True
with tf.Session(config=tfConfig) as sess:
	sess.run(tf.global_variables_initializer())
	if opt.fromIt!=0:
		util.restoreModelFromIt(opt,sess,saver,opt.fromIt)
		print(util.toMagenta("resuming from iteration {0}...".format(opt.fromIt)))
	else:
		summaryWriter.add_graph(sess.graph)
	print(util.toMagenta("start training..."))

	chunkResumeN,chunkMaxN = opt.fromIt//opt.itPerChunk,opt.toIt//opt.itPerChunk
	# training loop
	for c in range(chunkResumeN,chunkMaxN):
		dataloader.shipChunk()
		dataloader.thread = threading.Thread(target=dataloader.loadChunk,args=[opt])
		dataloader.thread.start()
		for i in range(c*opt.itPerChunk,(c+1)*opt.itPerChunk):
			lr = opt.lr*opt.lrDecay**(i//opt.lrStep)
			# make training batch
			batch = data.makeBatchFixed(opt,dataloader,PH)
			batch[lr_PH] = lr
			# run one step
			runList = [optim,loss,loss_XYZ,loss_mask,maskLogit]
			_,l,lx,lm,ml = sess.run(runList,feed_dict=batch)
			if (i+1)%50==0:
				print("it. {0}/{1}, lr={2}, loss={4} ({5},{6}), time={3}"
					.format(util.toCyan("{0}".format(i+1)),
							opt.toIt,
							util.toYellow("{0:.0e}".format(lr)),
							util.toGreen("{0:.2f}".format(time.time()-timeStart)),
							util.toRed("{0:.2f}".format(l)),
							util.toRed("{0:.2f}".format(lx)),
							util.toRed("{0:.2f}".format(lm))))
			if (i+1)%200==0:
				summaryWriter.add_summary(sess.run(summaryLoss,feed_dict=batch),i+1)
			if (i+1)%1000==0:
				summaryWriter.add_summary(sess.run(summaryImage,feed_dict=batch),i+1)
			if (i+1)%10000==0:
				util.saveModel(opt,sess,saver,i+1)
				print(util.toGreen("model saved: {0}/{1}, it.{2}".format(opt.group,opt.model,i+1)))
		dataloader.thread.join()

print(util.toYellow("======= TRAINING DONE ======="))
