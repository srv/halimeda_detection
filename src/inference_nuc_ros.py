#!/usr/bin/env python

import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1" # -1 --> Do not use CUDA
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

ros_path = '/opt/ros/kinetic/lib/python2.7/dist-packages'
conda_path="/home/sparus/anaconda3/bin:$PATH"

import sys
import time
import torch
import scipy
import numpy as np
import tensorflow as tf
import imageio.v2 as imageio
from threading import Thread, Lock

sys.path.append(ros_path) #afegir ros al puthon path per a que robi les llibreries de ros

if conda_path in sys.path:
	sys.path.remove(conda_path)

import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo


class Halimeda_detection:


	def __init__(self, name):
		self.name = name

		self.period = 1 # rospy.get_param('mine_detec/period')

		self.shape = 1024

		self.model_path_ss = "../models"
		self.model_path_od = "../models/od.pt"

		# Params
		self.init = False
		self.new_image = False
	
		# Set subscribers
		image_sub = message_filters.Subscriber('/stereo_down/left/image_rect_color', Image)
		info_sub = message_filters.Subscriber('/stereo_down/left/camera_info', CameraInfo)

		image_sub.registerCallback(self.cb_image)
		info_sub.registerCallback(self.cb_info)

		# Set publishers
		self.pub_img_merged = rospy.Publisher('segmented', Image, queue_size=4)

		# Set classification timer
		rospy.Timer(rospy.Duration(self.period), self.run)


	def cb_image(self, image):
		self.image = image
		self.new_image = True


	def cb_info(self, info):
		self.info = info


	def set_models(self):
		self.model_ss = tf.keras.models.load_model(os.path.join(self.model_path_ss, "model.h5"))
		self.model_od = torch.hub.load('/home/sparus/Halimeda/object_detection/yolov5', 'custom', path=self.model_path_od, source='local',force_reload = False)
		self.model_od.to(torch.device('cpu')).eval()


	def run(self,_):
		
		# New image available
		if not self.new_image:
			return
		self.new_image = False

		try:
			image = self.image
			header = self.image.header
			info = self.info

			info.width = self.shape
			info.height = self.shape

			self.pub_img_merged.header = header
			
		except:
			rospy.logwarn('[%s]: There is no input image to run the inference', self.name)
			return

		# Set model
		if not self.init: 
			self.set_models()
			self.init = True
			print("Model init")
			
		rospy.loginfo('[%s]: Starting inferences', self.name)	

		# Object detection
		image_np = np.array(np.frombuffer(image.data, dtype=np.uint8).reshape(1440, 1920,3))

		#image_np = imageio.imread("../halimeda_56.JPG")

		self.image_np_rsz = self.resize_volume(image_np)

		tinf1 = time.time()
		self.inference_ss()
		self.inference_od()
		# thread_ss = Thread(target=self.inference_ss)
		# thread_od = Thread(target=self.inference_od)
		# thread_ss.start()
		# thread_od.start()
		# thread_ss.join() # Don't exit while threads are running
		# thread_od.join() # Don't exit while threads are running		
		tinf2 = time.time()
		tinf = tinf2-tinf1
		print("inference took: " + str(tinf)  + "seconds")

		image_merged_np = self.merge()

		name = str(image.header.stamp.secs)
		imageio.imwrite(os.path.join("../out", name + "_merged.png"), image_merged_np)
		imageio.imwrite(os.path.join("../out", name + "_ss.png"), self.image_np_ss)
		imageio.imwrite(os.path.join("../out", name + "_od.png"), self.image_np_od)

		
	def inference_ss(self):
		tss1 = time.time()
		X_test = np.zeros((1, self.shape, self.shape, 3), dtype=np.uint8)
		X_test[0] = self.image_np_rsz
		preds_test = self.model_ss.predict(X_test)
		self.image_np_ss = np.squeeze(preds_test[0])*255
		#rospy.loginfo('[%s]: SS inference done', self.name)	
		tss2 = time.time()
		tss =tss2-tss1
		print("ss took: " + str(tss)  + "seconds")

	
	
	def inference_od(self):
		tod1 = time.time()
		dets_od = self.model_od([self.image_np_rsz])
		self.image_np_od = np.zeros([self.shape, self.shape], dtype=np.uint8) 
		dets_pandas = dets_od.pandas().xyxy[0]

		for index, row in dets_pandas.iterrows():
			conf = row['confidence']
			xmin=int(row['xmin'])
			ymin=int(row['ymin'])
			xmax=int(row['xmax'])
			ymax=int(row['ymax'])

			for j in range(ymin, ymax):	
				for k in range(xmin, xmax):	
					self.image_np_od[j, k] = int(255*conf)

		#rospy.loginfo('[%s]: OD inference done', self.name)	
		tod2 = time.time()
		tod =tod2-tod1
		print("od took: " + str(tod)  + "seconds")


	def merge(self):
		image_merged = self.image_np_ss*0.2 + self.image_np_od*0.8
		image_merged=np.asarray(image_merged)
		image_merged=image_merged.astype(np.uint8)
		image_merged = np.where(image_merged<22, 0, 255)
		return image_merged


	def resize_volume(self, img):
		
		desired_width = 1024
		desired_height = 1024
		desired_depth = 3

		current_width = img.shape[0]
		current_height = img.shape[1]
		current_depth = img.shape[2]

		width = current_width / desired_width
		height = current_height / desired_height
		depth = current_depth / desired_depth
		
		width_factor = 1 / width
		height_factor = 1 / height
		depth_factor = 1 / depth
		
		img = scipy.ndimage.zoom(img, (width_factor, height_factor, depth_factor), order=1)
		return img


if __name__ == '__main__':
	try:
		rospy.init_node('detect_halimeda')
		Halimeda_detection(rospy.get_name())

		rospy.spin()
	except rospy.ROSInterruptException:
		pass
