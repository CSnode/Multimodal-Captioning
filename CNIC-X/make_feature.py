import cPickle
import pandas as pd
from collections import Counter
import tensorflow as tf
import numpy as np
import pandas as pd
from detector import Detector
from util import load_image
import os
import ipdb
import math

trainset_path='./allset.pickle'
weight_path='./caffe_layers_value.pickle'
batch_size=10
model_path='./models/multi-label/model-10'
trainset=pd.read_pickle(trainset_path)
test_index_file_path="./all_image_names.txt"
test_filenames=list(pd.read_table(test_index_file_path,header=None,names=['filename']).filename)
testset=trainset.loc[trainset['filename'].isin(test_filenames)]
image_path='../Flickr8k_Dataset/Flicker8k_Dataset'
testset['filename']=testset['filename'].map(lambda x:os.path.join(image_path,x))
wordset_path='./words.pkl'
with open(wordset_path)as f:
 wordset=cPickle.load(f)
inv_map={}
for i,x in enumerate(wordset):
 inv_map[i]=x
n_labels=len(wordset)
images_tf=tf.placeholder(tf.float32,[None,224,224,3],name="images")
detector=Detector(weight_path,n_labels)
c1,c2,c3,c4,conv5,conv6,gap,output=detector.inference(images_tf)
sig_output=tf.nn.sigmoid(output)
sess=tf.InteractiveSession()
saver=tf.train.Saver()
saver.restore(sess,model_path)
testset.index =range(len(testset))
result={}
for start,end in zip(range(0,len(testset)+batch_size,batch_size),range(batch_size,len(testset)+batch_size,batch_size)):
 current_data=testset[start:end]
 current_image_paths=current_data['filename'].values
 current_images=np.array(map(lambda x:load_image(x),current_image_paths))
 conv6_val,output_val,sig_output_val=sess.run([conv6,output,sig_output],feed_dict={images_tf:current_images})
 for x,y in zip(current_image_paths,sig_output_val):
  print os.path.basename(x)
  result[os.path.basename(x)]=y
with open("features.pkl",'w')as f:
 cPickle.dump(result,f)

