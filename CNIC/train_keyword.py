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

weight_path='caffe_layers_value.pickle'
model_path='./models/multi-label/'
pretrained_model_path=None 
n_epochs=100
init_learning_rate=0.01
weight_decay_rate=0.0005
momentum=0.9
batch_size=10
allset_path='./allset.pickle'
wordset_path='./words.pkl'
image_path='../Flickr8k_Dataset/Flicker8k_Dataset'
allset=pd.read_pickle(allset_path)
train_index_file_path="./Flickr_8k.trainImages.txt"
val_index_file_path="./Flickr_8k.devImages.txt"
test_index_file_path="./Flickr_8k.testImages.txt"

test_filenames=list(pd.read_table(test_index_file_path,header=None,names=['filename']).filename)
val_filenames=list(pd.read_table(val_index_file_path,header=None,names=['filename']).filename)
train_filenames=list(pd.read_table(train_index_file_path,header=None,names=['filename']).filename)
testset=allset.loc[allset['filename'].isin(test_filenames)]
valset=allset.loc[allset['filename'].isin(val_filenames)]
trainset=allset.loc[allset['filename'].isin(train_filenames)]
trainset['filename']=trainset['filename'].map(lambda x:os.path.join(image_path,x))
valset['filename']=valset['filename'].map(lambda x:os.path.join(image_path,x))
testset['filename']=testset['filename'].map(lambda x:os.path.join(image_path,x))
with open(wordset_path)as f:
 wordset=cPickle.load(f)
n_labels=len(wordset)

learning_rate=tf.placeholder(tf.float32,[])
images_tf=tf.placeholder(tf.float32,[None,224,224,3],name="images")
labels_tf=tf.placeholder(tf.float32,[None,319],name='labels')
detector=Detector(weight_path,n_labels)
p1,p2,p3,p4,conv5,conv6,gap,output=detector.inference(images_tf)
sig_output=tf.nn.sigmoid(output)
loss_tf=-tf.reduce_mean((labels_tf*tf.log(sig_output+1e-9))+((1-labels_tf)*tf.log(1-sig_output+1e-9)))
weights_only=filter(lambda x:x.name.endswith('W:0'),tf.trainable_variables())
weight_decay=tf.reduce_sum(tf.stack([tf.nn.l2_loss(x)for x in weights_only]))*weight_decay_rate
loss_tf+=weight_decay
sess=tf.InteractiveSession()
saver=tf.train.Saver(max_to_keep=500)
optimizer=tf.train.MomentumOptimizer(learning_rate,momentum)
grads_and_vars=optimizer.compute_gradients(loss_tf)
grads_and_vars=map(lambda gv:(gv[0],gv[1])if('conv6' in gv[1].name or 'GAP' in gv[1].name)else(gv[0]*0.1,gv[1]),grads_and_vars)
train_op=optimizer.apply_gradients(grads_and_vars)
tf.initialize_all_variables().run()

if pretrained_model_path:
 print "Pretrained"
 saver.restore(sess,pretrained_model_path)
testset.index =range(len(testset))
valset.index =range(len(valset))
iterations=0
loss_list=[]
for epoch in range(n_epochs):
 trainset.index=range(len(trainset))
 trainset=trainset.ix[np.random.permutation(len(trainset))]
 for start,end in zip(range(0,len(trainset)+batch_size,batch_size),range(batch_size,len(trainset)+batch_size,batch_size)):
  current_data=trainset[start:end]
  current_image_paths=current_data['filename'].values
  current_images=np.array(map(lambda x:load_image(x),current_image_paths))
  good_index=np.array(map(lambda x:x is not None,current_images))
  current_data=current_data[good_index]
  current_images=np.stack(current_images[good_index])
  current_labels=current_data['label'].values
  current_labels=np.array(list(current_labels))
  _,loss_val,output_val,sig_output_val=sess.run([train_op,loss_tf,output,sig_output],feed_dict={learning_rate:init_learning_rate,images_tf:current_images,labels_tf:current_labels})
  loss_list.append(loss_val)
  iterations+=1
  if iterations%5==0:
   print "======================================"
   print "Epoch",epoch,"Iteration",iterations
   print "Processed",start,'/',len(trainset)
   rlabel_predictions=sig_output_val.round()
   temp=(current_labels==rlabel_predictions)
   acc=float(1.0*np.count_nonzero(temp)/(temp.shape[0]*temp.shape[1]))
   print "Train Accuracy:",acc,'/',(current_labels==rlabel_predictions).shape[0]*(current_labels==rlabel_predictions).shape[1]
   print "Training Loss:",np.mean(loss_list)
   print "\n"
   loss_list=[]
 n_correct=0
 n_data=0
 for start,end in zip(range(0,len(testset)+batch_size,batch_size),range(batch_size,len(testset)+batch_size,batch_size)):
  current_data=testset[start:end]
  current_image_paths=current_data['filename'].values
  current_images=np.array(map(lambda x:load_image(x),current_image_paths))
  good_index=np.array(map(lambda x:x is not None,current_images))
  current_data=current_data[good_index]
  current_images=np.stack(current_images[good_index])
  current_labels=current_data['label'].values
  current_labels=list(current_labels)
  current_labels=np.array(current_labels)
  output_vals,sig_output_vals=sess.run([output,sig_output],feed_dict={images_tf:current_images})
  rlabel_predictions=sig_output_vals.round()
  temp=(current_labels==rlabel_predictions)
  acc=np.count_nonzero(temp)
  n_correct+=acc
  n_data+=len(current_data)
 acc_all=1.0*n_correct/float(n_data)/temp.shape[1]
 f_log=open('./results/log.multi-label','a')
 f_log.write('epoch:'+str(epoch)+'\ttest acc:'+str(acc_all)+'/'+str(n_data)+'\n')
 f_log.close()
 print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
 print 'epoch:'+str(epoch)+'\ttest acc:'+str(acc_all)+'/'+str(n_data)+'\n'
 print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
 n_correct=0
 n_data=0
 for start,end in zip(range(0,len(valset)+batch_size,batch_size),range(batch_size,len(valset)+batch_size,batch_size)):
  current_data=valset[start:end]
  current_image_paths=current_data['filename'].values
  current_images=np.array(map(lambda x:load_image(x),current_image_paths))
  good_index=np.array(map(lambda x:x is not None,current_images))
  current_data=current_data[good_index]
  current_images=np.stack(current_images[good_index])
  current_labels=current_data['label'].values
  current_labels=list(current_labels)
  current_labels=np.array(current_labels)
  output_vals,sig_output_vals=sess.run([output,sig_output],feed_dict={images_tf:current_images})
  rlabel_predictions=sig_output_vals.round()
  temp=(current_labels==rlabel_predictions)
  acc=np.count_nonzero(temp)
  n_correct+=acc
  n_data+=len(current_data)
 acc_all=1.0*n_correct/float(n_data)/temp.shape[1]
 f_log=open('./results/log.mil','a')
 f_log.write('epoch:'+str(epoch)+'\tval acc:'+str(acc_all)+'/'+str(n_data)+'\n')
 f_log.close()
 print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
 print 'epoch:'+str(epoch)+'\tval acc:'+str(acc_all)+'/'+str(n_data)+'\n'
 print "$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$"
 saver.save(sess,os.path.join(model_path,'model'),global_step=epoch)
 init_learning_rate*=0.99

