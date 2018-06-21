# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import show_and_tell_model
from inference_utils import inference_wrapper_base
class InferenceWrapper(inference_wrapper_base.InferenceWrapperBase):
 def __init__(self):
  super(InferenceWrapper,self).__init__()
 def build_model(self,model_config):
  model=show_and_tell_model.ShowAndTellModel(model_config,mode="inference")
  model.build()
  return model
 def feed_image_and_word_prob(self,sess,encoded_image,word_prob):
  initial_state=sess.run(fetches="lstm/initial_state:0",feed_dict={"image_feed:0":encoded_image,"word_prob_feed:0":word_prob})
  return initial_state
 def inference_step(self,sess,input_feed,state_feed):
  softmax_output,state_output=sess.run(fetches=["softmax:0","lstm/state:0"],feed_dict={"input_feed:0":input_feed,"lstm/state_feed:0":state_feed,})
  return softmax_output,state_output,None


