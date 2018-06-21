#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import os
import tensorflow as tf
import multiprocessing
import CNIC.ensemble_metrics1
import CNIC_C.ensemble_metrics1
import CNIC_H.ensemble_metrics1
import CNIC_HC.ensemble_metrics1
import CNIC_X.ensemble_metrics1

FLAGS=tf.flags.FLAGS
tf.flags.DEFINE_string("test_json_path","","Path of test json path.")
tf.logging.set_verbosity(tf.logging.INFO)
def main(_):
 model_list = ['./CNIC', './CNIC-C', './CNIC-H', 'CNIC-X', './CNIC-HC']
 p1 = multiprocessing.Process(target=CNIC.ensemble_metrics1.main, args=(FLAGS.test_json_path,))
 p1.start()
 p1.join()
 p2 = multiprocessing.Process(target=CNIC_C.ensemble_metrics1.main, args=(FLAGS.test_json_path,))
 p2.start()
 p2.join()
 p3 = multiprocessing.Process(target=CNIC_H.ensemble_metrics1.main, args=(FLAGS.test_json_path,))
 p3.start()
 p3.join()
 p4 = multiprocessing.Process(target=CNIC_X.ensemble_metrics1.main, args=(FLAGS.test_json_path,))
 p4.start()
 p4.join()
 p5 = multiprocessing.Process(target=CNIC_HC.ensemble_metrics1.main, args=(FLAGS.test_json_path,))
 p5.start()
 p5.join()
if __name__=="__main__":
 tf.app.run()

