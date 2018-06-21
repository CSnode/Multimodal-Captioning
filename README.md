# Multimodal-Captioning
Chinese Image captioning method based on deep multimodal semantic fusion

Source code for Chinese image captioning method based on deep multimodal semantic fusion runnable on GPU and CPU.

### License
This code is released under the MIT License (refer to the LICENSE file for details).

## Dependencies
#### 1. tensorflow
The model is trained using Tensorflow, a popular Python framework for training deep neural network. To install Tensorflow, please refer to  [Installing Tensorflow](https://www.tensorflow.org/install/).
#### 2. python libs
The code is written in python, you also need to install following python dependencies:
- bottle==0.12.13
- ipdb==0.10.3
- matplotlib==2.1.0
- numpy==1.13.3
- Pillow==4.3.0
- scikit-image==0.13.1
- scipy==1.0.0
- jieba==0.38

For convenience, you can alse use requirements.txt to install python dependencies:

	pip install -r requirements.txt

To use the evaluation script: see
[coco-caption](https://github.com/tylin/coco-caption) for the requirements.

## Hardware
Though you can run the code on CPU, we highly recommend you to equip a GPU card. To run on cpu, please use

	export CUDA_VISIBLE_DEVICES=""

## Caption model
We have five different model to generate captions. For different model, please look at CNIC, CNIC-H, CNIC-C, CNIC-X, CNIC-HC directory, training and testing command for eash caption model is the same. For simplicity, we use CNIC as an example:

	cd CNIC

## Prepare data
To generate training data, please use build_data.py script:

	python build_data.py --train_image_dir="../Dataset/Images" --val_image_dir="../Dataset/Images"  --train_captions_file="data/captions_train.json" --val_captions_file="data/captions_val.json" --test_captions_file="data/captions_test_c.json" --output_dir="data/records"

## Train single-lable visual encoding model
We use Google Inception V3 for single-lable visual encoding network: see
[Inception](https://github.com/tensorflow/models/tree/master/research/inception) for the instructions.

## Train multi-lable keyword prediction model
Please run train_keyword.py using gpu:

	CUDA_VISIBLE_DEVICES=0 python train_keyword.py

## Train multimodal caption generation model
For multimodal caption generation network use train.py:

	python train.py --input_file_pattern="data/records/train-?????-of-00256" --inception_checkpoint_file="data/inception_v3.ckpt" --train_dir="models/train" --train_inception=false –number_of_steps=1000000 
  
## Validate on dev split
For validation on dev split and select best model please use evaluate_every.py:

	python evaluate_every.py  --input_file_pattern="data/records/val-?????-of-00004" --checkpoint_dir="models/train"  --eval_dir="models/eval"

## Generate caption and visualize
Use server.py to load models, and use client.py to request caption generation:

	python server.py
 	python client.py

## Tensorboard visualization
To use tensorboard to monitor training process:

	tensorboard --logdir="MODEL_PATH"
  
## Test metrics
For metrics evaluation process:

	python metrics.py --checkpoint_path="models/train/model.ckpt-xxx" --vocab_file="data/word_counts.txt" --image_path="../Flickr8k_Dataset/Flicker8k_Dataset" --temp_path="./" --test_json_path="./data/captions_test_c.json" --keyword_pickle_file="data/features.pkl"
   
## Ensemble metrics
For ensemble, you need to complete CNIC, CNIC-H, CNIC-C, CNIC-HC model training, and use ensemble.py:

	cd ../Ensemble
	python ensemble.py --test_json_path="./Data/captions_test_c.json"
