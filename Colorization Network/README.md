# Colorization Network
## Pre -processing
(A) <b>Generation of TXT file. </b>
After collecting the image dataset, we need to convert them to LMDB files. Run `./Data_preprocess.py` to get a picture list file `./train.txt`.<br>

(B) <b>Generation of LMDB file. </b>
If you have configured the Caffe environment, use the `create_imagenet.sh` under the caffe to make the LMDB file. The file can be found under `caffe/examples/imagenet`.

## Training Usage
The following contains instructions for training the colorization network, and how to obtain the probability distribution of the final prediction.

(A) <b>fetch caffemodel. </b>
Run `./train/fetch_init_model.sh`. This will load model `./models/init_v2.caffemodel`. 

(B) <b>Modify the file. </b>
Modify **source** in **data_param** `./models/colorization_train_val_v2.prototxt` to locate the path of LMDB file before.

(C) <b>Start training. </b>
Run `./train/train_model.sh [GPU_ID]`, where [GPU_ID] is the gpu you choose to specify. 
During the training process, every 1000 iterations will generate snapshots about `colornet_iter_[ITERNUMBER].caffemodel` and `colornet_iter_[ITERNUMBER].solverstate`, which can be used to resume network training when training is interupted. You can run `./train_resume.sh` and specify ITERNUMBER(iterations) in the `train_resume.sh`

(D) <b>Get color probability distribution. </b>
After getting the final Caffemodel, run `python Output_313pro.py` to get the final probability distribution, and the final output is npy file.
