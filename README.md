## Attention Neural Bag of Feature Learning
This repository provides the implementation of the Attention Neural Bag of Feature Learning model proposed in [[1]](#anbof). The implementation of the attention layer and the quantization layers can be found in *models.py*   
 
We provide a script (*train_af.py*) that can be used to train the AF dataset [[2]](#af). Please download the preprocessed dataset [here](https://drive.google.com/file/d/1LWRmurTp_8Gi13u9nUIHSL7nZ6a5cCca/view?usp=sharing) and put the file in this directory. The AF dataset is splitted into 5 folds. To train the a model with the neural bag-of-feature layer (`nbof`) and temporal attention mechanism, we can run the following command::

	python train_af.py --fold-idx 0 --quantization-type nbof --attention-type temporal  

 
  
