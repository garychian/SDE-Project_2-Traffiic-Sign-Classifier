# Traffic Sign Classifier
 
## Overview

In this project, I used deep neural networks and three classic convolutional neural network architectures(**LeNet, AlexNet, GooLeNet**) to classify the traffic signs. 

The traffic sign datasets using [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) 

All the images has been resized to 32x32 and contain training, validation and test set. You can download the [dataset](https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip)

## Goals/ steps of this project are following

* Load and explore the data set.
* Realize **LeNet** architecture and use `Relu`, `mini-batch gradient descent` and `dropout`.
* Realize **AlexNet** and make some modifications, use `learning rate decay`, `Adam optimization` and ` L2 regulization`
* Use **GoogLeNet** to classify traffic signs and make some modifications, use `inceptionn` and `average pooling`
* Analyze the softmax probabilities of the new images
* Summarize the results and difference between these two model.  

## Data Pre-process
I used the numpy library to calculate summary statistics of the traffic signs data set:

The size of training set is: 34799

The size of the validation set is: 4410

The size of test set is: 12630

The shape of a traffic sign image is: (32, 32 ,3)

The number of unique classes/labels in the data set is: 43

Here is an exploratory visualization of the training data set.
![](https://github.com/liferlisiqi/Traffic-Sign-Classifier/blob/master/result_images/exploratory.jpg)

The distribution of training, validation and testing set is showing in the following bar charts.
![](https://github.com/liferlisiqi/Traffic-Sign-Classifier/blob/master/result_images/distribution.jpg)








## LeNet
The LeNet model is proposed by Yann LeCun in 1998, it is the most classific cnn model for image recognition, its architecture is as following:
![](https://github.com/liferlisiqi/Traffic-Sign-Classifier/blob/master/result_images/lenet.png)

In the LeNet architecture I realized for traffic signs recognition, three tricks as used as follows:

1 ReLu
ReLu nonlinear function is used as the activation function after the convolutional layer. More information about ReLu and other activation functions can be find at [Lecture 6 | Training Neural Networks I](https://www.youtube.com/watch?v=wEoyxE0GP2M&index=6&list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&t=0s).

2 Mini-batch gradient descent
Mini-batch gradient descent is the combine of batch gradient descent and stochastic gradient descent, it is based on the statistics to estimate the average of gradient of all the training data by a batch of selected samples. You can find more details [here](https://machinelearningmastery.com/gentle-introduction-mini-batch-gradient-descent-configure-batch-size/)

3 Dropout
Dropout is a regularization technique for reducing overfitting in neural networks by preventing complex co-adaptations on training data. It is proposed in the paper Dropout: A Simple Way to Prevent Neural Networks from Overfitting. It is usually after fully connected layers. Awkwardly, there is a very small problem that LeNet will not overfitting to trainging set sometimes. Thus the dropout will not play a big role or even make the model worse for simple like LeNet. And the training set error maybe be higher than validation set error while training.

### Model details
| Layer         		|Stride|Pad|Kernel size|in |out|# of Param      | Input     | Output      |
|:---------------------:|:----:|:-:|:---------:|:-:|:-:|:--------:      |:---------:|:-----------:| 
| Convolution1       	|  1   | 0 |     5x5   | 3 | 6 |   456  	    | 32x32x3   | 28x28x6     |
| Max pooling1	      	|  2   | 0 |     2x2   | 6 | 6 |   0	        | 28x28x6   | 14x14x6     |
| Convolution2       	|  1   | 0 |     5x5   | 6 | 16|   2416         | 14x14x6   | 10x10x16    |
| Max pooling2	      	|  2   | 0 |     2x2   | 16|16 |   0            | 10x10x16  | 5x5x16      |
| Flatten				|      |   |           |   |   |         	  	| 5x5x16    | 400         |
| Fully connected1		|      |   |     1x1   |400|120|   48000	  	| 400       | 120         |
| Fully connected2		|      |   |     1x1   |120|80 |   9600 	    | 120       | 80          |
| Fully connected3		|      |   |     1x1   |80 | 43|   3440	  		| 80        | 43          |

The network has a total of 63,912 parameters.

### Training
I have turned the following three hyperparameters to train my model.
LEARNING_RATE = 1e-2
EPOCHS = 50
BATCH_SIZE = 16

The results are:

accuracy of training set: 99.8%
accuracy of validation set: 96.3%
accuracy of test set: 94.2%

running time: 37min 53s

We can see that the model is overfitting to the training data and the accuracy on validation set is a little lower than on training set. The LeNet model is efficient and simple, many cnn architectures are inspired by it, like AlexNet.

## AlexNet
![](https://github.com/liferlisiqi/Traffic-Sign-Classifier/blob/master/result_images/alexnet.png)

### Model Deatails 
| Layer         		|Stride|Pad|Kernel size|in |out|# of Param      | Input     | Output      |
|:---------------------:|:----:|:-:|:---------:|:-:|:-:|:--------:      |:---------:|:-----------:| 
| Convolution1       	|  1   | 0 |     5x5   | 3 | 9 |   684  	    | 32x32x3   | 28x28x9     |
| Max pooling1	      	|  2   | 0 |     2x2   | 9 | 9 |   0	        | 28x28x9   | 14x14x9     |
| Convolution2       	|  1   | 0 |     3x3   | 9 | 32|   2624         | 14x14x9   | 12x12x32    |
| Max pooling2	      	|  2   | 0 |     2x2   | 32|32 |   0            | 12x12x32  | 6x6x32      |
| Convolution3      	|  1   | 2 |     3x3   | 32| 48|   13872	   	| 6x6x32    | 6x6x48      |
| Convolution4       	|  1   | 2 |     3x3   | 48|64 |   27712	   	| 6x6x48    | 6x6x64      |
| Convolution5       	|  1   | 2 |     3x3   | 64| 96|   55392	  	| 6x6x64    | 6x6x96      |
| Max pooling3	      	|  2   | 0 |     2x2   | 96| 96|   0            | 6x6x96    | 3x3x96      |
| Flatten				|      |   |           |   |   |         	  	| 3x3x96    | 864         |
| Fully connected1		|      |   |     1x1   |864|400|   345600	  	| 864       | 400         |
| Fully connected2		|      |   |     1x1   |400|160|   64000	    | 400       | 160         |
| Fully connected3		|      |   |     1x1   |160| 43|   6880	  		| 160       | 43          |

The network has a total of 516,764 parameters.

* Learning rate decay
	
	In training deep networks, when the learning rate is large, the system contains too much kinetic energy and the parameter vector bounces around chaotically, unable to seetle down to deep. 
	When the learning rate is too small, we will waste computation bouncing around chaotically with improvement for a long time. If the learning rate can decay from large to small whilr training, the network will move fast at the begining and improve little by little in the end. 

	There are three commonly used types of mrthod, you can find [here](https://cs231n.github.io/neural-networks-3/#anneal)
	* Step Decay
	* Exponential decay and 1/t decay

* Adam optimization 
	Adam is a popular optimization recently proposed by Diederik P. Kingma and Jimmy Ba, like previous proposed Adagrad and RMSprop, it is a kind of adaptive learning rate method. With Adam, we don't have to use learning rate decay and tune three parameters for perfect learning rate. It is fabilous, so I will use it in most of times. After adapting Adam, the accuracy for training set, validation set and testing set are 99.9%, 96.9% and 94.2% respectively. The model is a little overfitting to training set, so some regularization methods are used to reduce it.

* L2 regulization 
	L2 regulization is used to reduce overfitting by adding regulization loss to loss function, it is based on the assume that the bigger regulization loss is the more complex the model is. It is well known that complex model is more easily overfit to training set, thus, through reducing regulization loss to make the model simpler. The regulization loss is the sum of L2 norm of weights for each layer multiple regulization parameter `lambda` in most cases, `lambda` is a small positive number that controls the regulization degree. Tensorflow documetn for how to use l2 regulization can be find [here](https://www.tensorflow.org/api_docs/python/tf/nn/l2_loss).

### Training 

* LEARNING_RATE = 5e-4
* EPOCHS = 30
* BATCH_SIZE = 64
* keep_prop = 0.5
* LAMBDA = 1e-5

The results are:
* accuracy of training set: 99.9%
* accuracy of validation set: 96.3%
* accuracy of test set: 94.4%
* running time: 38min 18s


## GoogLeNet
GoogLeNet was the winner of the ILSVRC 2014, it main contribution was the development of Inception Module that dramatically reduced the number of parameters in the network.

![](https://github.com/liferlisiqi/Traffic-Sign-Classifier/blob/master/result_images/inception.jpg)

Additionally, this paper uses Average Pooling instead of Fully connected layer at the top of the ConvNet, eliminating a large amount of parameters that do not seem to matter much. The overall architecture of GoogLeNet is as the following table.

![](https://github.com/liferlisiqi/Traffic-Sign-Classifier/blob/master/result_images/GoogLeNet.png)

Why come up inception? In an image classification tasks, the size of salient feature can considerably vary within the image frame. Hence, deciding on a fixed kernel size is rather difficult. Large kernels are preferred for more global features that are distributed over large area of the image, on the other hand, smaller kernels provide good resluts in detecting area specific features that are distributed across the image frame. So for more effective, we need kernels of diifferent sizes. 

That is what is **inception** does. Instead of simply going deeper in terms of number of layers, it goes wider. 

The classic GoogLeNet will cost a bit of computional power, so in the this implementation, the No. of layers from 22 to 14. 

### Model details

| Type          | Kernel/Stride	| Output    | Parameters  |
|:-------------:|:-------------:|:---------:|:-----------:|
| conv          | 3x3/2x2       | 16x16x64  | 1,792       |
| inception(2a) |               | 16x16x256 | 137,072     |
| inception(2b)	|               | 16x16x480 | 388,736     |
| max pool    	| 3x3/2x2      	| 7x7x480   |             |
| inception(3a) |  	            | 7x7x512   | 433,792     |
| inception(3a) |  	            | 7x7x512   | 449,160     |
| max pool 	    | 3x3/2x2  	    | 3x3x512   |             |
| inception(4a) |  	            | 3x3x832   | 859,136     |
| inception(4a) |  	            | 3x3x1024  | 1,444,080   |
| avg pool 	    | 3x3/1x1  	    | 1x1x1024  |             |
| flatten	    | 864			| 1024      |             |
| full		    | 43            | 43        | 44,032      |
|**inception**  |               |           |             |
| conv11        | 1x1/1x1       |           |             |
| conv33_reduce | 1x1/1x1       |           |             |
| conv33        | 3x3/1x1       |           |             |
| conv55_reduce | 1x1/1x1       |           |             |
| conv55        | 5x5/1x1       |           |             |
| pool_proj     | 3x3/1x1       |           |             |
| pool11        | 1x1/1x1       |           |             |

### I have turned the following three hyperparameters to train my model.

LEARNING_RATE = 5e-4
EPOCHS = 35
BATCH_SIZE = 128
keep_prop = 0.5
The results are:

accuracy of training set: 100.0%
accuracy of validation set: 98.5%
accuracy of test set: 98.1%



## Summary 
AlexNet: was born out of need to imorove the results of ImageNet challenge. At 2012, AlexNet achieve considerable accuracy of 84.7% on competition. AlexNet use *ReLU* activation function which sloved the **Vanishing Gradient** problem. And also achieved 25% error rate about 6 times faster than the same network with *tanh* non linearity. 