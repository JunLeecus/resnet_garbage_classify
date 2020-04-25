# tensorflow_resnet_classify
> This repository aims to implement a ResNet with TensorFlow. 
> The train file contains 30000 images (garbage). 
> We built this ResNet in Windows ,  it's very convenient for most of you to train the net.

## Requirements
* Python 3.7
* TensorFlow 1.14.0
* Numpy
* garbage images [here](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)

## Pre-trained model
* Download the official pre-trained resnet_v1_50 model from [here](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)
## Usage 
1'  Make sure that you have already changed file directory to the right format.

**example:**


    /dataset/kitchenWaste/kitchenWaste_1.jpg

    /dataset/Recyclables/Recyclables_1.jpg

    /dataset/harmfulGarbage/harmfulGarbage_2.jpg
    
	/dataset/otherGarbage/otherGarbage_1.jpg

    
2'  Modify parameters of the beginning of main function in the train_resnet.py file.

**example:**


    ratio = 0.3
    learning_rate = 1e-3
    num_epochs = 17
    train_batch_size = 5
    test_batch_size = 1
    dropout_rate = 0.5
    num_classes = 4 
    display_step = 2 
    
    filewriter_path = "./tmp/tensorboard" 
    checkpoint_path = "./tmp/checkpoints"  
    ckpt_path = "./resnet_v1_50.ckpt"
    
    image_format = 'jpg'



## Notes:
* 模型直接调用了 slim.resnet_v1 ,并将其转化为4分类任务，因此导入的预训练模型不含最后全连接层。为了能够导入预训练，对恢复的参数进行了筛选，而加载本代码训练好的模型不会进行删减。
* The datagenerator.py files have been builded, you don't have to modify it. But if you have more concise or effective codes, please do share them with us.
* The train_resnet.py is aimed to tune the weights and bias in the resnet.
* This model is easily transfered into a multi-class classification model. All you need to do is modifying parameters of the beginning of main function in the train_resnet.py file.

## Output:
* The videoclassify.py files calls your camera to capture video, and classify garbage in the video. Unfortunately, there are no backgroud class which means it can only show the type of garbage without on garbage.

