
"""
modified by lijun
"""
import os
import numpy as np
import tensorflow as tf
from datagenerator import ImageDataGenerator
from datetime import datetime
import math
from tensorflow.data import Iterator
from tensorflow.contrib.slim.nets import resnet_v1
import tensorflow.contrib.slim as slim



def main():
    # 初始参数设置
    ratio = 0.3 # 分割训练集和验证集的比例
    learning_rate = 1e-3
    num_epochs = 17  # 代的个数 之前是10
    train_batch_size = 5 # 之前是1024
    test_batch_size = 1
    dropout_rate = 0.5
    num_classes = 4  # 类别标签
    display_step = 2 # display_step个train_batch_size训练完了就在tensorboard中写入loss和accuracy
                     # need: display_step <= train_dataset_size / train_batch_size

    filewriter_path = "./tmp/tensorboard"  # 存储tensorboard文件
    checkpoint_path = "./tmp/checkpoints"  # 训练好的模型和参数存放目录
    ckpt_path = "./resnet_v1_50.ckpt"
    image_format = 'jpg'

    # ============================================================================
    # -----------------生成图片路径和标签的List------------------------------------

    file_dir = './dataset'

    kitchenWaste = []
    label_kitchenWaste = []
    Recyclables = []
    label_Recyclables = []
    harmfulGarbage = []
    label_harmfulGarbage = []
    otherGarbage = []
    label_otherGarbage = []

    # step1：获取所有的图片路径名，存放到
    # 对应的列表中，同时贴上标签，存放到label列表中。

    for file in os.listdir(file_dir + '/kitchenWaste'):
        kitchenWaste.append(file_dir + '/kitchenWaste' + '/' + file)
        label_kitchenWaste.append(0)
    for file in os.listdir(file_dir + '/Recyclables'):
        Recyclables.append(file_dir + '/Recyclables' + '/' + file)
        label_Recyclables.append(1)
    for file in os.listdir(file_dir + '/harmfulGarbage'):
        harmfulGarbage.append(file_dir + '/harmfulGarbage' + '/' + file)
        label_harmfulGarbage.append(2)
    for file in os.listdir(file_dir + '/otherGarbage'):
        otherGarbage.append(file_dir + '/otherGarbage' + '/' + file)
        label_otherGarbage.append(3)

    # step2：对生成的图片路径和标签List做打乱处理
    image_list = np.hstack((kitchenWaste, Recyclables, harmfulGarbage, otherGarbage))
    label_list = np.hstack((label_kitchenWaste, label_Recyclables, label_harmfulGarbage, label_otherGarbage))

    # 利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # 将所有的img和lab转换成list
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])

    # 将所得List分为两部分，一部分用来训练tra，一部分用来测试val
    # ratio是测试集的比例
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample * ratio))  # 测试样本数
    n_train = n_sample - n_val  # 训练样本数

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    # get Datasets
    # 调用图片生成器，把训练集图片转换成三维数组
    train_data = ImageDataGenerator(
        images=tra_images,
        labels=tra_labels,
        batch_size=train_batch_size,
        num_classes=num_classes,
        image_format=image_format,
        shuffle=True)

    # 调用图片生成器，把测试集图片转换成三维数组
    test_data = ImageDataGenerator(
        images=val_images,
        labels=val_labels,
        batch_size=test_batch_size,
        num_classes=num_classes,
        image_format=image_format,
        shuffle=False)

    # get Iterators
    with tf.name_scope('input'):
        # 定义迭代器
        train_iterator = Iterator.from_structure(train_data.data.output_types,
                                        train_data.data.output_shapes)
        training_initalizer=train_iterator.make_initializer(train_data.data)
        test_iterator = Iterator.from_structure(test_data.data.output_types,
                                        test_data.data.output_shapes)
        testing_initalizer=test_iterator.make_initializer(test_data.data)
        # 定义每次迭代的数据
        train_next_batch = train_iterator.get_next()
        test_next_batch = test_iterator.get_next()

    x = tf.placeholder(tf.float32, [None, 227, 227, 3])
    y = tf.placeholder(tf.float32, [None, num_classes])
    keep_prob = tf.placeholder(tf.float32)

    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        fc8, end_points = resnet_v1.resnet_v1_50(x, num_classes, is_training=True)
        fc8 = tf.squeeze(fc8, [1, 2])

    # loss
    with tf.name_scope('loss'):    
        loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=fc8,
                                                                  labels=y))
    # optimizer
    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss_op)

    # accuracy
    with tf.name_scope("accuracy"):
        correct_pred = tf.equal(tf.argmax(fc8, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    init = tf.global_variables_initializer()

    # Tensorboard
    tf.summary.scalar('loss', loss_op)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(filewriter_path)

    # load vars
    flag = False
    if ckpt_path.split('.')[-2] == '/resnet_v1_50':
        init = tf.global_variables_initializer()
        variables = tf.contrib.framework.get_variables_to_restore()
        variables_to_resotre = [v for v in variables if v.name.split('/')[1] != 'logits' and
                                v.name.split('/')[0] != 'optimizer' and
                                v.name.split('/')[0] != 'save' and
                                v.name.split('/')[-1] != 'Adam:0' and
                                v.name.split('/')[-1] != 'Adam_1:0' and
                                v.name.split('/')[-1] != 'gamma:0']
        # saver = tf.train.Saver()
        saver = tf.train.Saver(variables_to_resotre)
        flag = True
    else:
        saver = tf.train.Saver()

    # 定义一代的迭代次数
    train_batches_per_epoch = int(np.floor(train_data.data_size / train_batch_size))
    test_batches_per_epoch = int(np.floor(test_data.data_size / test_batch_size))

    # Start training
    with tf.Session() as sess:
        if flag:
            sess.run(init)
        saver.restore(sess, ckpt_path)

        # Tensorboard
        writer.add_graph(sess.graph)

        print("{}: Start training...".format(datetime.now()))
        print("{}: Open Tensorboard at --logdir {}".format(datetime.now(),
                                                          filewriter_path))

        for epoch in range(num_epochs):
            sess.run(training_initalizer)
            print("{}: Epoch number: {} start".format(datetime.now(), epoch + 1))

            # train
            for step in range(train_batches_per_epoch):
                img_batch, label_batch = sess.run(train_next_batch)
                loss,_ = sess.run([loss_op,train_op], feed_dict={x: img_batch,
                                               y: label_batch,
                                               keep_prob: dropout_rate})
                if step % display_step == 0:
                    # loss
                    print("{}: loss = {}".format(datetime.now(), loss))

                    # Tensorboard
                    s = sess.run(merged_summary, feed_dict={x: img_batch,
                                                            y: label_batch,
                                                            keep_prob: 1.})
                    writer.add_summary(s, epoch * train_batches_per_epoch + step)
  
            # accuracy
            print("{}: Start validation".format(datetime.now()))
            sess.run(testing_initalizer)
            test_acc = 0.
            test_count = 0
            for _ in range(test_batches_per_epoch):
                img_batch, label_batch = sess.run(test_next_batch)
                acc = sess.run(accuracy, feed_dict={x: img_batch,
                                                    y: label_batch,
                                                    keep_prob: 1.0})
                test_acc += acc
                test_count += 1
            try:
                test_acc /= test_count
            except:
                print('ZeroDivisionError!')
            print("{}: Validation Accuracy = {:.4f}".format(datetime.now(), test_acc))

            # save model
            print("{}: Saving checkpoint of model...".format(datetime.now()))
            checkpoint_name = os.path.join(checkpoint_path, 'model_epoch' + str(epoch + 1) + '.ckpt')
            save_path = saver.save(sess, checkpoint_name)

            # this epoch is over
            print("{}: Epoch number: {} end".format(datetime.now(), epoch + 1))


if __name__ == '__main__':
    main()
