"""
modified by lijun
"""
import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1
import tensorflow.contrib.slim as slim

ckpt_path = "./resnet_v1_50.ckpt"

# Opens Video Camera
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# get frame to judge pic size
isframeread, frame = cap.read()
if not isframeread:
    raise ValueError("Can't receive frame. Exiting ...")
height = frame.shape[0]
width = frame.shape[1]

# create placeholder according to pic size
x = tf.placeholder(tf.float32, [None, height, width, 3])
num_classes = 4

# inference model
with slim.arg_scope(resnet_v1.resnet_arg_scope()):
    fc8, end_points = resnet_v1.resnet_v1_50(x, num_classes, is_training=False)
    fc8 = tf.squeeze(fc8, [1, 2])
    prob = tf.nn.softmax(fc8)

# define model restore
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

# loading model
with tf.Session() as sess:
    if flag:
        sess.run(init)
    saver.restore(sess, ckpt_path)

    # 从相机持续获取图片
    while True:
        isframeread, frame = cap.read()

        # If frame is read correctly ret is True
        if not isframeread:
            print("Can't receive frame. Exiting ...")
            break

        # Convert to RGB colorspace because that fits model, and expand dims
        RGB_img = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        RGB_img = np.expand_dims(RGB_img, axis=0)

        # run inference and back process
        logits = sess.run(fetches=prob, feed_dict={x: RGB_img})
        class_num = np.argmax(logits)
        typedic = {0:'kitchenWaste',
                   1: 'Recyclables',
                   2: 'harmfulGarbage',
                   3: 'otherGarbage'}
        classify = typedic[class_num]
        confidence = round(logits[0][class_num], 2)

        # output and show
        if classify is not None:
            # This displays text if no ball is detected
            text = 'Class is:' + str(classify) + ' Confidence is:' + str(confidence)
            textsize = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            rows, columns, channels = frame.shape

            # Finds origin to center text
            textorigin = ((columns // 2) - (textsize[0][0] // 2), textsize[0][1] + 10)

            # Draw a Rectangle to help text stand out
            bottomleftvertex = (textorigin[0] - 5, textorigin[1] + 5)
            toprightvertex = (textorigin[0] + textsize[0][0] + 5, 5)
            cv.rectangle(frame, bottomleftvertex, toprightvertex, [0, 0, 255], cv.FILLED)

            # Draw text
            cv.putText(frame, text, textorigin, cv.FONT_HERSHEY_SIMPLEX, 0.5, [255, 255, 255], 1, cv.FILLED, False)

        # Show image
        cv.imshow('Result', frame)

        if cv.waitKey(1) == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


