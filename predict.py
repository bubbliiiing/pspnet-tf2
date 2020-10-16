#-------------------------------------#
#       对单张图片进行预测
#-------------------------------------#
from pspnet import Pspnet
from PIL import Image

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    
pspnet = Pspnet()

while True:
    img = input('Input image filename:')
    try:
        image = Image.open(img)
    except:         
        print('Open Error! Try again!')
        continue
    else:
        r_image = pspnet.detect_image(image)
        r_image.show()
