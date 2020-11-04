import time
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.optimizers import Adam
from nets.pspnet import pspnet
from nets.pspnet_training import Generator, dice_loss_with_CE, CE
from utils.metrics import Iou_score, f_score
from utils.utils import ModelCheckpoint
from tqdm import tqdm
from functools import partial

# 防止bug
def get_train_step_fn():
    @tf.function
    def train_step(images, labels, net, optimizer, loss):
        with tf.GradientTape() as tape:
            # 计算loss
            prediction = net(images, training=True)
            loss_value = loss(labels, prediction)

        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        
        _f_score = f_score()(labels, prediction)
        return loss_value, _f_score
    return train_step

@tf.function
def val_step(images, labels, net, optimizer, loss):
    # 计算loss
    prediction = net(images, training=False)
    loss_value = loss(labels, prediction)
    _f_score = f_score()(labels, prediction)

    return loss_value, _f_score

def fit_one_epoch(net, loss, optimizer, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, train_step):
    total_loss = 0
    val_loss = 0
    total_f_score = 0
    val_f_score = 0

    start_time = time.time()
    with tqdm(total=epoch_size,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration>=epoch_size:
                break
            images, labels = batch[0], batch[1]
            labels = tf.cast(tf.convert_to_tensor(labels),tf.float32)

            loss_value, _f_score = train_step(images, labels, net, optimizer, loss)
            total_loss      += loss_value.numpy()
            total_f_score   += _f_score.numpy()

            waste_time = time.time() - start_time
            pbar.set_postfix(**{'Total Loss'        : total_loss / (iteration + 1), 
                                'Total f_score'     : total_f_score / (iteration + 1),
                                'lr'                : optimizer._decayed_lr(tf.float32).numpy(),
                                's/step'            : waste_time})
            pbar.update(1)
            start_time = time.time()
        
    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration>=epoch_size_val:
                break
            # 计算验证集loss
            images, labels = batch[0], batch[1]
            labels = tf.convert_to_tensor(labels)

            loss_value, _f_score = val_step(images, labels, net, optimizer, loss)
            # 更新验证集loss
            val_loss    += loss_value.numpy()
            val_f_score += _f_score.numpy()

            waste_time = time.time() - start_time
            pbar.set_postfix(**{'Val Loss'      : val_loss / (iteration + 1), 
                                'Val f_score'   : val_f_score / (iteration + 1),
                                's/step'        : waste_time})
            pbar.update(1)
            start_time = time.time()

    print('Finish Validation')
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
    net.save_weights('logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.h5'%((epoch+1),total_loss/(epoch_size+1),val_loss/(epoch_size_val+1)))
      
mobilenet_freeze = 146
resnet_freeze = 172

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

if __name__ == "__main__":     
    inputs_size = [473,473,3]
    log_dir = "logs/"
    #---------------------#
    #   分类个数+1
    #   2+1
    #---------------------#
    num_classes = 21
    #--------------------------------------------------------------------#
    #   建议选项：
    #   种类少（几类）时，设置为True
    #   种类多（十几类）时，如果batch_size比较大（10以上），那么设置为True
    #   种类多（十几类）时，如果batch_size比较小（10以下），那么设置为False
    #---------------------------------------------------------------------# 
    dice_loss = False
    #---------------------#
    #   主干网络选择
    #   mobilenet
    #   resnet50
    #---------------------#
    backbone = "mobilenet"
    #---------------------#
    #   是否使用辅助分支
    #   会占用大量显存
    #---------------------#
    aux_branch = False
    #---------------------#
    #   下采样的倍数
    #   16显存占用小
    #   8显存占用大
    #---------------------#
    downsample_factor = 16
    #-------------------------------#
    #   Dataloder的使用
    #-------------------------------#
    Use_Data_Loader = True

    # 获取model
    model = pspnet(num_classes,inputs_size,downsample_factor=downsample_factor,backbone=backbone,aux_branch=aux_branch)
    model.summary()

    #-------------------------------------------#
    #   权值文件的下载请看README
    #   权值和主干特征提取网络一定要对应
    #-------------------------------------------#
    model_path = "./model_data/pspnet_mobilenetv2.h5"
    model.load_weights(model_path, by_name=True, skip_mismatch=True)

    # 打开数据集的txt
    with open(r"VOCdevkit\VOC2007\ImageSets\Segmentation\train.txt","r") as f:
        train_lines = f.readlines()

    # 打开数据集的txt
    with open(r"VOCdevkit\VOC2007\ImageSets\Segmentation\val.txt","r") as f:
        val_lines = f.readlines()
        

    if backbone=="mobilenet":
        freeze_layers = mobilenet_freeze
    else:
        freeze_layers = resnet_freeze

    for i in range(freeze_layers): 
        model.layers[i].trainable = False
    print('Freeze the first {} layers of total {} layers.'.format(freeze_layers, len(model.layers)))

    loss = dice_loss_with_CE() if dice_loss else CE()
    #------------------------------------------------------#
    #   主干特征提取网络特征通用，冻结训练可以加快训练速度
    #   也可以在训练初期防止权值被破坏。
    #   Init_Epoch为起始世代
    #   Freeze_Epoch为冻结训练的世代
    #   Epoch总训练世代
    #   提示OOM或者显存不足请调小Batch_size
    #------------------------------------------------------#
    if True:
        Lr = 1e-4
        Init_Epoch = 0
        Freeze_Epoch = 50
        Batch_size = 8

        if Use_Data_Loader:
            gen = partial(Generator(Batch_size, train_lines, inputs_size, num_classes,aux_branch).generate, random_data = True)
            gen = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))
                
            gen_val = partial(Generator(Batch_size, val_lines, inputs_size, num_classes,aux_branch).generate, random_data = False)
            gen_val = tf.data.Dataset.from_generator(gen_val, (tf.float32, tf.float32))

            gen = gen.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)
            gen_val = gen_val.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)

        else:
            gen = Generator(Batch_size, train_lines, inputs_size, num_classes,aux_branch).generate()
            gen_val = Generator(Batch_size, val_lines, inputs_size, num_classes,aux_branch).generate(False)


        epoch_size = len(train_lines)//Batch_size
        epoch_size_val = len(val_lines)//Batch_size
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=Lr,
            decay_steps=epoch_size,
            decay_rate=0.95,
            staircase=True
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        for epoch in range(Init_Epoch,Freeze_Epoch):
            fit_one_epoch(model, loss, optimizer, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, get_train_step_fn())

    for i in range(freeze_layers): 
        model.layers[i].trainable = True

    if True:
        Lr = 1e-5
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100
        Batch_size = 4
        
        if Use_Data_Loader:
            gen = partial(Generator(Batch_size, train_lines, inputs_size, num_classes,aux_branch).generate, random_data = True)
            gen = tf.data.Dataset.from_generator(gen, (tf.float32, tf.float32))
                
            gen_val = partial(Generator(Batch_size, val_lines, inputs_size, num_classes,aux_branch).generate, random_data = False)
            gen_val = tf.data.Dataset.from_generator(gen_val, (tf.float32, tf.float32))

            gen = gen.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)
            gen_val = gen_val.shuffle(buffer_size=Batch_size).prefetch(buffer_size=Batch_size)

        else:
            gen = Generator(Batch_size, train_lines, inputs_size, num_classes,aux_branch).generate()
            gen_val = Generator(Batch_size, val_lines, inputs_size, num_classes,aux_branch).generate(False)

        epoch_size = len(train_lines)//Batch_size
        epoch_size_val = len(val_lines)//Batch_size
        
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=Lr,
            decay_steps=epoch_size,
            decay_rate=0.95,
            staircase=True
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        for epoch in range(Freeze_Epoch,Unfreeze_Epoch):
            fit_one_epoch(model, loss, optimizer, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, get_train_step_fn())
