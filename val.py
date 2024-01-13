import os
import stat
import numpy as np
import matplotlib.pyplot as plt

import mindspore.nn as nn
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore import dtype as mstype
from mindspore.train.callback import TimeMonitor, Callback
from mindspore import Model, Tensor, context, save_checkpoint, load_checkpoint, load_param_into_net
from resnet import resnet50


#设置使用设备，CPU/GPU/Ascend
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

#数据路径
train_data_path = 'dataset/train'
val_data_path = 'dataset/val'


def create_dataset(data_path, batch_size=24, repeat_num=1):
    """定义数据集"""
    data_set = ds.ImageFolderDataset(data_path, num_parallel_workers=8, shuffle=True)

    # 对数据进行增强操作
    image_size = [224, 224]
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    trans = [
        CV.Decode(),
        CV.Resize(image_size),
        CV.Normalize(mean=mean, std=std),
        CV.HWC2CHW()
    ]
    type_cast_op = C.TypeCast(mstype.int32)

    # 实现数据的map映射、批量处理和数据重复的操作
    data_set = data_set.map(operations=trans, input_columns="image", num_parallel_workers=8)
    data_set = data_set.map(operations=type_cast_op, input_columns="label", num_parallel_workers=8)
    data_set = data_set.batch(batch_size, drop_remainder=True)
    data_set = data_set.repeat(repeat_num)

    return data_set

# 定义网络并加载参数，对验证集进行预测
def visualize_model(best_ckpt_path,val_ds):
    net = resnet50(2)
    param_dict = load_checkpoint(best_ckpt_path)
    load_param_into_net(net,param_dict)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True,reduction='mean')
    model = Model(net, loss,metrics={"Accuracy":nn.Accuracy()})
    data = next(val_ds.create_dict_iterator())
    images = data["image"].asnumpy()
    labels = data["label"].asnumpy()
    class_name = {0:"husky",1:"labrador"}
    output = model.predict(Tensor(data['image']))
    print(output)
    pred = np.argmax(output.asnumpy(), axis=1)

    # 可视化模型预测
    plt.figure(figsize=(12,5))
    for i in range(len(labels)):
        plt.subplot(3,8,i+1)
        color = 'blue' if pred[i] == labels[i] else 'red'
        plt.title('pre:{}'.format(class_name[pred[i]]), color=color)
        picture_show = np.transpose(images[i],(1,2,0))
        picture_show = picture_show/np.amax(picture_show)
        picture_show = np.clip(picture_show, 0, 1)
        plt.imshow(picture_show)
        plt.axis('off')
    plt.show()


# 定义网络
net = resnet50(2)
num_epochs=20

# 定义优化器和损失函数
opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.01, momentum=0.9)
loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

# 实例化模型
model = Model(net, loss,opt,metrics={"Accuracy":nn.Accuracy()})

# 加载验证数据集
val_ds = create_dataset(val_data_path)
visualize_model('transfer_best.ckpt', val_ds)
# visualize_model('best.ckpt', val_ds)