import os
import numpy as np
import cv2
import mindspore.nn as nn
from mindspore import dtype as mstype
import mindspore.dataset.vision.c_transforms as CV
from mindspore import Model, Tensor, context, load_checkpoint, load_param_into_net
from resnet import resnet50

#设置使用设备，CPU/GPU/Ascend
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


def ms_normalize(image):
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    resize = CV.Resize([224, 224])
    normalization = CV.Normalize(mean=mean, std=std)
    hwc2chw = CV.HWC2CHW()
    image = resize(image)
    image = normalization(image)
    image = hwc2chw(image)
    return image


def normalize(image):
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    image = cv2.resize(image, [224, 224], cv2.INTER_LINEAR)
    image = image / 1.0
    image = (image[:, :] - mean) / std
    image = image[:, :, ::-1].transpose((2, 0, 1))  # HWC-->CHW
    return image


def pre_deal(data_path):
    image = cv2.imread(data_path)
    norm_img = normalize(image)
    #norm_img = ms_normalize(image)
    images = [norm_img]
    images = Tensor(images, mstype.float32)
    return images


def infer(ckpt_path, data_path, num_class):
    image = pre_deal(data_path)
    net = resnet50(num_class)
    param_dict = load_checkpoint(ckpt_path)
    load_param_into_net(net, param_dict)
    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    model = Model(net, loss, metrics={"Accuracy": nn.Accuracy()})
    output = model.predict(image)
    print(output)
    pred = np.argmax(output.asnumpy(), axis=1)
    return pred


if __name__ == '__main__':
    # ckpt_path = 'transfer_best.ckpt'
    ckpt_path = 'best.ckpt'
    data_path = './dataset/test/Cat'
    class_name = {0: 'Cat', 1: 'Dog'}
    for path in os.listdir(os.path.join(data_path)):
        path = os.path.join(data_path) + '/' + path
        print(path)
        result = infer(ckpt_path, path, 2)
        print(class_name[result[0]])