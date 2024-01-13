
from mindspore import export, load_checkpoint, load_param_into_net
from mindspore import Tensor
import numpy as np
from resnet import resnet50

net = resnet50(2)
# 将模型参数存入parameter的字典中
param_dict = load_checkpoint("best.ckpt")

# 将参数加载到网络中
load_param_into_net(net, param_dict)
input = np.random.uniform(0.0, 1.0, size=[1, 3, 224, 224]).astype(np.float32)
#导出模型，可导出ONNX、MINDIR、AIR格式
export(net, Tensor(input), file_name='resnet50_best', file_format='ONNX')
# export(net, Tensor(input), file_name='resnet50_best', file_format='MINDIR')