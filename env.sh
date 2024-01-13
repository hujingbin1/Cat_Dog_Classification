conda create -n py39mindspore python=3.9
conda activate py39mindspore
pip install https://ms-release.obs.cn-north-4.myhuaweicloud.com/2.2.10/MindSpore/unified/x86_64/mindspore-2.2.10-cp39-cp39-linux_x86_64.whl --trusted-host ms-release.obs.cn-north-4.myhuaweicloud.com -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pyyaml
pip install pandas
pip install opencv_python
pip install ruamel_yaml
pip install matplotlib
# 安装onnx
pip install onnx 
# 安装onnx runtime
pip install onnxruntime # 使用CPU进行推理
# pip install onnxruntime-gpu # 使用GPU进行推理
pip install PyQt5