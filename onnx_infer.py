# # import onnx

# # try:
# #     # 加载模型文件
# #     onnx_model = onnx.load("resnet50_best.onnx")
    
# #     # 检查模型的有效性
# #     onnx.checker.check_model(onnx_model)
# # except onnx.checker.ValidationError as e:
# #     # 如果模型无效，将会抛出异常并打印错误信息
# #     print("The model is invalid: %s" % e)
# # else:
# #     # 如果模型有效，将会输出“The model is valid!”
# #     print("The model is valid!")



# from PIL import Image
# import numpy as np
# import onnxruntime as ort

# def preprocess_image(image_path):
#     image_size = [224, 224]
#     mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
#     std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

#     # 加载图像
#     img = Image.open(image_path)

#     # 调整大小
#     img = img.resize(image_size)

#     # 转换为numpy数组
#     img = np.array(img).astype(np.float32)

#     # 归一化
#     img = (img - mean) / std

#     # 确保在最后返回 float32 类型的数据
#     img = img.astype(np.float32)

#     # 改变通道顺序：从 HWC 到 CHW
#     img = np.transpose(img, (2, 0, 1))

#     return img



# # 加载 ONNX 模型
# ort_session = ort.InferenceSession("resnet50_best.onnx")

# # 图像预处理
# input_img = preprocess_image('./dataset/test/Cat/0.jpg')

# # 确保数据类型为 float32
# if input_img.dtype != np.float32:
#     input_img = input_img.astype(np.float32)

# # 准备模型输入
# ort_inputs = {ort_session.get_inputs()[0].name: input_img[np.newaxis, ...]}

# # 执行推理
# ort_outputs = ort_session.run(None, ort_inputs)[0]

# # 获取最可能的类别索引
# predicted_class_index = np.argmax(ort_outputs)

# print("Predicted class index:", predicted_class_index)

# if predicted_class_index==0:
#     print("喵喵，是小喵喵哦！")
# else:
#     print("汪汪,是小狗狗哦！")





from PIL import Image
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QFileDialog
from PyQt5.QtGui import QPixmap
import numpy as np
import onnxruntime as ort

def preprocess_image(image_path):
    image_size = [224, 224]
    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    # 加载图像
    img = Image.open(image_path)

    # 调整大小
    img = img.resize(image_size)

    # 转换为numpy数组
    img = np.array(img).astype(np.float32)

    # 归一化
    img = (img - mean) / std

    # 确保在最后返回 float32 类型的数据
    img = img.astype(np.float32)

    # 改变通道顺序：从 HWC 到 CHW
    img = np.transpose(img, (2, 0, 1))

    return img


# ONNX 模型推理函数
def infer_image(onnx_session, image_path):
    # 使用之前定义的图像预处理函数
    input_img = preprocess_image(image_path)

    # 确保数据类型为 float32
    if input_img.dtype != np.float32:
        input_img = input_img.astype(np.float32)

    # 准备模型输入并执行推理
    ort_inputs = {onnx_session.get_inputs()[0].name: input_img[np.newaxis, ...]}
    ort_outputs = onnx_session.run(None, ort_inputs)[0]

    return ort_outputs

# PyQt 窗口类
class ImageInferApp(QWidget):
    def __init__(self):
        super().__init__()

        # 初始化 ONNX 运行时会话
        self.ort_session = ort.InferenceSession("resnet50_best.onnx")

        # 设置窗口
        self.initUI()

    def initUI(self):
        # 设置布局
        layout = QVBoxLayout()

        # 上传图片按钮
        self.btn_upload = QPushButton('上传图片', self)
        self.btn_upload.clicked.connect(self.uploadImage)
        layout.addWidget(self.btn_upload)

        # 显示图片的标签
        self.label_image = QLabel(self)
        layout.addWidget(self.label_image)

        # 显示结果的标签
        self.label_result = QLabel('结果将在这里显示', self)
        layout.addWidget(self.label_result)

        # 设置窗口布局
        self.setLayout(layout)
        self.setWindowTitle('图片分类推理')
        self.setGeometry(300, 300, 350, 300)

    def uploadImage(self):
        # 打开文件选择对话框
        fname, _ = QFileDialog.getOpenFileName(self, '选择图片', '/home', 'Image files (*.jpg *.png)')
        if fname:
            # 显示选中的图片
            pixmap = QPixmap(fname)
            self.label_image.setPixmap(pixmap.scaled(224, 224))

            # 执行 ONNX 推理
            result = infer_image(self.ort_session, fname)

            # 处理并显示结果
            predicted_class_index = np.argmax(result)
            self.label_result.setText(f'预测类别索引: {predicted_class_index}')
            if predicted_class_index==0:
                self.label_result.setText("喵喵，是小喵喵哦！")
            else:
                self.label_result.setText("汪汪,是小狗狗哦！")

# 运行程序
if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageInferApp()
    ex.show()
    sys.exit(app.exec_())
