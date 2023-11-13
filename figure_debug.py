from PIL import Image

# 你的图片文件名列表
image_files = [
    "figure/confusion_matrix_resnet18.jpg",
    "figure/confusion_matrix_without_resnet18.jpg",
    "figure/confusion_matrix_resnet34.jpg",
    "figure/confusion_matrix_without_resnet34.jpg",
    "figure/confusion_matrix_resnet50.jpg",
    "figure/confusion_matrix_without_resnet50.jpg",
    "figure/confusion_matrix_resnet101.jpg",
    "figure/confusion_matrix_without_resnet101.jpg",
    "figure/confusion_matrix_resnet152.jpg",
    "figure/confusion_matrix_without_resnet152.jpg"
]

# 打开所有图片并获取它们的宽度和高度
images = [Image.open(file) for file in image_files]
width, height = images[0].size

# 创建一个新的大图像，大小为5行2列
result_image = Image.new('RGB', (2 * width, 5 * height))

# 将图片拼接到大图像中
for i in range(5):
    for j in range(2):
        index = i * 2 + j
        if index < len(images):
            result_image.paste(images[index], (j * width, i * height))

# 保存拼接后的图像
result_image.save("figure/confusion_matrix.jpg")