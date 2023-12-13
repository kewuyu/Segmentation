from model import UNet  # 确保已正确导入 UNet 和其他必要的类
import torch
import cv2
import numpy as np
from torchvision import transforms
# 初始化模型并移动到设备
mode = UNet(3, 2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型权重
mode.load_state_dict(torch.load('best.pt', map_location=device))
mode.to(device)


# 图像处理函数
def process(image_path):
    # 使用 OpenCV 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to find or open the file {image_path}")

    # 调整图像大小
    image = cv2.resize(image, (224, 224))

    # BGR 转 RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 转换为 PyTorch 张量
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image_rgb).unsqueeze(0)
    return image_tensor, image


# 推理函数
# 推理函数
def infer(image_path,model = mode):
    data, original_image = process(image_path)
    model.eval()
    with torch.no_grad():
        output = model(data.to(device))
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()

    # 应用形态学开运算来消除小区域
    kernel = np.ones((6, 6), np.uint8)  # 可以调整核的大小
    pred_opened = cv2.morphologyEx(pred.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # 显示原图像
    #cv2.imshow('Original Image', original_image)

    # 创建一个预测结果的遮罩
    mask = np.zeros(original_image.shape, dtype=np.uint8)
    mask[pred_opened == 1] = [255, 0, 0]  # 红色遮罩

    overlay_image = cv2.addWeighted(original_image, 1, mask, 0.5, 0)
    # cv2.imshow('Original Image with Segmentation Overlay', overlay_image)
    #
    # cv2.waitKey(0)  # 等待用户按键，然后关闭窗口
    # cv2.destroyAllWindows()

    return overlay_image, original_image

