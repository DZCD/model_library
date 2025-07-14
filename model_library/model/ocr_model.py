"""用于车牌识别时，识别车牌上的字符"""
import torch.nn as nn
import torch
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")
color = ['黑色', '蓝色', '绿色', '白色', '黄色']
plateName = r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
mean_value, std_value = (0.588, 0.193)


class myNet_ocr_color(nn.Module):
    def __init__(self, cfg=None, num_classes=78, export=False, color_num=None):
        super(myNet_ocr_color, self).__init__()
        if cfg is None:
            cfg = [32, 32, 64, 64, 'M', 128, 128, 'M', 196, 196, 'M', 256, 256]
            # cfg =[32,32,'M',64,64,'M',128,128,'M',256,256]
        self.feature = self.make_layers(cfg, True)
        self.export = export
        self.color_num = color_num
        self.conv_out_num = 12  # 颜色第一个卷积层输出通道12
        if self.color_num:
            self.conv1 = nn.Conv2d(cfg[-1], self.conv_out_num, kernel_size=3, stride=2)
            self.bn1 = nn.BatchNorm2d(self.conv_out_num)
            self.relu1 = nn.ReLU(inplace=True)
            self.gap = nn.AdaptiveAvgPool2d(output_size=1)
            self.color_classifier = nn.Conv2d(self.conv_out_num, self.color_num, kernel_size=1, stride=1)
            self.color_bn = nn.BatchNorm2d(self.color_num)
            self.flatten = nn.Flatten()
        self.loc = nn.MaxPool2d((5, 2), (1, 1), (0, 1), ceil_mode=False)
        self.newCnn = nn.Conv2d(cfg[-1], num_classes, 1, 1)
        # self.newBn=nn.BatchNorm2d(num_classes)

    def make_layers(self, cfg, batch_norm=False):
        layers = []
        in_channels = 3
        for i in range(len(cfg)):
            if i == 0:
                conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=5, stride=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = cfg[i]
            else:
                if cfg[i] == 'M':
                    layers += [nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)]
                else:
                    conv2d = nn.Conv2d(in_channels, cfg[i], kernel_size=3, padding=(1, 1), stride=1)
                    if batch_norm:
                        layers += [conv2d, nn.BatchNorm2d(cfg[i]), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True)]
                    in_channels = cfg[i]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.feature(x)
        if self.color_num:
            x_color = self.conv1(x)
            x_color = self.bn1(x_color)
            x_color = self.relu1(x_color)
            x_color = self.color_classifier(x_color)
            x_color = self.color_bn(x_color)
            x_color = self.gap(x_color)
            x_color = self.flatten(x_color)
        x = self.loc(x)
        x = self.newCnn(x)

        if self.export:
            conv = x.squeeze(2)  # b *512 * width
            conv = conv.transpose(2, 1)  # [w, b, c]
            if self.color_num:
                return conv, x_color
            return conv
        else:
            b, c, h, w = x.size()
            assert h == 1, "the height of conv must be 1"
            conv = x.squeeze(2)  # b *512 * width
            conv = conv.permute(2, 0, 1)  # [w, b, c]
            output = F.log_softmax(conv, dim=2)
            if self.color_num:
                return output, x_color
            return output


def init_model(model_path, is_color=False):
    """初始化字符串检测模型"""
    check_point = torch.load(model_path, map_location=device)
    model_state = check_point['state_dict']
    cfg = check_point['cfg']
    color_classes = 0
    if is_color:
        color_classes = 5  # 颜色类别数
    model = myNet_ocr_color(num_classes=len(plateName), export=True, cfg=cfg, color_num=color_classes)

    model.load_state_dict(model_state, strict=False)
    model.to(device)
    model.eval()
    return model


def image_processing(img):
    img = cv2.resize(img, (168, 48))
    img = np.reshape(img, (48, 168, 3))

    # normalize
    img = img.astype(np.float32)
    img = (img / 255. - mean_value) / std_value
    img = img.transpose([2, 0, 1])
    img = torch.from_numpy(img)

    img = img.to(device)
    img = img.view(1, *img.size())
    return img


def decodePlate(preds):
    pre = 0
    newPreds = []
    index = []
    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != pre:
            newPreds.append(preds[i])
            index.append(i)
        pre = preds[i]
    return newPreds, index


def get_plate_result(img, device, model, is_color=True):
    input = image_processing(img, device)
    if is_color:  # 是否识别颜色
        preds, color_preds = model(input)
        color_preds = torch.softmax(color_preds, dim=-1)
        color_conf, color_index = torch.max(color_preds, dim=-1)
        color_conf = color_conf.item()
    else:
        preds = model(input)
    preds = torch.softmax(preds, dim=-1)
    prob, index = preds.max(dim=-1)
    index = index.view(-1).detach().cpu().numpy()
    prob = prob.view(-1).detach().cpu().numpy()

    # preds=preds.view(-1).detach().cpu().numpy()
    newPreds, new_index = decodePlate(index)
    prob = prob[new_index]
    plate = ""
    for i in newPreds:
        plate += plateName[i]
    # if not (plate[0] in plateName[1:44] ):
    #     return ""
    if is_color:
        return plate, prob, color[color_index], color_conf  # 返回车牌号以及每个字符的概率,以及颜色，和颜色的概率
    else:
        return plate, prob


def get_split_merge(img):
    """处理双层的推理结果"""
    h, w, c = img.shape
    img_upper = img[0:int(5 / 12 * h), :]
    img_lower = img[int(1 / 3 * h):, :]
    img_upper = cv2.resize(img_upper, (img_lower.shape[1], img_lower.shape[0]))
    new_img = np.hstack((img_upper, img_lower))
    return new_img


def draw_license_plates(image_np, plate_boxes, plate_texts, font_path="simsun.ttc", RGB=True):
    """
    在图像上绘制车牌框和车牌字符串（支持中文）。

    参数:
        image_np (np.ndarray): 原图的 ndarray 数组，形状为 (H, W, C)，通道顺序取决于RGB参数。
        plate_boxes (list): 车牌位置框的列表，每个框的格式为 (x1, y1, x2, y2)。
        plate_texts (list): 车牌字符串的列表，与 plate_boxes 一一对应。
        font_path (str): 字体文件路径（支持中文的字体文件，如 simsun.ttc）。
        RGB (bool): True表示输入为RGB格式，False表示输入为BGR格式。

    返回:
        np.ndarray: 绘制后的图像，格式与输入相同。
    """
    # 定义RGB格式下的颜色值
    colors_rgb = {
        'yellow': (255, 255, 0),  # 黄色
        'red': (255, 0, 0),  # 红色
        'white': (255, 255, 255),  # 白色
        'black': (0, 0, 0)  # 黑色
    }

    # 如果输入是BGR格式，需要先转换为RGB供PIL使用
    working_image = image_np if RGB else cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)

    # 将 ndarray 转换为 PIL 图像
    image_pil = Image.fromarray(working_image)

    # 创建绘图对象
    draw = ImageDraw.Draw(image_pil)

    # 加载字体
    try:
        font = ImageFont.truetype(font_path, 40)  # 字体大小为40
    except IOError:
        raise ValueError(f"字体文件 {font_path} 未找到，请提供支持中文的字体文件路径。")

    # 遍历每个车牌框和字符串
    for box, text in zip(plate_boxes, plate_texts):
        x1, y1, x2, y2 = box

        # 绘制黄色车牌框
        draw.rectangle([x1, y1, x2, y2], outline=colors_rgb['yellow'], width=2)

        # 计算文字位置（在框的上方居中）
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        text_width = right - left  # 计算文本宽度
        text_height = bottom - top  # 计算文本高度
        text_x = x1 + (x2 - x1 - text_width) // 2  # 水平居中
        text_y = y1 - text_height - 10  # 在框的上方，留出10像素的间距

        # 绘制黄色背景框
        padding = 10  # 文字周围的内边距
        bg_x1 = text_x - padding
        bg_y1 = text_y - padding
        bg_x2 = text_x + text_width + padding
        bg_y2 = text_y + text_height + padding
        draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2], fill=colors_rgb['yellow'], outline=colors_rgb['black'], width=2)

        # 绘制红色文字（带白色描边）
        draw.text((text_x, text_y), text, fill=colors_rgb['red'], font=font, stroke_width=1,
                  stroke_fill=colors_rgb['white'])

    # 将PIL图像转回numpy数组
    result = np.array(image_pil)

    # 如果输入是BGR格式，需要将结果转换回BGR
    if not RGB:
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    return result