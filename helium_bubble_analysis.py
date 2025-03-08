import cv2
import numpy as np
from skimage import exposure, measure, morphology
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

class BubbleDetectionNet(nn.Module):
    def __init__(self, pretrained=True):
        super(BubbleDetectionNet, self).__init__()
        # 使用预训练的ResNet18作为基础网络
        resnet = models.resnet18(pretrained=pretrained)
        
        # 提取特征提取层
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # 添加自定义层
        self.conv1x1 = nn.Conv2d(512, 256, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_final = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        # 特征提取
        features = self.features(x)
        
        # 特征融合和上采样
        x = self.conv1x1(features)
        x = self.upsample(x)
        x = F.relu(x)
        
        # 最终预测
        out = self.conv_final(x)
        return torch.sigmoid(out)

class HeliumBubbleAnalyzer:
    def __init__(self, model_path=None):
        self.original_image = None
        self.processed_image = None
        self.binary_image = None
        self.bubbles = []
        
        # 初始化深度学习模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BubbleDetectionNet().to(self.device)
        if model_path:
            self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # 图像预处理转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])

    def load_image(self, image_path):
        """加载TEM图像"""
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError("无法加载图像")
        return self.original_image
    
    def preprocess_image(self):
        """结合传统方法和深度学习的图像预处理"""
        # 传统预处理
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        background = cv2.medianBlur(gray, 101)
        foreground = cv2.subtract(gray, background)
        foreground = cv2.normalize(foreground, None, 0, 255, cv2.NORM_MINMAX)
        denoised = cv2.fastNlMeansDenoising(foreground, None, 5, 7, 21)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        self.processed_image = clahe.apply(denoised)
        
        return self.processed_image

    def deep_feature_extraction(self):
        """使用深度学习模型提取特征"""
        # 准备输入
        img = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        # 使用模型提取特征
        with torch.no_grad():
            pred = self.model(img_tensor)
        
        # 转换预测结果
        pred_mask = (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
        self.binary_image = cv2.resize(pred_mask, 
                                     (self.original_image.shape[1], 
                                      self.original_image.shape[0]))
        return self.binary_image

    def extract_features(self):
        """特征提取（结合传统方法和深度学习）"""
        # 获取深度学习的预测结果
        dl_binary = self.deep_feature_extraction()
        
        # 获取传统方法的结果
        traditional_binary = cv2.adaptiveThreshold(
            self.processed_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )
        
        # 特征融合
        self.binary_image = cv2.bitwise_and(dl_binary, traditional_binary)
        
        # 形态学处理
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        self.binary_image = cv2.morphologyEx(self.binary_image, 
                                           cv2.MORPH_OPEN, kernel)
        
        return self.binary_image
    
    def detect_bubbles(self):
        """改进的气泡检测"""
        # 连通域分析
        labels = measure.label(self.binary_image)
        regions = measure.regionprops(labels)
        
        # 提取气泡特征
        self.bubbles = []
        for region in regions:
            perimeter = region.perimeter
            area = region.area
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            if (area >= 10 and area <= 3000 and circularity > 0.3):
                y, x = region.centroid
                radius = np.sqrt(area / np.pi)
                self.bubbles.append({
                    'center': (int(x), int(y)),
                    'radius': radius,
                    'area': area,
                    'circularity': circularity
                })
        
        return self.bubbles
    
    def analyze_features(self):
        """特征分析与统计"""
        if not self.bubbles:
            return None
            
        # 提取气泡特征数据
        centers = np.array([b['center'] for b in self.bubbles])
        
        # DBSCAN聚类分析
        clustering = DBSCAN(eps=50, min_samples=3).fit(centers)
        
        # 统计分析
        stats = {
            'total_bubbles': len(self.bubbles),
            'avg_radius': np.mean([b['radius'] for b in self.bubbles]),
            'avg_area': np.mean([b['area'] for b in self.bubbles]),
            'clusters': len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        }
        
        return stats
    
    def visualize_results(self, output_path=None):
        """结果可视化"""
        result_image = self.original_image.copy()
        
        # 绘制检测到的气泡
        for bubble in self.bubbles:
            cv2.circle(
                result_image,
                bubble['center'],
                int(bubble['radius']),
                (0, 255, 0),
                2
            )
            
        if output_path:
            cv2.imwrite(output_path, result_image)
            
        return result_image 