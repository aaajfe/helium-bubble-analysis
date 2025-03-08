import cv2
import numpy as np
from sklearn.cluster import DBSCAN

class HeliumBubbleAnalyzer:
    def __init__(self):
        self.original_image = None
        self.preprocessed = None
        self.binary = None
        # 根据图像调整参数
        self.min_bubble_size = 3    # 降低最小尺寸，因为气泡普遍较小
        self.max_bubble_size = 100  # 根据图像调整最大尺寸
        
    def load_image(self, image_path):
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"无法加载图像: {image_path}")
            
    def preprocess_image(self):
        """优化的图像预处理"""
        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        # 使用小核的高斯模糊以保留小气泡的细节
        blurred = cv2.GaussianBlur(gray, (3, 3), 0.5)
        
        # 使用局部对比度增强
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
        enhanced = clahe.apply(blurred)
        
        # 应用局部均值滤波来减少背景噪声
        local_mean = cv2.blur(enhanced, (15, 15))
        normalized = cv2.subtract(enhanced, local_mean)
        normalized = cv2.add(normalized, 128)
        
        self.preprocessed = normalized
        return normalized
        
    def extract_features(self):
        """优化的特征提取"""
        if self.preprocessed is None:
            raise ValueError("请先调用 preprocess_image()")
            
        # 使用自适应阈值处理
        binary = cv2.adaptiveThreshold(
            self.preprocessed,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,  # 邻域大小
            -2   # 常数调整值
        )
        
        # 形态学操作
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
        # 开运算去除小噪点
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        self.binary = binary
        return binary
        
    def detect_bubbles(self):
        """优化的气泡检测"""
        if self.binary is None:
            raise ValueError("请先调用 extract_features()")
            
        # 使用watershed算法进行分水岭分割
        dist_transform = cv2.distanceTransform(self.binary, cv2.DIST_L2, 3)
        _, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # 寻找轮廓
        contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        valid_bubbles = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_bubble_size < area < self.max_bubble_size:
                # 计算圆度
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # 只保留较圆的气泡
                if circularity > 0.6:  # 圆度阈值
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    valid_bubbles.append({
                        'center': (int(x), int(y)),
                        'radius': int(radius),
                        'area': area,
                        'circularity': circularity
                    })
                    
        return valid_bubbles
        
    def analyze_features(self):
        """特征分析"""
        bubbles = self.detect_bubbles()
        
        if not bubbles:
            return {
                'total_bubbles': 0,
                'avg_radius': 0,
                'avg_area': 0,
                'clusters': 0
            }
            
        # 计算统计信息
        total_bubbles = len(bubbles)
        avg_radius = np.mean([b['radius'] for b in bubbles])
        avg_area = np.mean([b['area'] for b in bubbles])
        
        # 使用DBSCAN进行聚类分析
        centers = np.array([b['center'] for b in bubbles])
        clustering = DBSCAN(eps=30, min_samples=2).fit(centers)
        n_clusters = len(set(clustering.labels_)) - (1 if -1 in clustering.labels_ else 0)
        
        return {
            'total_bubbles': total_bubbles,
            'avg_radius': avg_radius,
            'avg_area': avg_area,
            'clusters': n_clusters
        }
        
    def visualize_results(self, output_path):
        """结果可视化"""
        result_image = self.original_image.copy()
        bubbles = self.detect_bubbles()
        
        # 绘制检测到的气泡
        for bubble in bubbles:
            cv2.circle(
                result_image,
                bubble['center'],
                int(bubble['radius']),
                (0, 255, 0),
                2
            )
            
        # 保存结果
        cv2.imwrite(output_path, result_image)
        return result_image
    
    # ... 其他方法 ... 