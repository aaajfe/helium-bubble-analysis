try:
    import helium_bubble_analysis as hba  # 先完整导入模块
    from helium_bubble_analysis import HeliumBubbleAnalyzer
    print("模块导入成功")
    print("模块位置:", hba.__file__)  # 使用模块别名
except ImportError as e:
    print("模块导入失败:", str(e))

import cv2
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macOS 的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def process_single_image(analyzer, image_path, output_dir):
    """处理单张图像"""
    # 加载图像
    analyzer.load_image(image_path)
    
    # 图像处理流程
    preprocessed = analyzer.preprocess_image()
    binary = analyzer.extract_features()
    bubbles = analyzer.detect_bubbles()
    stats = analyzer.analyze_features()
    
    # 准备输出路径
    image_name = Path(image_path).stem
    result_image_path = os.path.join(output_dir, f"{image_name}_result.jpg")
    
    # 可视化结果
    result = analyzer.visualize_results(result_image_path)
    
    # 保存处理过程的图像
    process_image_path = os.path.join(output_dir, f"{image_name}_process.jpg")
    
    # 创建处理过程的可视化
    plt.figure(figsize=(15, 5))
    
    plt.subplot(141)
    plt.imshow(cv2.cvtColor(analyzer.original_image, cv2.COLOR_BGR2RGB))
    plt.title("原始图像")
    
    plt.subplot(142)
    plt.imshow(preprocessed, cmap='gray')
    plt.title("预处理后")
    
    plt.subplot(143)
    plt.imshow(binary, cmap='gray')
    plt.title("二值化结果")
    
    plt.subplot(144)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title("检测结果")
    
    plt.tight_layout()
    plt.savefig(process_image_path)
    plt.close()
    
    return {
        'image_name': image_name,
        'total_bubbles': stats['total_bubbles'],
        'avg_radius': stats['avg_radius'],
        'avg_area': stats['avg_area'],
        'clusters': stats['clusters']
    }

def batch_process(input_dir, output_dir):
    """批量处理文件夹中的图像"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建分析器实例
    analyzer = HeliumBubbleAnalyzer()
    
    # 支持的图像格式
    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(image_extensions)]
    
    # 处理结果列表
    results = []
    
    # 处理每张图像
    for i, image_file in enumerate(image_files, 1):
        image_path = os.path.join(input_dir, image_file)
        print(f"正在处理图像 {i}/{len(image_files)}: {image_file}")
        
        try:
            result = process_single_image(analyzer, image_path, output_dir)
            results.append(result)
        except Exception as e:
            print(f"处理图像 {image_file} 时出错: {str(e)}")
    
    # 将结果保存为Excel文件
    if results:
        df = pd.DataFrame(results)
        excel_path = os.path.join(output_dir, 'analysis_results.xlsx')
        df.to_excel(excel_path, index=False)
        print(f"\n分析结果已保存到: {excel_path}")

def main():
    # 设置输入和输出目录
    input_dir = "input_images"  # 存放待处理图像的文件夹
    output_dir = "output_results"  # 存放处理结果的文件夹
    
    # 创建输入目录
    os.makedirs(input_dir, exist_ok=True)
    print(f"已创建输入目录: {input_dir}")
    print("请将需要处理的TEM图像放入此目录")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    print(f"已创建输出目录: {output_dir}")
    
    # 检查输入目录是否有图像文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print("\n注意：input_images 目录中还没有图像文件")
        print("请将要处理的图像文件放入该目录后再运行程序")
        return
        
    # 批量处理图像
    batch_process(input_dir, output_dir)

if __name__ == "__main__":
    main() 