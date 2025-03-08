# test.py
import subprocess
import sys

def check_and_install(package_name, import_name=None):
    """检查并安装包"""
    if import_name is None:
        import_name = package_name
        
    try:
        __import__(import_name)
        print(f"{package_name} 已安装成功")
    except ImportError:
        print(f"正在安装 {package_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"{package_name} 安装完成")

def main():
    # 必需的包列表
    packages = [
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("matplotlib", "matplotlib"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("scikit-image", "skimage"),
        ("openpyxl", "openpyxl")
    ]
    
    print("开始检查并安装必需的库...\n")
    
    for package, import_name in packages:
        try:
            check_and_install(package, import_name)
        except Exception as e:
            print(f"安装 {package} 时出错: {str(e)}")
            
    print("\n检查特定版本的包...")
    
    # 指定版本的包
    specific_versions = [
        "opencv-python==4.11.0.86",
        "numpy==2.0.2",
        "scikit-image==0.24.0",
        "scikit-learn==1.6.1",
        "matplotlib==3.9.4",
        "pandas==2.2.3"
    ]
    
    for package in specific_versions:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} 安装成功")
        except Exception as e:
            print(f"安装 {package} 时出错: {str(e)}")
    
    print("\n所有依赖库检查完成！")
    print("\n建议：安装完成后重启 Python 环境以确保所有库都能正常工作。")

if __name__ == "__main__":
    main()