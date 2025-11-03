# build.py - 自动处理依赖版本
import subprocess
import sys
import shutil
import os
import platform

def install_system_deps():
    """安装系统依赖"""
    if platform.system() == "Linux":
        print("📦 检查系统依赖...")
        try:
            subprocess.run(["which", "patchelf"], check=True, capture_output=True)
            print("✅ patchelf已安装")
        except subprocess.CalledProcessError:
            print("⚠️  需要安装patchelf")
            print("请运行: sudo apt install patchelf")
            print("或者: sudo dnf install patchelf")
            return False
    return True

# 检查依赖
if not install_system_deps():
    sys.exit(1)

# 删除旧的编译文件
for item in ["main.dist", "main.build", "main.exe", "main.bin", "main"]:
    if os.path.exists(item):
        if os.path.isdir(item):
            shutil.rmtree(item)
        else:
            os.remove(item)
        print(f"🧹 删除: {item}")

# 编译
subprocess.run([
    sys.executable, "-m", "nuitka",
    "--onefile",
    "--standalone", 
    "--assume-yes-for-downloads",
    "--static-libpython=no",
    "--include-data-dir=model_library=model_library",
    "--include-package=torch",
    "--include-package=ultralytics", 
    "--include-package=fastapi",
    "--include-package=uvicorn",
    "main.py"
], check=True)

# 复制配置文件
if os.path.exists("config.yaml"):
    shutil.copy2("config.yaml", "config.yaml")
    print("✅ 复制配置文件")

# 复制权重文件目录
if os.path.exists("weight"):
    shutil.copytree("weight", "weight")
    print("✅ 复制权重文件")

print("✅ 编译完成！")