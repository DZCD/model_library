# build.py - 跨平台目录形式编译
import PyInstaller.__main__
import shutil
import os
import platform


def get_path_separator():
    return ';' if platform.system() == 'Windows' else ':'


def main():
    print(f"🔥 在 {platform.system()} 系统上编译...")

    # 删除原先的编译文件
    if os.path.exists('build'):
        shutil.rmtree('build')
    if os.path.exists('dist'):
        shutil.rmtree('dist')

    # 编译
    PyInstaller.__main__.run([
        '--onedir',
        '--name=ai-service',
        '--hidden-import=torch',
        '--hidden-import=ultralytics',
        '--hidden-import=fastapi',
        '--hidden-import=uvicorn',
        '--hidden-import=numpy',
        '--hidden-import=numpy._core',
        '--hidden-import=numpy._core._exceptions',

        'main.py'
    ])

    # 复制权重文件
    app_dir = os.path.join('dist', 'ai-service')
    weight_dist_dir = os.path.join(app_dir, 'weight')
    os.makedirs(weight_dist_dir, exist_ok=True)

    if os.path.exists('weight'):
        for file in os.listdir('weight'):
            if file.endswith(('.pt', '.pth', '.trt', '.onnx')):
                src = os.path.join('weight', file)
                dst = os.path.join(weight_dist_dir, file)
                shutil.copy2(src, dst)

    # 复制配置文件
    if os.path.exists('config.yaml'):
        dst_config = os.path.join(app_dir, 'config.yaml')
        shutil.copy2('config.yaml', dst_config)

    print("✅ 编译完成！")

    # 显示运行方式
    if platform.system() == 'Windows':
        print("🚀 运行: cd dist\\ai-service && ai-service.exe")
    else:
        print("🚀 运行: cd dist/ai-service && ./ai-service")


if __name__ == "__main__":
    main()