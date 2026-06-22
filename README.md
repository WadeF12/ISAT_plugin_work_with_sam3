### 安装插件

```bash
cd ISAT_plugin_sam3_text_prompt
pip install -e .
```

### 验证安装

启动 ISAT，在Plugins 中应该能看到 SAM3TextPromptPlugin

### 常见问题

#### Qt 平台插件错误 (Linux)

如果在 Linux 上遇到以下错误:

```
Could not load the Qt platform plugin "xcb" in ".../cv2/qt/plugins"
This application failed to start because no Qt platform plugin could be initialized.
```

**原因:** `opencv-python` (完整版) 自带的 `cv2/qt/plugins/` 目录中的 Qt 插件与系统 PyQt5 冲突。

**解决方法 (任选其一):**

1. **替换为 headless 版本 (推荐):**
   ```bash
   pip uninstall opencv-python -y
   pip install opencv-python-headless
   ```

2. **临时删除 opencv 的 Qt 插件目录:**
   ```bash
   rm -rf $(python -c "import cv2, os; print(os.path.join(os.path.dirname(cv2.__file__), 'qt'))")
   ```

3. **设置环境变量指向系统 Qt 插件:**
   ```bash
   export QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms
   ```

#### 缺少 xcb 库

如果错误提示 `libxcb.so` 相关:

```bash
# Ubuntu/Debian
sudo apt install libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0

# CentOS/RHEL
sudo yum install xcb-util-wm xcb-util-image xcb-util-keysyms xcb-util-renderutil
```

