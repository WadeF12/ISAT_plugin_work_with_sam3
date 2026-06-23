## 安装

```bash
cd ISAT_plugin_sam3_text_prompt
pip install -e .
```

启动 ISAT，在右侧板会有 **SAM3TextPromptPlugin** ，可以点击最小化窗口单独拉出来

> **前置条件:** ISAT 中需已加载 SAM3 模型（SAM1/SAM2 不支持文本提示）。

---


## 各栏目详解

### Row 1 — 当前图片操作

**Predict Current** 对当前打开的这张图片执行 SAM3 文本提示预测。类别取自 Row 2 的 `Cat:` 输入框。 |
**Stop** 停止正在进行的批量任务（扫描/预测/删除）。 |

> `Predict Current` 不依赖 `From #` / `To #` 范围设置。

---

### Row 2 — 范围设置 & 类别输入

**From #** 起始编号（1-based）。例如填 `5` 表示从第 5 张图开始。 |
**To #** 结束编号（1-based，含）。例如填 `20` 表示到第 20 张图为止。 |

> 编号对应 ISAT 文件列表中的顺序。开图时自动同步上限。

#### Cat:（统一类别输入框）

用于 sam3 识别的类别词，**逗号分隔**多个类别。


#### Map:（标签映射，选填）

格式：`检测词:保存标签`，逗号分隔。

例：
```
Cat:  grass, green vegetation
Map:  grass:lawn, green vegetation:lawn
```

- **多对一支持:** 多个检测词可映射到同一个 label。
- **留空 = 不映射:** 检测词直接作为保存的 label。
- **对所有 SAM3 操作生效:** Predict Current / Predict Range / Resume Range。

---

### Row 3 — 范围操作按钮

所有按钮都受 `From #` / `To #` 范围限制。

 **Predict Range** 🔵  对范围内**所有图片**执行 SAM3 文本提示预测。新的 mask **追加**在已有标注之上，不覆盖旧标注。
 
 **Resume Range**  🟢  对范围内**尚无标注文件的图片**执行预测。已有 `.json` 标注的图片自动跳过。适合中断后继续。
 
 **Annotate Range**  🟠  将范围内每张图片**整个画面区域**标注为 `Cat:` 中指定的类别。已有标注会被**替换**。不调用 SAM3。
 
 **Delete Range**  🔴  **删除**范围内所有图片的 `.json` 标注文件。不可逆，二次确认。

---

### Row 4 — 小 Mask 清理

#### Threshold:

面积阈值百分比。mask 面积 ÷ 图片总像素 × 100% 低于此值的 mask 将被视为「小 mask」。

> 例如设为 `2.00%`，一张 1920×1080 的图上面积小于 41,472 像素的 mask 会被删除。

#### Cat:

类别过滤（选填）。只删除**特定类别**的小 mask。逗号分隔多个类别。

- 填 `person` → 只删除 person 类的小 mask
- 留空 → 删除**所有类别**的小 mask

#### Delete Small Masks 按钮

**Delete Small Masks**  🟣 异步扫描 → 弹出统计 → 确认后异步删除。全程不阻塞 UI，可随时 Stop 中断。 |

---


## 常见问题

### Qt 平台插件错误 (Linux)

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

```bash
# Ubuntu/Debian
sudo apt install libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0

# CentOS/RHEL
sudo yum install xcb-util-wm xcb-util-image xcb-util-keysyms xcb-util-renderutil
```

