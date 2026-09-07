## 安装

```bash
conda activate [ISAT环境]
cd [解压后的文件夹：ISAT_plugin_sam3_text_prompt]
pip install -e .
```

启动 ISAT，在右侧面板会出现 **SAM3 Text Prompt Batch** 插件窗口，可拖出独立显示。

> 使用推理功能前，需在 ISAT 中加载 SAM3 模型（SAM1/SAM2 不支持文本提示）。

---

## 各栏目功能

### Row 1 — 当前图片 & 停止

- **Predict Current** — 对当前图片执行 SAM3 文本提示预测，类别取自 Row 2 的 `Cat:` 输入框，不依赖 `From #` / `To #`
- **Stop** — 停止正在进行的批量任务（扫描/预测/删除等）

---

### Row 2 — 范围设置 & 类别 & 映射

- **From #** — 起始图片编号（1-based）
- **To #** — 结束图片编号（1-based，含），右侧 `of N` 显示当前图片总数
- **Cat:** — SAM3 识别的类别词，**逗号分隔**多个类别
- **Map:** — 标签映射（选填），格式：`检测词:保存标签`，逗号分隔

#### Map: 映射说明

```
Cat:  grass, lawn, soil, fence, person
Map:  grass:lawn, person:__background__
```

- **多对一支持：** 多个检测词可映射到同一个标签
- **留空 = 不映射：** 检测词直接作为保存的类别名
- `__background__` 是特殊值，表示忽略该类别（不保存）
- 对 **Predict Current / Predict Range / Resume Range / Remap / Erase Overlap** 均生效

> 注：Cat 中的词应符合大模型的理解。业务场景中的特殊类别名（如 tallgrass, hardroad）可能与大模型理解不一致，建议使用通用词检测并结合 Map 映射为实际类别名

---

### Row 3 — 范围操作按钮

所有按钮均受 `From #` – `To #` 范围限制。

- 🟦 **Predict Range** — 对范围内所有图片执行 SAM3 预测，**追加**到已有标注之上，不覆盖旧标注
- 🟩 **Resume Range** — 对范围内**尚无标注文件**的图片执行预测，已有标注的自动跳过
- 🟧 **Annotate Range** — 将范围内每张图片**整个画面**标注为 `Cat:` 中的类别（全图多边形），**替换**已有标注，不调用 SAM3
- 🟥 **Delete Range** — **删除**范围内所有图片的 `.json` 标注文件（不可逆）

---

### Row 4 — 删除小 Mask

- **Threshold:** — 面积阈值百分比。mask 面积 ÷ 图片总像素 × 100% 低于此值即视为「小 mask」
- **Cat:** — 类别过滤（选填），逗号分隔多个类别；留空 = 删除所有类别的小 mask
- 🟪 **Delete Small Masks** — 先扫描统计 → 弹窗确认 → 确认后删除

> 例如 `Threshold: 2.00%`，1920×1080 的图上面积 < 41,472 像素的 mask 会被删除
> 此部分也可以用来删除指定类别的 Mask，只需将删除阈值设为 100%，并填写待删除类别即可

---

### Row 5 — 合并同类 Mask

- **Merge Cat:** — 要合并的类别名
- 🟩 **Merge Same Category** — 将范围内每张图片中指定类别的所有 mask **取并集**合并为一个 mask，即合并所有有重叠部分的指定类别 mask

---

### Row 6 — 小 Mask 重映射

将小面积的 mask 从一个类别重标记为另一个类别

- **Remap:** 第 1 输入框 — 源类别（source cat）
- **&lt; %** — 面积阈值：小于此百分比的 mask 会被重映射
- **→** 第 2 输入框 — 目标类别（target cat），支持 Map 映射
- 🟪 **Remap** — 执行重映射

---

### Row 7 — 重叠区域擦除

当两个类别的 mask 存在重叠时，移除 "Erase Cat" 在重叠区域的部分，保留 "Keep Cat"。

- **Erase Cat:** — 被裁剪的类别（重叠部分会被移除）
- **Keep Cat:** — 保留优先级的类别（重叠部分保持不变）
- 🟠 **Erase Overlap** — 执行擦除

**行为细节：**
- 重叠区域 → Erase Cat 被挖掉，Keep Cat 保留，**共享边界精确无缝**
- 完全被覆盖的 Erase Cat mask → 整体删除
- 一个 mask 被切割成多块 → 自动创建多个独立 mask
- 两个类别相同时会弹窗报错
- 支持 Map 映射

---

### Row 8 – Row 11 — 比例区域标注

通过拖拽四边形角点为范围内图片追加一个区域 mask。

- **Row 8** — **Region Label:** + **Layer:** 区域类别名 + 层级（最底层/最上层），支持 Map 映射
- **Row 9** — **W:** **H:** + **Update Ratio** 参考图像宽高，用于设定下方四边形比例
- **Row 10** — 四边形拖拽组件：拖拽 4 个青色角点定义区域，比例与 W:H 一致
- **Row 11** — 🟪 **Apply Region** 将四边形区域作为 mask **追加**到范围内所有图片的标注中

---

## 常见问题

### Qt 平台插件错误

如果在 Linux 上遇到以下错误:

```
Could not load the Qt platform plugin "xcb" in ".../cv2/qt/plugins"
This application failed to start because no Qt platform plugin could be initialized.
```

**原因:** `opencv-python` (完整版) 自带的 `cv2/qt/plugins/` 目录中的 Qt 插件与系统 PyQt5 冲突。

**解决方法: 替换为 headless 版本**
   ```bash
   pip uninstall opencv-python -y
   pip install opencv-python-headless
   ```

#### 缺少 xcb 库

```bash
# Ubuntu/Debian
sudo apt install libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0

# CentOS/RHEL
sudo yum install xcb-util-wm xcb-util-image xcb-util-keysyms xcb-util-renderutil
```

