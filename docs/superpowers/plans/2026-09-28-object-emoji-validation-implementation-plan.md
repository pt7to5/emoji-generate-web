# 物品 Emoji 本地 MVP 实施计划（2～3 天）

## 1. MVP 目标

在 2～3 天内完成一个本地运行、可演示和可测试的 HTML 原型：

```text
首页 → 上传图片 → 自动展示多个候选主体 → 点击或框选目标主体
→ 生成透明底 3D Emoji → 下载透明 Emoji
→ 可选移除原主体并回贴 Emoji → 下载结果图
```

MVP 只验证三件事：用户能否选中目标物品；生成结果能否保留姿态、颜色和关键特征；Emoji 回贴原图后是否基本自然。

本阶段不建设生产级稳定性、商业化、高并发、登录、付费、长期存储和复杂评测系统。

## 2. 已确认约束

- 使用第三方 API，并允许测试图片发送给第三方服务。
- 用户提供 10～20 张测试图片。
- 首轮只在本地运行。
- 目标风格为 iOS 系统 Emoji 所代表的圆润 3D 贴纸审美，但不复制 Apple 的具体 Emoji 资产。
- 目标时长为 2～3 天，以完成可用 MVP 为第一目标。

## 3. MVP 技术路线

### 3.1 技术栈

- 前端：React、TypeScript、Vite。
- 后端：Python、FastAPI。
- 图像处理：Pillow、OpenCV、NumPy。
- 主体分割：fal.ai SAM 2 API。
- Emoji 生成与背景修复：OpenAI Image API。
- 本地存储：项目内临时目录。
- 测试：Pytest、Vitest，以及少量 Playwright 端到端测试。

### 3.2 API 选择依据

- fal.ai SAM 2 自动分割接口可以输出多个独立蒙版；提示式接口支持点和边界框提示，适合点击主体和重新框选。
- OpenAI Image API 支持参考图编辑、蒙版编辑和透明背景 PNG 输出。
- 两类能力都通过后端代理调用，API Key 不进入浏览器。

### 3.3 简化后的处理链路

```text
上传原图
→ fal.ai SAM 2 自动分割
→ 前端展示面积和稳定性较高的前 5 个蒙版
→ 用户点击蒙版或框选后调用提示式分割
→ 根据蒙版提取透明底主体和紧凑裁切图
→ OpenAI Image API 参考主体生成透明底 3D Emoji
→ 本地清理 Alpha 边缘并恢复尺寸、位置
→ OpenAI Image API 使用原主体蒙版修复背景
→ 本地将 Emoji 合成回原图
```

MVP 不实现独立目标检测和物品类别命名。候选主体以编号展示，减少接入第三个模型的时间和失败点。

## 4. 风格规范

- 圆润、友好、精致的 3D 贴纸质感。
- 主体轮廓简化，但保持原姿态、朝向和长宽比例。
- 保留主色、显著图案以及至少两个关键结构特征。
- 材质柔和，使用统一方向的高光、环境遮蔽和轻微内阴影。
- 使用少量大色块，降低照片纹理和细碎噪点。
- 主体完整、居中、透明背景，无底座、文字、边框或额外物体。
- 不复制 Apple 的现有 Emoji 造型、具体角色或商标元素。

### 4.1 MVP 生成提示模板

```text
Transform the isolated object in the reference image into an original,
polished 3D emoji sticker. Preserve the object's exact pose, viewing
angle, silhouette proportions, primary colors, distinctive pattern,
and essential structural features. Simplify photographic texture into
a small number of clean rounded color regions. Use soft studio lighting,
gentle inner shading, subtle ambient occlusion, smooth highlights, and
a friendly premium mobile-emoji aesthetic. Keep the complete object
centered on a fully transparent background. Do not add text, a platform,
a border, scenery, extra objects, faces, limbs, or features not present
in the reference.
```

后端可以附加用户预先标注的关键特征，例如“保留蓝色灯罩、米色球形底座和正面视角”。

## 5. 2～3 天任务安排

### 第 1 天：项目骨架、上传与主体选择

Codex 自动完成：

1. 创建 React 前端和 FastAPI 后端。
2. 创建首页和图像编辑页。
3. 实现图片上传、预览、基础校验、方向修正和 EXIF 清理。
4. 接入 fal.ai SAM 2 自动分割。
5. 筛选并展示最多 5 个候选蒙版。
6. 实现点击选择、候选切换和高亮轮廓。
7. 实现框选后重新调用提示式分割。
8. 实现透明底主体预览。

当天验收：至少 3 张测试图可以上传、选择目标主体并看到透明底预览；选择错误时可以通过框选重新分割。

### 第 2 天：透明 Emoji 生成

Codex 自动完成：

1. 接入 OpenAI Image API。
2. 将透明底主体作为参考图发送给图像编辑接口。
3. 固化 3D Emoji 提示模板和参数。
4. 请求透明 PNG 输出。
5. 实现结果状态、失败提示和一次手动重试。
6. 本地清理透明边缘、裁切和缩放。
7. 实现透明 Emoji 下载。
8. 对 10～20 张测试图记录生成耗时和结果。

当天验收：用户可以生成并下载透明底 Emoji；无实色矩形背景；至少 60% 的测试结果能被识别为原主体，并基本符合目标风格。

如果第二天结束时透明 Emoji 仍无法稳定保留主体，第三天优先继续调试生成，不投入复杂回贴。

### 第 3 天：背景修复、回贴与演示整理

Codex 自动完成：

1. 根据原主体蒙版生成适当扩张的修复蒙版。
2. 调用 OpenAI Image API 修复主体背后的背景。
3. 使用原蒙版质心和面积恢复 Emoji 的位置与大小。
4. 增加轻微接触阴影和局部亮度适配。
5. 实现原图/结果图切换和下载。
6. 补充阶段进度、60 秒超时和错误恢复。
7. 增加主要流程自动测试。
8. 输出原图、蒙版、透明 Emoji 和最终合成图。
9. 生成一页简化验证结果表。

当天验收：至少 5 张代表性测试图完成端到端处理；用户可分别下载透明 Emoji 和替换结果；失败能够定位到分割、生成、修复或合成阶段。

## 6. 为压缩周期删除的内容

- 账户、登录、付费和权限。
- 数据库、Redis、分布式队列和云对象存储。
- 公开部署、域名和 HTTPS。
- 人物、宠物和复杂遮挡专项能力。
- 精细画笔式蒙版编辑。
- 多主体同时替换和编辑历史。
- 自动生成多个候选并复杂评分。
- 高并发、断点恢复和跨设备恢复。
- 生产级监控、审计和数据治理。
- 100 张规模的正式评测。
- 单独训练或微调模型。
- SVG、分层 3D 或可编辑图层输出。

## 7. 自动化实现与用户控制

### 7.1 Codex 可以自动完成

- 项目创建、依赖配置和本地启动脚本。
- 两个页面的 UI 和完整状态流转。
- 图片上传、校验、显示和下载。
- fal.ai 与 OpenAI 的后端适配器。
- 候选蒙版、点击选择和框选纠错。
- Emoji 提示模板和透明图片后处理。
- 背景修复、尺寸恢复和本地合成。
- 超时、失败提示和一次手动重试。
- 主要自动测试和验证结果整理。
- 本地运行说明和 API Key 配置说明。

### 7.2 用户必须提供或控制

| 项目 | 用户操作 |
|---|---|
| fal.ai 凭据 | 创建 `FAL_KEY`，保存在本地环境变量中 |
| OpenAI 凭据 | 创建 `OPENAI_API_KEY`，保存在本地环境变量中 |
| 测试图片 | 将 10～20 张有使用权的图片放入约定目录 |
| 风格判断 | 指出最符合和最不符合预期的结果 |
| 关键特征 | 对失败案例说明必须保留的结构或颜色 |
| API 费用 | 确认第三方账户有可用额度并承担调用费用 |
| 最终取舍 | 第 2 天决定第三天优先做回贴，还是继续优化透明 Emoji |

密钥由用户在本机写入 `.env` 或系统环境变量。Codex只检查变量是否存在，不在对话、日志和代码中输出密钥值。

## 8. MVP 接口

```text
POST /api/images
GET /api/images/{imageId}/objects
POST /api/images/{imageId}/refine
POST /api/emojis
GET /api/jobs/{jobId}
POST /api/compositions
```

MVP 使用短轮询获取任务状态，不引入 WebSocket、Webhook 或外部队列。

## 9. MVP 成功标准

在用户提供的 10～20 张图片上：

- 80% 的图片可以通过候选蒙版或框选得到目标主体。
- 60% 的透明 Emoji 能识别为原主体，并基本符合目标风格。
- 透明底 PNG 无明显矩形背景和严重边缘脏污。
- 至少 5 张图片完成背景修复和原位回贴。
- 回贴结果不存在明显尺寸和位置错误。
- 正常 API 队列下，主要流程尽量在 60 秒内结束或明确失败。
- 每个失败都能归因到分割、生成、修复或合成阶段。

该门槛用于判断技术方向是否值得继续，不代表生产环境质量。

## 10. 开始实施前的准备

用户需要：

1. 在本机配置 `FAL_KEY`。
2. 在本机配置 `OPENAI_API_KEY`。
3. 将 10～20 张测试图片放入项目约定的测试目录。

完成以上准备后，其余 MVP 开发和本地验证可以由 Codex连续执行。
