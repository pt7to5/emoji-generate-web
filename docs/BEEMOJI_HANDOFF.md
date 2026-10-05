# BeEmoji 项目交接文档

更新时间：2026-10-03  
项目阶段：MVP 已完成公网端到端联调，进入稳定性、性能与小规模用户验证阶段

## 1. 产品概述

BeEmoji 是一款 AI 图片编辑产品。用户上传照片后，可以选择其中的真实物品，将其转换为保留原有轮廓、姿态、颜色和装饰特征的 Emoji 风格图像，并替换回原图对应位置。

当前优先场景为甜品、饮品、玩偶和常见静物。复杂遮挡、透明物体、人物和多人场景暂不作为 MVP 的重点。

核心产品原则：

- 普通用户不填写 API Key，所有第三方能力由服务端统一调用。
- 生成结果优先保留主体身份、结构、视角、颜色分区和关键细节。
- 用户可以涂抹指定主体，也可以自动识别并手动修正 Mask。
- 前后端独立部署，后端密钥不进入浏览器和公开仓库。

## 2. 当前线上状态

### 2.1 前端

- 托管平台：GitHub Pages。
- 仓库：`tongpan0725/emoji-generate-web`，当前为公开仓库。
- 线上地址：<https://tongpan0725.github.io/emoji-generate-web/>
- 发布方式：推送 `main` 分支后，由 `.github/workflows/deploy-pages.yml` 自动构建和部署。
- 生产构建参数：

```text
VITE_API_BASE_URL=https://emoji-function-bkheqosclx.cn-hangzhou.fcapp.run
Base path=/emoji-generate-web/
```

GitHub Actions 只监听 `frontend/**` 和工作流文件，纯后端提交不会重复部署前端。

### 2.2 后端

- 云平台：阿里云函数计算 FC，自定义容器。
- 地域：华东 1（杭州），`cn-hangzhou`。
- 函数名称：`emoji-function`。
- 公网基础地址：<https://emoji-function-bkheqosclx.cn-hangzhou.fcapp.run>
- 健康检查：<https://emoji-function-bkheqosclx.cn-hangzhou.fcapp.run/api/health>
- 监听端口：`9000`。
- 当前计算规格：`2 vCPU / 4 GB`，单实例并发度 `1`。
- 容器仓库：ACR `pt-emoji/emoji-backend`。
- 当前优化版本使用镜像标签：`v1.0.2`；应在 FC 控制台再次确认实际生效的镜像 Digest。
- 函数角色：`emoji-fc-role`，信任主体为函数计算服务，并具有访问任务 OSS Bucket 的权限。

直接在浏览器打开 `fcapp.run` 文件或健康检查时，阿里云共享域名可能触发下载。这不等于 API 不可调用；前端通过 `fetch` 使用该地址。

### 2.3 数据存储

- 后端任务 Bucket：`emoji-pt-bucket`。
- 前端实验 Bucket：`emoji-web-bucket`，目前不承担正式前端托管。
- 推荐 OSS 挂载：Bucket 根目录挂载到函数 `/mnt/oss`，权限为读写。
- 推荐环境变量：`DATA_DIR=/mnt/oss/emoji-tasks`。
- 原图、Mask、主体预览、Emoji 和最终合成图按 `imageId` 存放。
- FastAPI 通过 `/files/...` 暴露任务文件，`PUBLIC_BASE_URL` 用于生成绝对地址。

必须保证任务文件写入 OSS 挂载，而不是容器临时磁盘。否则实例切换或重启后会出现“图片任务不存在”。

## 3. 已实现的产品功能

### 3.1 上传与预览

- 支持 JPG、PNG、WebP。
- 单文件最大 15 MB。
- 后端最长边限制为 4096 像素。
- 上传后前端立即显示本地预览，服务端保存标准化 PNG。

### 3.2 主体选择

- `Paint`：用户在主体内部涂抹，MobileSAM 根据正向点提示生成完整 Mask。
- `Auto Detect`：百度多主体检测与 Qwen 视觉检测并行，合并候选主体；随后通过百度智能抠图生成 Mask。
- `Refine`：用户可以增加或删除 Mask 区域。
- 支持多个主体缩略图、主体切换、白色虚线边缘和半透明遮罩。

### 3.3 Emoji 生成与合成

- 阿里云百炼 `wan2.6-image` 生成 Emoji 风格主体。
- 百度智能抠图将生成图处理为透明 PNG。
- 本地进行颜色强度匹配，抑制过亮、过饱和和明显色块。
- 生成结果与原主体差异过大时自动执行一次严格纠偏生成。
- 阿里云百炼 `wan2.7-image` 修复主体后方背景。
- Pillow/OpenCV 在原位置完成 Emoji 合成。
- 支持多主体连续替换、撤回最近替换和下载最终图片。

### 3.4 当前前端操作顺序

```text
Paint → Refine → Auto Detect → Generate → Download
```

对应图标：

```text
🪄 → ✏️ → 🔍 → 🎨 → ✨
```

## 4. 技术架构

```text
手机/浏览器
    │
    ├── GitHub Pages：React + TypeScript + Vite
    │
    └── HTTPS API：阿里云函数计算 FC
            │
            ├── FastAPI + Uvicorn
            ├── MobileSAM + PyTorch + Ultralytics
            ├── OpenCV + Pillow
            ├── 百度：检测、智能抠图
            ├── 阿里云百炼：视觉检测、Emoji 生成、背景修复
            └── OSS 挂载：任务原图、Mask 和结果文件
```

主要代码位置：

```text
frontend/src/App.tsx                 页面、编辑器状态与交互
frontend/src/api.ts                 后端 API 封装
frontend/src/styles.css             页面与编辑器样式
backend/app/main.py                 FastAPI 路由与任务编排
backend/app/local_segmentation.py   MobileSAM 涂抹分割
backend/app/providers_domestic.py   百度与阿里云百炼适配
backend/app/image_ops.py            Mask、颜色、背景和合成算法
backend/app/config.py               服务端环境变量
backend/Dockerfile                  FC 自定义容器镜像
```

## 5. 关键请求流程

### 5.1 涂抹识别

```text
上传图片
→ POST /api/images
→ 用户涂抹
→ POST /api/images/{imageId}/refine-paint
→ MobileSAM 点提示分割
→ 保存 mask 和主体预览
→ 前端 Canvas 绘制遮罩与白色虚线
```

大图在 SAM 推理前缩放至最长边 1536，点坐标和笔刷半径同步缩放，最终 Mask 恢复到原图尺寸。

### 5.2 自动识别

```text
POST /api/images/{imageId}/detect
→ 百度主体检测 + Qwen 视觉检测并行
→ 合并候选框
→ 百度智能抠图
→ 完整性过滤
→ 返回多个候选主体
```

### 5.3 生成与合成

```text
POST /api/emojis
├── 百炼生成 Emoji → 百度透明抠图 → 质量检查/必要时纠偏
└── 同时预生成并缓存背景修复

POST /api/compositions/batch
→ 复用同一 Mask 版本的背景缓存
→ 颜色与光线协调
→ 合成并返回 resultUrl
```

前端接口保持同步模式，尚未引入任务队列和轮询。

## 6. 已解决的关键问题

### 6.1 Git 仓库所有权错误

Windows 沙箱用户和管理员用户 SID 不一致，Git 曾提示 `dubious ownership`。已由用户完成合并、提交和推送；后续仍应由实际登录用户执行 Git 写操作。

### 6.2 FC 角色无法 AssumeRole

创建 `emoji-fc-role` 后，补充了信任策略：

```json
{
  "Statement": [
    {
      "Action": "sts:AssumeRole",
      "Effect": "Allow",
      "Principal": { "Service": ["fc.aliyuncs.com"] }
    }
  ],
  "Version": "1"
}
```

并为角色授予 OSS 访问权限。

### 6.3 OpenCV 容器启动失败

曾出现：

```text
ImportError: libxcb.so.1: cannot open shared object file
```

Dockerfile 已安装 `libgl1`、`libglib2.0-0`、`libsm6`、`libx11-6`、`libxext6`、`libxrender1` 和 `libxcb1`。

### 6.4 GitHub Pages 子路径资源错误

生产构建使用 `/emoji-generate-web/` base path，静态资源通过 `import.meta.env.BASE_URL` 解析。

### 6.5 前后端跨域与 Mask 不显示

前端与后端分别位于 GitHub Pages 和阿里云 FC，Canvas 读取跨域 Mask 时曾被浏览器阻止。已在 Mask 图片加载前设置：

```ts
image.crossOrigin = "anonymous"
```

同时后端必须正确配置 GitHub Pages Origin。

### 6.6 图片任务不存在

根因是任务目录未持久化、实例切换或 OSS 挂载路径与 `DATA_DIR` 不一致。正式运行必须验证 `/mnt/oss/emoji-tasks/{imageId}` 可以跨实例读取。

### 6.7 处理耗时过长

已完成以下优化：

- FC 使用 2 vCPU / 4 GB，单实例并发 1。
- 大图 SAM 推理最长边限制为 1536，再还原 Mask。
- Emoji 生成与背景修复并行。
- 按 Mask 文件版本缓存背景修复结果。
- 添加分阶段耗时日志。

质量校验、自动纠偏和失败重试继续保留。

## 7. 必需环境变量

环境变量在阿里云 FC 控制台中以“名称”和“值”分别填写，不要把 `NAME=value` 整体填入名称字段。

```text
CORS_ORIGINS=https://tongpan0725.github.io
PUBLIC_BASE_URL=https://emoji-function-bkheqosclx.cn-hangzhou.fcapp.run
DATA_DIR=/mnt/oss/emoji-tasks

BAIDU_DETECT_API_KEY=<服务端密钥>
BAIDU_DETECT_SECRET_KEY=<服务端密钥>
BAIDU_PROCESS_API_KEY=<服务端密钥>
BAIDU_PROCESS_SECRET_KEY=<服务端密钥>
DASHSCOPE_API_KEY=<服务端密钥>
```

要求：

- `CORS_ORIGINS` 只填写 Origin，不包含 `/emoji-generate-web`，末尾不加 `/`。
- `PUBLIC_BASE_URL` 末尾不加 `/`。
- API Key 不提交到 Git，不写入前端环境变量。
- 若未来绑定自定义前端域名，需要把该 Origin 加入 `CORS_ORIGINS`。

## 8. 发布流程

### 8.1 前端

本地验证：

```powershell
cd frontend
$env:VITE_API_BASE_URL="https://emoji-function-bkheqosclx.cn-hangzhou.fcapp.run"
npm.cmd run build -- --base=/emoji-generate-web/
```

提交涉及 `frontend/**` 的修改并推送 `main`，GitHub Actions 自动发布。等待 `Deploy frontend to GitHub Pages` 变绿后，再用无痕窗口验证。

### 8.2 后端

本地验证：

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest -q --basetemp=.test-tmp-deploy -p no:cacheprovider
.\.venv\Scripts\python.exe -m compileall app
```

ACR 构建规则：

```text
分支：main
构建上下文：/backend
Dockerfile：Dockerfile
镜像标签：使用新版本号，例如 v1.0.3；不要覆盖旧标签
```

镜像构建成功后，在 FC 中更新镜像版本并部署。确认“镜像资源准备完成”后，用新图片执行回归测试。保留上一镜像标签用于回滚。

## 9. 验证清单

每次发布至少检查：

1. `/api/health` 返回 `{"ok":true}`。
2. 上传一张新的 JPG/PNG 成功。
3. `Paint` 能返回主体缩略图。
4. 原图上出现半透明遮罩和白色虚线。
5. `Refine` 的增加、删除和撤回可用。
6. `Auto Detect` 能返回候选主体或明确失败提示。
7. `Generate` 能生成 Emoji 并完成背景合成。
8. `Download` 能获得最终 PNG。
9. 再上传第二张图片，确认不会出现“图片任务不存在”。
10. 分别使用桌面浏览器和手机测试。

建议记录：

```text
上传耗时：
首次 Paint 耗时：
第二次 Paint 耗时：
Emoji generation 耗时：
Background prepared/cache_hit：
Batch composition 耗时：
最终结果与原主体相似度：
```

函数日志可搜索：

```text
paint segmentation
emoji generation
background prepared
batch composition
```

## 10. 当前实现效果

已验证的正向结果：

- GitHub Pages 可以正常打开前端。
- 浏览器可以访问阿里云后端，完整链路曾成功生成最终图片。
- 涂抹识别可以输出主体预览。
- Mask 跨域 Canvas 问题已有代码修复。
- 后端容器可以在 FC 启动，OpenCV 依赖问题已解决。
- 性能优化代码已通过 Python 编译检查和测试。

最新自动化验证结果：

```text
30 passed
```

仍需在线回归确认：

- `v1.0.2` 实际 FC Digest 是否已生效。
- 优化后首次/热实例 Paint 的真实耗时。
- 并行背景修复后的 Generate 总耗时。
- OSS 挂载跨实例持久性。
- 手机端在不同 Wi-Fi/蜂窝网络下的稳定性。

## 11. 已知风险与待办

### P0：上线前必须处理

- 确认 OSS 生命周期规则，任务图片建议 24 小时自动删除。
- 验证 OSS 挂载持久化，避免实例切换后任务丢失。
- 增加匿名接口限流或访问令牌，防止第三方 API 额度被滥用。
- 确认 FC 执行超时时间足够覆盖生成链路，建议 300 秒。
- 为失败接口提供可定位的错误码，减少统一显示 `Load failed`。

### P1：体验与性能

- 记录真实 P50/P95 耗时和第三方 API 成功率。
- 将同步长请求升级为异步任务：创建任务、查询状态、获取结果。
- 提供明确的阶段进度和超时提示。
- 缓存百度 Access Token，减少重复认证请求。
- 评估 MobileSAM ONNX Runtime 或专用推理服务。

### P2：正式发布

- 购买并备案自定义域名。
- 前端迁移到适合中国大陆访问的正式静态托管/CDN。
- 后端绑定正式自定义域名或 API 网关，替代共享 `fcapp.run` 域名。
- 完成隐私政策、用户协议、第三方处理说明和内容安全策略。
- 建立监控、告警、灰度发布和版本回滚流程。

## 12. 注意事项

- 旧页面中保存的 `imageId` 不应跨部署继续使用，回归测试必须重新上传图片。
- 直接打开 FC 文件地址出现下载行为属于共享域名特性；判断 API 是否可用应查看 HTTP 状态和前端请求。
- iPhone HEIC/HEIF 当前不在后端允许格式中，应先转换为 JPG/PNG，或后续增加 HEIC 支持。
- GitHub Pages 适合当前测试，不等同于已完成中国大陆正式上线和 ICP 备案。
- `docs/superpowers/specs/2026-10-02-aliyun-fc-oss-deployment-design.md` 是早期方案，其中 OSS 前端托管和短期签名 URL 尚未按文档实现；以本交接文档描述的实际状态为准。

## 13. 当前工作区状态

性能优化主体代码已提交到：

```text
f95a767 Optimize segmentation and image generation performance
```

本交接时仍有新增回归测试处于工作区未提交状态：

```text
backend/tests/test_image_ops.py
backend/tests/test_local_segmentation.py
```

提交交接文档时应将上述测试一并纳入下一次提交。
