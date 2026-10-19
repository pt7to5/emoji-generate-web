# BeEmoji 更新发布说明

本发布包包含当前前端首页、生成页面、双文件下载能力，以及后端 Prompt V2、质量审核与本地启动修复。发布包不包含 `.env`、API Key、用户上传图片、生成结果、虚拟环境或 `node_modules`。

## GitHub 更新

1. 解压发布包。
2. 将解压后的文件覆盖到 `pt7to5/emoji-generate-web` 仓库本地副本。
3. 不要上传 `.env`。
4. 提交并推送到 `main`。仓库中的 `.github/workflows/deploy-pages.yml` 会自动构建并发布 GitHub Pages。
5. 在 GitHub 的 **Actions** 页面确认 `Deploy frontend to GitHub Pages` 成功。

线上前端预计地址：<https://pt7to5.github.io/emoji-generate-web/>

## 阿里云后端更新

后端位于 `backend/`，构建上下文为 `/backend`，Dockerfile 为 `backend/Dockerfile`。在 ACR `pt-emoji/emoji-backend` 新建不可变镜像标签（建议使用发布日期或 Git 提交号），构建成功后，将 FC 杭州区域的 `emoji-function` 更新到新镜像并部署。不要覆盖旧镜像标签，以便回滚。

FC 中保留原有密钥与 OSS 配置，并确认至少存在：`CORS_ORIGINS=https://pt7to5.github.io`、`PUBLIC_BASE_URL=https://emoji-function-bkheqosclx.cn-hangzhou.fcapp.run`、`DATA_DIR=/mnt/oss/emoji-tasks`。API Key 只能配置在 FC 环境变量中。

部署后先访问 `/api/health`，再实际验证上传、选区、生成、合成，以及“编辑图 + 透明 Emoji 贴纸”双文件下载。
