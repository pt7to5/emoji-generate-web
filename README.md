# EmojiHTML MVP

本地运行的多主体识别、Emoji 风格化与原图替换原型。前端实现自 Figma 节点 `123:302` 与 `129:558`。

## 环境变量

项目根目录 `.env` 需要以下配置：

```dotenv
BAIDU_DETECT_API_KEY=多主体检测应用的API Key
BAIDU_DETECT_SECRET_KEY=多主体检测应用的Secret Key
BAIDU_PROCESS_API_KEY=图像处理应用的API Key
BAIDU_PROCESS_SECRET_KEY=图像处理应用的Secret Key
DASHSCOPE_API_KEY=阿里百炼API Key
```

## Windows 本地启动

后端终端：

```powershell
cd backend
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

前端终端：

```powershell
cd frontend
npm install
npm run dev -- --host 127.0.0.1
```

浏览器打开 `http://127.0.0.1:5173/`。

## 受组织管理设备的安装方式

如果系统 npm 缓存目录没有写入权限，改用项目内缓存：

```powershell
cd frontend
npm install --cache ..\work\npm-cache
npm run build
```

如果公司网络阻止 npm 或模型接口，把整个项目目录复制到另一台 Windows 设备；保留 `.env`、`backend/.venv` 和 `frontend/node_modules` 可以减少重复安装。密钥仅由本地后端读取，不会进入前端构建产物。

## 当前交互

1. 首页点击 `Generate`。
2. 编辑页上传 JPG、PNG 或 WebP。
3. 系统显示多个主体轮廓；白色虚线表示可选主体，白色加橙红高亮表示当前主体。
4. 点击主体轮廓进行切换。
5. 点击 `Generate` 完成风格化及原位替换。
6. 点击 `Download` 下载最终图片。
