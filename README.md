# 手势识别系统 (C/S 架构)

基于 FastAPI 和 MediaPipe 开发的手势识别存证系统。

## 📂 目录结构
- `client/`: 客户端发送脚本
- `server/`: 服务端 FastAPI 接口
- `configs/`: 统一配置管理
- `data/`: 模型文件、运行数据及识别结果

## 🚀 快速开始

### 1. 环境准备
确保已安装 Python 3.8+，并在根目录创建虚拟环境：
```bash
python -m venv .venv
source .venv/bin/activate  # Linux
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

### 2. 准备模型
将 gesture_recognizer.task 放入 data/models/ 目录。

### 3. 启动服务端
``` bash
cd server
python main.py --config ../configs/server_config.yaml
```
### 3. 运行客户端
``` bash
cd client
python sender.py --config ../configs/client_config.yaml
```