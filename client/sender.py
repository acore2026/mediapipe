import cv2
import yaml
import argparse
import requests
import os, sys, time

# 定位到项目根目录 (mediapipe/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_abs_path(rel_path):
    if os.path.isabs(rel_path):
        return os.path.normpath(rel_path)
    return os.path.normpath(os.path.join(PROJECT_ROOT, rel_path))

def process_and_send(cfg, source_override=None):
    # 确定输入源路径
    raw_source = source_override if source_override else cfg['input_source']
    source_path = get_abs_path(raw_source)
    
    if not os.path.exists(source_path):
        print(f"❌ 找不到输入图片: {source_path}")
        return

    img = cv2.imread(source_path)
    if img is None:
        print(f"❌ 无法解析图片: {source_path}")
        return
    
    # 性能测试开始：记录从准备发送到收到响应的时间
    start_total = time.perf_counter()

    # 预处理：缩放
    if img.shape[1] > cfg['resize_width']:
        h, w = img.shape[:2]
        new_w = cfg['resize_width']
        img = cv2.resize(img, (new_w, int(h * (new_w / w))))
    
    # 编码压缩
    _, img_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, cfg['quality']])
    
    print(f"🚀 正在发送图片: {source_path}")
    try:
        resp = requests.post(
            cfg['server_url'], 
            files={'file': ('image.jpg', img_buf.tobytes(), 'image/jpeg')},
            timeout=10
        )
        # 性能测试结束
        total_ms = (time.perf_counter() - start_total) * 1000
        
        if resp.status_code == 200:
            data = resp.json()
            print("-" * 30)
            print(f"✅ 识别结果: {data['gesture']} ({data['confidence']})")
            print(f"⏱️ 纯推理耗时: {data['inference_ms']} ms")
            print(f"🌐 端到端总计: {round(total_ms, 2)} ms")
            print(f"📡 网络与开销: {round(total_ms - data['inference_ms'], 2)} ms")
            print("-" * 30)
        else:
            print(f"⚠️ 响应异常: {resp.status_code}")
    except Exception as e:
        print(f"💥 网络通信失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs/client_config.yaml"))
    parser.add_argument("--source", help="手动指定输入图片路径")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        client_cfg = yaml.safe_load(f)
    
    process_and_send(client_cfg, args.source)
