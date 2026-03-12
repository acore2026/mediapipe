import cv2
import yaml
import argparse
import requests
import os

def process_and_send(cfg):
    source = cfg['input_source']
    
    # 预留逻辑：如果是视频流模式，这里可以改为 cv2.VideoCapture
    if not os.path.exists(source):
        print(f"❌ 找不到输入源: {source}")
        return

    img = cv2.imread(source)
    # 图像预处理
    if img.shape[1] > cfg['resize_width']:
        h, w = img.shape[:2]
        img = cv2.resize(img, (cfg['resize_width'], int(h * (cfg['resize_width'] / w))))
    
    _, img_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, cfg['quality']])
    
    try:
        resp = requests.post(cfg['server_url'], files={'file': ('img.jpg', img_buf.tobytes(), 'image/jpeg')})
        print(f"📡 响应内容: {resp.json()}")
    except Exception as e:
        print(f"💥 发送失败: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/client_config.yaml")
    parser.add_argument("--source", help="覆盖输入图片路径")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        client_cfg = yaml.safe_load(f)
    
    if args.source: client_cfg['input_source'] = args.source
    
    process_and_send(client_cfg)