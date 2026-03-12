import cv2
import yaml
import argparse
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os, time, uvicorn, sys

app = FastAPI()
recognizer = None
cfg = {}

# 定位到项目根目录 (mediapipe/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_abs_path(rel_path):
    """智能转换路径：如果是相对路径则基于根目录拼接"""
    if os.path.isabs(rel_path):
        return os.path.normpath(rel_path)
    return os.path.normpath(os.path.join(PROJECT_ROOT, rel_path))

def load_model():
    global recognizer
    abs_model_path = get_abs_path(cfg['model_path'])
    if not os.path.exists(abs_model_path):
        print(f"❌ 找不到模型文件: {abs_model_path}")
        sys.exit(1)
        
    base_options = python.BaseOptions(model_asset_path=abs_model_path)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=cfg['threshold']
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)
    print(f"✅ 模型加载成功: {abs_model_path}")

def save_result(image_bytes, gesture, score):
    """后台异步绘制并保存识别结果"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    text = f"{gesture} ({score:.2f})"
    color = (0, 255, 0) if gesture != "None" else (0, 0, 255)
    cv2.putText(img, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    abs_save_dir = get_abs_path(cfg['save_dir'])
    if not os.path.exists(abs_save_dir):
        os.makedirs(abs_save_dir)
        
    fname = f"{time.strftime('%Y%m%d_%H%M%S')}_{gesture}.jpg"
    cv2.imwrite(os.path.join(abs_save_dir, fname), img)
    print(f"💾 识别结果已存入: {os.path.join(abs_save_dir, fname)}")

@app.post("/gesture")
async def recognize(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # 记录推理开始时间
    start_inference = time.perf_counter()

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result = recognizer.recognize(mp_image)

    # 计算推理耗时 (秒转为毫秒)
    inference_ms = (time.perf_counter() - start_inference) * 1000
    
    gesture = result.gestures[0][0].category_name if result.gestures else "None"
    score = result.gestures[0][0].score if result.gestures else 0.0
    
    background_tasks.add_task(save_result, contents, gesture, score)
    return {
        "gesture": gesture, 
        "confidence": round(float(score), 3),
        "inference_ms": round(inference_ms, 2)  # 推理耗时
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(PROJECT_ROOT, "configs/server_config.yaml"))
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    load_model()
    uvicorn.run(app, host=cfg['host'], port=cfg['port'])
