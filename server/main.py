import cv2
import yaml
import argparse
import numpy as np
import mediapipe as mp
from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os, time, uvicorn

app = FastAPI()
recognizer = None
cfg = {}

def load_model():
    global recognizer
    base_options = python.BaseOptions(model_asset_path=cfg['model_path'])
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=cfg['threshold']
    )
    recognizer = vision.GestureRecognizer.create_from_options(options)
    print(f"✅ 模型加载成功，输入模式: {cfg['input_mode']}")

def save_result(image_bytes, gesture, score):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # 标注逻辑
    text = f"{gesture} ({score:.2f})"
    color = (0, 255, 0) if gesture != "None" else (0, 0, 255)
    cv2.putText(img, text, (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    # 存储路径处理
    if not os.path.exists(cfg['save_dir']): os.makedirs(cfg['save_dir'])
    fname = f"{time.strftime('%Y%m%d_%H%M%S')}_{gesture}.jpg"
    cv2.imwrite(os.path.join(cfg['save_dir'], fname), img)

@app.post("/gesture")
async def recognize(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # 核心推理
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    result = recognizer.recognize(mp_image)
    
    gesture = result.gestures[0][0].category_name if result.gestures else "None"
    score = result.gestures[0][0].score if result.gestures else 0.0
    
    background_tasks.add_task(save_result, contents, gesture, score)
    return {"gesture": gesture, "confidence": round(float(score), 3)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/server_config.yaml", help="配置文件路径")
    parser.add_argument("--port", type=int, help="覆盖端口配置")
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        client_cfg = yaml.safe_load(f)
    
    if args.port: cfg['port'] = args.port
    
    load_model()
    uvicorn.run(app, host=cfg['host'], port=cfg['port'])