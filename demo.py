import cv2
import mediapipe as mp
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def test_and_visualize(input_path, output_path, model_path):    
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        return

    # --- 1. 初始化模型 ---
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.5
    )

    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        # 读取原图
        image = cv2.imread(input_path)
        if image is None:
            print(f"错误：无法读取图片 {input_path}")
            return

        # 转换为 RGB 进行推理
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        result = recognizer.recognize(mp_image)

        # --- 2. 结果判定与绘图逻辑 ---
        # 默认显示文字
        display_text = "None"
        color = (0, 0, 255) # 默认红色 (BGR格式)

        if result.gestures and len(result.gestures) > 0:
            top_gesture = result.gestures[0][0]
            gesture_name = top_gesture.category_name
            confidence = top_gesture.score

            if gesture_name != "None":
                display_text = f"{gesture_name} ({confidence:.2f})"
                color = (0, 255, 0) # 识别成功用绿色
        
        # --- 3. 在原图上绘制文字 ---
        # 参数：图片, 文字内容, 坐标(x,y), 字体, 字号, 颜色, 粗细
        cv2.putText(image, display_text, (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

        # --- 4. 打印并保存 ---
        print(f"最终判定: {display_text}")
        cv2.imwrite(output_path, image)
        print(f"结果图已保存至: {output_path}")

if __name__ == "__main__":
    model_path = 'gesture_recognizer.task'
    input_path = 'thumb_up.jpg'
    output_path = 'result01.jpg'
    test_and_visualize(input_path, output_path, model_path)