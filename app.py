import os
import time
import requests
import json
from flask import Flask, render_template, request, jsonify

# 尝试导入 dotenv，用于本地开发读取 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = Flask(__name__)

# ================= 配置区域 =================
# 1. 设置图片保存路径
STATIC_FOLDER = 'static'
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)

# 2. Hugging Face API 配置
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

if not HF_API_TOKEN:
    print("⚠️ 严重警告: 未检测到 HF_API_TOKEN 环境变量！")

# 使用 Stable Diffusion XL Base 1.0
API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# ================= 核心逻辑 =================

def build_expert_prompt(keyword):
    """Prompt 工程"""
    base_style = (
        "traditional Tujia brocade (Xilankapu) textile pattern, "
        "pixel art style, cross-stitch embroidery texture, "
        "visible woven thread grain, flat orthographic view, "
        "strict geometric straight lines, no curves. "
    )

    if "花" in keyword or "岩墙" in keyword:
        content_desc = (
            "Subject: Traditional Tujia Brocade 'Rock Wall Flower' (Yanqianghua). "
            "Composition: A vertical runner pattern (long strip). "
            "Pattern: A continuous column of interlocking Hexagons and Diamonds. "
            "Structure: The 'Eight-Hook' geometric skeleton frame. "
            "Center: Abstract nested rhombuses, NOT a realistic flower. " 
            "Layout: Repeating pattern from top to bottom (Two-way continuous). "
            "Style: NOT a square rug, but a long textile fabric. "
        )
    elif "鸟" in keyword or "阳雀" in keyword:
        content_desc = "Subject: Abstract geometric bird totem, sharp triangles, symmetric totem, repeating pattern. "
    else:
        content_desc = f"Subject: Geometric pattern based on concept '{keyword}', repeating abstract shapes. "

    color_rule = (
        "Color Palette: "
        "Background is 40% Indigo Blue (Deep Navy). "
        "Decor is 30% Madder Red, 20% Gardenia Yellow, 10% Black outlines. "
        "Solid color blocks, high contrast, no gradients, no blending. "
    )

    return f"{base_style} {content_desc} {color_rule}"

def query_huggingface_api(payload):
    try:
        response = requests.post(API_URL, headers=HEADERS, json=payload)
        return response.content
    except Exception as e:
        print(f"网络请求错误: {e}")
        return None

# ================= 路由接口 =================

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    keyword = data.get('keyword', '').strip()
    
    if not keyword:
        return jsonify({"success": False, "error": "请输入关键词"})

    try:
        prompt_text = build_expert_prompt(keyword)
        print(f"正在请求 AI, Prompt: {prompt_text[:60]}...")

        image_bytes = query_huggingface_api({
            "inputs": prompt_text,
            "parameters": {
                "negative_prompt": "rug, carpet, central medallion, realistic flower, round shape, blurry, low quality, 3d render, messy lines, curves, organic shapes, watermark, text, realistic photo",
                "width": 768,
                "height": 1024,
                "num_inference_steps": 25, 
            }
        })

        if image_bytes is None:
             return jsonify({"success": False, "error": "网络连接失败。"})

        try:
            text_response = image_bytes.decode('utf-8')
            error_msg = json.loads(text_response)
            if isinstance(error_msg, dict) and 'error' in error_msg:
                err_str = str(error_msg['error'])
                if "loading" in err_str:
                    return jsonify({"success": False, "error": "模型正在唤醒中，请等待 20 秒后再次点击生成..."})
                return jsonify({"success": False, "error": f"API 报错: {err_str}"})
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass

        filename = f"gen_{int(time.time())}.png"
        save_path = os.path.join(STATIC_FOLDER, filename)
        
        with open(save_path, 'wb') as f:
            f.write(image_bytes)

        return jsonify({"success": True, "image_url": f"static/{filename}"})

    except Exception as e:
        print(f"生成失败: {e}")
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    app.run()
