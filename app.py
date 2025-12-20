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
# 1. 设置图片路径
# 注意：在 GitHub 上一定要确保 static/gallery 文件夹里已经有图片了
STATIC_FOLDER = 'static'
GALLERY_FOLDER = os.path.join(STATIC_FOLDER, 'gallery')

# 确保文件夹存在（防止报错）
for folder in [STATIC_FOLDER, GALLERY_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

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
        "Traditional Tujia brocade (Xilankapu) textile pattern. "
        "Pixel art style, cross-stitch embroidery texture. "
        "Visible woven thread grain. Flat orthographic view. "
        "Strict geometric straight lines, NO curves. "
    )

    if "花" in keyword or "岩墙" in keyword:
        content_desc = (
            "Subject: Geometric Diamond Flower Pattern (Yanqianghua style). "
            "Main Motif: A vertical column of LARGE, prominent, nested concentric Rhombus (Diamond) shapes. "
            "Center: Inside each large diamond, there are smaller nested hexagons and triangles. "
            "Skeleton: The large diamonds are surrounded by a continuous 'Hook' meander fret pattern border. "
            "Composition: Vertical runner, repeating from top to bottom (Two-way continuous). "
            "Style: Abstract geometric, indigenous folk art, NOT architectural. "
        )
    elif "鸟" in keyword or "阳雀" in keyword:
        content_desc = "Subject: Abstract geometric bird totem, sharp triangles, symmetric totem, repeating pattern. "
    else:
        content_desc = f"Subject: Geometric pattern based on concept '{keyword}', repeating abstract shapes. "

    color_rule = (
        "Color Palette: "
        "Background is Deep Indigo Blue (Blackish Blue). "
        "Pattern colors: Bright Madder Red, Golden Yellow, Cyan Blue, White and Black accents. "
        "Solid color blocks, high contrast, no gradients. "
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

# ✅ 新增功能：获取本地画廊图片列表
@app.route('/get_gallery', methods=['GET'])
def get_gallery_images():
    images = []
    # 扫描 static/gallery 文件夹里的所有文件
    if os.path.exists(GALLERY_FOLDER):
        for filename in os.listdir(GALLERY_FOLDER):
            # 只读取图片文件
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                # 生成给网页用的链接
                images.append(f"/static/gallery/{filename}")
    
    # 按文件名排序，保证大家看到的顺序一样
    images.sort()
    return jsonify({"success": True, "images": images})

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    keyword = data.get('keyword', '').strip()
    
    if not keyword:
        return jsonify({"success": False, "error": "请输入关键词"})

    try:
        prompt_text = build_expert_prompt(keyword)
        print(f"Prompt: {prompt_text[:60]}...")

        image_bytes = query_huggingface_api({
            "inputs": prompt_text,
            "parameters": {
                "negative_prompt": "architecture, building, tower, face, rug, carpet, central medallion, realistic flower, round shape, blurry, low quality, 3d render, messy lines, curves, organic shapes, watermark, text, realistic photo",
                "width": 768,
                "height": 1024,
                "num_inference_steps": 30, 
                "guidance_scale": 7.5,
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
