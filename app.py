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
# ✅ 从环境变量获取 Token (安全模式)
HF_API_TOKEN = os.environ.get("HF_API_TOKEN")

# 增加一个检查，防止没配置变量导致程序崩溃
if not HF_API_TOKEN:
    print("⚠️ 严重警告: 未检测到 HF_API_TOKEN 环境变量！程序可能无法正常生成图片。")
    print("请在本地创建 .env 文件，或在 Render 后台添加 Environment Variable。")

# 使用 Stable Diffusion XL Base 1.0
API_URL = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# ================= 核心逻辑 =================

def build_expert_prompt(keyword):
    """
    Prompt 工程：针对 AI 误解“岩墙”为建筑的问题进行修正，强制 AI 画出类似参考图的“大菱形嵌套”结构。
    """
    # 基础风格：强调西兰卡普织锦的质感和几何风格
    base_style = (
        "Traditional Tujia brocade (Xilankapu) textile pattern. "
        "Pixel art style, cross-stitch embroidery texture. "
        "Visible woven thread grain. Flat orthographic view. "
        "Strict geometric straight lines, NO curves. "
    )

    # 针对“岩墙花”的特殊结构描述 (✅ 修正版 4.0 - 终极形态)
    if "花" in keyword or "岩墙" in keyword:
        content_desc = (
            "Subject: Geometric Diamond Flower Pattern (Yanqianghua style). "
            # 关键修改：精确描述参考图的结构
            "Main Motif: A vertical column of LARGE, prominent, nested concentric Rhombus (Diamond) shapes. "
            "Center: Inside each large diamond, there are smaller nested hexagons and triangles, forming a complex geometric core. "
            "Skeleton: The large diamonds are surrounded by a continuous 'Hook' meander fret pattern border. "
            "Composition: Vertical runner, repeating from top to bottom (Two-way continuous). "
            "Style: Abstract geometric, indigenous folk art, NOT architectural, NOT a face. "
        )
    elif "鸟" in keyword or "阳雀" in keyword:
        content_desc = "Subject: Abstract geometric bird totem, sharp triangles, symmetric totem, repeating pattern. "
    else:
        content_desc = f"Subject: Geometric pattern based on concept '{keyword}', repeating abstract shapes. "

    # 色彩规范：参考图的经典配色
    color_rule = (
        "Color Palette: "
        "Background is Deep Indigo Blue (Blackish Blue). "
        "Pattern colors: Bright Madder Red (Pinkish Red), Golden Yellow, Cyan Blue, White and Black accents. "
        "Solid color blocks, high contrast, no gradients. "
    )

    return f"{base_style} {content_desc} {color_rule}"

def query_huggingface_api(payload):
    """发送请求给 Hugging Face"""
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
        # 1. 构建提示词
        prompt_text = build_expert_prompt(keyword)
        print(f"正在请求 AI, Prompt: {prompt_text[:60]}...")

        # 2. 调用 API (增加 negative_prompt 禁止画地毯)
        image_bytes = query_huggingface_api({
            "inputs": prompt_text,
            "parameters": {
                # 负向提示词加大了力度，禁止 'tower', 'building', 'face', 'rug'
                "negative_prompt": "architecture, building, tower, face, totem pole, rug, carpet, central medallion, realistic flower, round shape, blurry, low quality, 3d render, messy lines, curves, organic shapes, watermark, text, realistic photo",
                "width": 768,  # 保持长条形比例
                "height": 1024,
                "num_inference_steps": 30, # 稍微增加步数，提高精细度
                "guidance_scale": 7.5, # 提高对 Prompt 的遵循度
            }
        })

        if image_bytes is None:
             return jsonify({"success": False, "error": "网络连接失败，请检查网络设置。"})

        # 3. 错误处理逻辑 (二进制检查)
        try:
            # 尝试将字节流解码为字符串
            text_response = image_bytes.decode('utf-8')
            # 如果能解码成字符串，尝试解析是否为 JSON 错误信息
            error_msg = json.loads(text_response)
            
            if isinstance(error_msg, dict) and 'error' in error_msg:
                err_str = str(error_msg['error'])
                if "loading" in err_str:
                    return jsonify({"success": False, "error": "模型正在唤醒中，请等待 20 秒后再次点击生成..."})
                return jsonify({"success": False, "error": f"API 报错: {err_str}"})
                
        except (UnicodeDecodeError, json.JSONDecodeError):
            # 捕获解码错误：说明 image_bytes 是图片（二进制数据），不是 JSON 文本
            # 这是成功的标志，直接跳过错误处理，继续保存图片
            pass

        # 4. 保存图片
        filename = f"gen_{int(time.time())}.png"
        save_path = os.path.join(STATIC_FOLDER, filename)
        
        with open(save_path, 'wb') as f:
            f.write(image_bytes)

        return jsonify({"success": True, "image_url": f"static/{filename}"})

    except Exception as e:
        print(f"生成失败: {e}")
        return jsonify({"success": False, "error": str(e)})

if __name__ == '__main__':
    # 本地运行时开启 Debug 模式
    # Render 部署时，Gunicorn 会忽略这里，直接调用 app
    app.run(debug=True, port=5000)
