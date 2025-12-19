import os
import time
import requests
import json
from flask import Flask, render_template, request, jsonify

# 尝试导入 dotenv，用于本地开发读取 .env 文件
# 如果在 Render 上运行，没有这个库也没关系，它会直接读取系统变量
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
    Prompt 工程：将用户简单的中文关键词，强制转化为“岩墙花”的非遗考据级指令
    """
    # 基础风格
    base_style = (
        "traditional Tujia brocade (Xilankapu) textile pattern, "
        "pixel art style, cross-stitch embroidery texture, "
        "visible woven thread grain, flat orthographic view, "
        "strict geometric straight lines, no curves. "
    )

    # 针对“岩墙花”的特殊结构描述 (✅ 修复版：强制长条形，禁止画成地毯)
    if "花" in keyword or "岩墙" in keyword:
        content_desc = (
            "Subject: Traditional Tujia Brocade 'Rock Wall Flower' (Yanqianghua). "
            "Composition: A vertical runner pattern (long strip). "  # 强制长条形
            "Pattern: A continuous column of interlocking Hexagons and Diamonds. " # 强制连续几何
            "Structure: The 'Eight-Hook' geometric skeleton frame. "
            "Center: Abstract nested rhombuses, NOT a realistic flower. " 
            "Layout: Repeating pattern from top to bottom (Two-way continuous). "
            "Style: NOT a square rug, but a long textile fabric. " # 禁止方地毯
        )
    elif "鸟" in keyword or "阳雀" in keyword:
        content_desc = "Subject: Abstract geometric bird totem, sharp triangles, symmetric totem, repeating pattern. "
    else:
        content_desc = f"Subject: Geometric pattern based on concept '{keyword}', repeating abstract shapes. "

    # 色彩规范
    color_rule = (
        "Color Palette: "
        "Background is 40% Indigo Blue (Deep Navy). "
        "Decor is 30% Madder Red, 20% Gardenia Yellow, 10% Black outlines. "
        "Solid color blocks, high contrast, no gradients, no blending. "
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
                "negative_prompt": "rug, carpet, central medallion, realistic flower, round shape, blurry, low quality, 3d render, messy lines, curves, organic shapes, watermark, text, realistic photo",
                "width": 768,  # ✅ 修改：宽度变窄
                "height": 1024, # ✅ 修改：高度变高，强制生成长条形
                "num_inference_steps": 25, 
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
