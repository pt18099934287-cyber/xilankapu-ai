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
STATIC_FOLDER = 'static'
GALLERY_FOLDER = os.path.join(STATIC_FOLDER, 'gallery')

# 确保文件夹存在
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

# ================= 核心逻辑：图片档案库 =================
# ✅ 已根据你的文件夹截图 (image_0cda02.jpg) 完美匹配文件名
GALLERY_DB = [
    # --- 经典纹样 ---
    {
        "filename": "8gou.jpg",
        "tags": ["八勾纹", "八勾", "8 hook", "brown", "土色"]
    },
    {
        "filename": "jihezi.jpg",
        "tags": ["鸡盒子花", "鸡", "盒子", "bird", "box"]
    },
    {
        "filename": "yangque.jpg",
        "tags": ["阳雀花", "阳雀", "鸟", "bird", "totem", "green", "绿色"]
    },
    {
        "filename": "jingoulian.jpg",
        "tags": ["金勾莲", "勾", "hook", "lotus", "flower", "red", "红色"]
    },
    {
        "filename": "48gou-1.jpg",
        "tags": ["四十八勾", "48勾", "hook", "long", "red", "红色"]
    },
    {
        "filename": "nanguahua.jpg",
        "tags": ["南瓜花", "南瓜", "flower", "black", "黑色"]
    },
    {
        "filename": "huwen.jpg",
        "tags": ["台台花", "虎纹", "tiger", "yellow", "黄色"]
    },
    {
        "filename": "48gou-2.jpg",
        "tags": ["四十八勾", "48勾", "hook", "black", "黑色", "土家织锦"]
    },
    {
        "filename": "jiaoshanmei.jpg",
        "tags": ["焦山梅", "梅花", "plum", "flower", "green", "绿色"]
    },
    {
        "filename": "juchihua.jpg",
        "tags": ["锯齿花", "锯齿", "波浪", "zigzag", "sawtooth", "yellow", "黄色"]
    },
    {
        "filename": "yizihua.jpg",
        "tags": ["椅子花", "椅子", "chair", "square", "pink", "粉色"]
    },

    # --- 实物参考图 (注意 ref_02 是 png) ---
    {
        "filename": "ref_01.jpg", 
        "tags": ["六边形", "几何", "hex", "blue", "蓝色", "实物"]
    },
    {
        "filename": "ref_02.png", # ✅ 修正为 .png，对应你的截图
        "tags": ["岩墙花", "菱形", "diamond", "red", "红色", "实物"]
    },
    {
        "filename": "ref_03.jpg", 
        "tags": ["几何", "参考图", "geometric", "yellow", "黄色"]
    },
    {
        "filename": "ref_04.jpg", 
        "tags": ["抱枕", "枕头", "pillow", "yellow", "黄色", "家居"]
    },
    {
        "filename": "ref_05.jpg", 
        "tags": ["挂饰", "长条", "runner", "brown", "土色"]
    }
]

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

    # ✅ 修复了之前的语法错误，确保字符串闭合
    color_rule = (
        "Color Palette: "
        "Background is Deep Indigo Blue (Blackish Blue). "
        "Pattern colors: Bright Madder Red, Golden Yellow, Cyan Blue, White and Black accents. "
        "Solid color blocks, high contrast, no gradients."
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

# ✅ 获取画廊图片接口 (支持搜索)
@app.route('/get_gallery', methods=['GET'])
def get_gallery_images():
    keyword = request.args.get('keyword', '').strip().lower()
    
    images = []
    
    # 遍历档案库
    for item in GALLERY_DB:
        # 检查文件是否真的存在
        if os.path.exists(os.path.join(GALLERY_FOLDER, item['filename'])):
            file_url = f"/static/gallery/{item['filename']}"
            
            # 如果没有搜索词，返回所有图片
            if not keyword:
                images.append(file_url)
            else:
                # 模糊匹配
                is_match = False
                for tag in item['tags']:
                    if keyword in tag.lower():
                        is_match = True
                        break
                if is_match:
                    images.append(file_url)
    
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
