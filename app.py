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

# ================= 核心逻辑：图片档案库 (已匹配你的文件名) =================
GALLERY_DB = [
    # --- 经典图谱 (带文字说明的图) ---
    {
        "filename": "8gou.jpg",
        "tags": ["八勾纹", "八勾", "8 hook", "brown", "土色", "几何"]
    },
    {
        "filename": "jihezi.jpg",
        "tags": ["鸡盒子花", "鸡", "盒子", "bird", "box", "彩色"]
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
        "filename": "nanguahua.jpg",  # 截图文件名为 nanguahua.jpg
        "tags": ["南瓜花", "南瓜", "flower", "black", "黑色", "植物"]
    },
    {
        "filename": "jiaoshanmei.jpg",
        "tags": ["焦山梅", "梅花", "plum", "flower", "green", "绿色"]
    },
    {
        "filename": "juchihua.jpg",   # 截图文件名为 juchihua.jpg
        "tags": ["锯齿花", "锯齿", "波浪", "zigzag", "sawtooth", "yellow", "黄色"]
    },
    {
        "filename": "huwen.jpg",
        "tags": ["台台花", "虎纹", "tiger", "yellow", "黄色", "动物"]
    },
    {
        "filename": "yizihua.jpg",
        "tags": ["椅子花", "椅子", "chair", "square", "pink", "粉色"]
    },
    
    # --- 四十八勾纹 (两张) ---
    {
        "filename": "48gou-1.jpg",
        "tags": ["四十八勾", "48勾", "hook", "long", "red", "红色"]
    },
    {
        "filename": "48gou-2.jpg",
        "tags": ["四十八勾", "48勾", "hook", "black", "黑色", "土家织锦"]
    },

    # --- 实物参考图 (ref系列) ---
    {
        "filename": "ref_01.jpg", # 蓝底六边形
        "tags": ["六边形", "几何", "hex", "blue", "蓝色", "实物"]
    },
    {
        "filename": "ref_02.jpg", # 红底大菱形
        "tags": ["岩墙花", "菱形", "diamond", "red", "红色", "实物"]
    },
    {
        "filename": "ref_03.jpg", # 黄底几何
        "tags": ["几何", "参考图", "geometric", "yellow", "黄色"]
    },
    {
        "filename": "ref_04.jpg", # 黄色抱枕
        "tags": ["抱枕", "枕头", "pillow", "yellow", "黄色", "家居"]
    },
    {
        "filename": "ref_05.jpg", # 长条挂饰
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

    # 针对“岩墙花”的特殊修正
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
        "Solid color blocks, high contrast, no gradients.
