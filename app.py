import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image
import os

# ページの設定
st.set_page_config(page_title="AnimeGANv2 Hayao Converter", layout="centered")
st.title("🎨 AnimeGANv2 (Hayao) Converter")
st.write("画像をアップロードするか、同じフォルダに `image.png` を配置すると宮崎駿（ジブリ）風のアニメ調に変換します。")

# 1. Hugging FaceからONNXモデルをダウンロード（キャッシュされます）
@st.cache_resource
def load_onnx_model():
    model_path = hf_hub_download(
        repo_id="IndianBeeR/AnimeGANv2_Hayao", 
        filename="AnimeGANv2_Hayao.onnx"
    )
    return ort.InferenceSession(model_path)

try:
    session = load_onnx_model()
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました: {e}")
    st.stop()

# 前処理関数
def preprocess(cv_img, target_size=(512, 512)):
    # BGRからRGBへ
    img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32)
    # モデルの期待する範囲 (-1.0 〜 1.0) に正規化
    img = (img / 127.5) - 1.0
    # [H, W, C] から [1, C, H, W] へ軸を変換
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# 後処理関数
def postprocess(output_tensor):
    # [1, C, H, W] から [H, W, C] に戻す
    img = np.squeeze(output_tensor, axis=0)
    img = np.transpose(img, (1, 2, 0))
    # 元の画素値の範囲 (0 〜 255) に復元
    img = (img + 1.0) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# 画像ソースの決定（アップローダー優先、なければ image.png）
uploaded_file = st.file_uploader("画像をアップロードしてください", type=["png", "jpg", "jpeg"])
image_path = "image.png"

cv_input_image = None

if uploaded_file is not None:
    # アップロードされた画像を読み込み
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    cv_input_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
elif os.path.exists(image_path):
    # ローカルの image.png を読み込み
    cv_input_image = cv2.imread(image_path)

# 画像が読み込めた場合の処理
if cv_input_image is not None:
    # 画面に2列で表示
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("元画像")
        st.image(cv2.cvtColor(cv_input_image, cv2.COLOR_BGR2RGB), use_container_width=True)
    
    with col2:
        st.subheader("アニメ化画像")
        with st.spinner("アニメ化処理中..."):
            try:
                # 前処理・推論・後処理
                input_data = preprocess(cv_input_image)
                input_name = session.get_inputs()[0].name
                output_name = session.get_outputs()[0].name
                raw_result = session.run([output_name], {input_name: input_data})
                
                anime_img = postprocess(raw_result[0])
                
                # 表示
                st.image(anime_img, use_container_width=True)
                
                # ダウンロードボタン用
                result_pil = Image.fromarray(anime_img)
                # バイトデータに変換
                import io
                buf = io.BytesIO()
                result_pil.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="画像をダウンロード",
                    data=byte_im,
                    file_name="anime_style.png",
                    mime="image/png"
                )
            except Exception as e:
                st.error(f"変換中にエラーが発生しました: {e}")
else:
    st.info("画像をアップロードするか、スクリプトと同じフォルダに `image.png` を配置してください。")
