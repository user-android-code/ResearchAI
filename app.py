import streamlit as st
import cv2
import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from PIL import Image

# ページの設定
st.set_page_config(page_title="AnimeGANv2 ジブリ風変換", layout="centered")
st.title("🎨 AnimeGANv2 (Hayao) 画像アニメ化ツール")
st.write("画像をアップロードすると、宮崎駿（ジブリ）監督風のタッチに変換します。")

# 1. Hugging FaceからONNXモデルをダウンロードしてセッションをキャッシュ
@st.cache_resource
def load_onnx_model():
    model_path = hf_hub_download(
        repo_id="IndianBeeR/AnimeGANv2_Hayao", 
        filename="AnimeGANv2_Hayao.onnx"
    )
    # 推論セッションの作成
    return ort.InferenceSession(model_path)

try:
    session = load_onnx_model()
except Exception as e:
    st.error(f"モデルの読み込み中にエラーが発生しました: {e}")
    st.stop()

# 2. 前処理関数の定義（エラー原因の修正対応）
def preprocess(cv_image, target_size=(512, 512)):
    # 1. 512x512 にリサイズ
    img = cv2.resize(cv_image, target_size)
    # 2. BGR から RGB へ変換
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    # 3. モデルの期待する範囲 (-1.0 〜 1.0) に正規化
    img = (img / 127.5) - 1.0
    
    # 【重要・エラー修正】
    # エラー画像では index: 3 Got: 512 Expected: 3 となっています。
    # つまり、モデルは [バッチサイズ, 高さ, 幅, チャンネル]、すなわち [1, 512, 512, 3] を期待しています。
    # 軸を入れ替えず、そのまま先頭にバッチ次元を追加します。
    img = np.expand_dims(img, axis=0) # 結果: [1, 512, 512, 3]
    return img

# 3. 後処理関数の定義
def postprocess(output_tensor):
    # バッチ次元を削除 [1, 512, 512, 3] -> [512, 512, 3]
    img = np.squeeze(output_tensor, axis=0)
    # 画素値を 0 〜 255 に復元
    img = (img + 1.0) * 127.5
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img

# 4. Streamlit UI: ファイルアップローダー
uploaded_file = st.file_uploader("画像ファイルを選択してください (png, jpg, jpeg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # アップロードされた画像を OpenCV 形式に変換
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_img = cv2.imdecode(file_bytes, 1)
    
    # 画面に元の画像を表示
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("元の画像")
        st.image(cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
    # 変換ボタン
    if st.button("✨ アニメ風に変換する"):
        with st.spinner("変換中..."):
            try:
                # 前処理
                input_data = preprocess(opencv_img)
                
                # ONNXモデルの入力名と出力名を取得
                input_name = session.get_inputs().name
                output_name = session.get_outputs().name
                
                # 推論実行
                raw_result = session.run([output_name], {input_name: input_data})
                
                # 後処理
                anime_img = postprocess(raw_result[0])
                
                # 結果の表示
                with col2:
                    st.subheader("アニメ化画像")
                    st.image(anime_img, use_container_width=True)
                st.success("変換が完了しました！")
                
            except Exception as e:
                st.error(f"変換中にエラーが発生しました:\n{str(e)}")
