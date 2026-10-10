import streamlit as st
from diffusers import StableDiffusionImg2ImgPipeline  # または DiffusionPipeline
import torch
from PIL import Image

# ページのタイトル設定
st.title("🎨 imagen アニメ化アプリケーション")
st.write("`User-Android/imagen` モデルを直接使用して、画像をアニメ化します。")

# モデルの読み込み（キャッシュ化）
@st.cache_resource
def load_pipeline():
    # 今回はこのモデル自体が本体なので、これだけを読み込みます
    model_id = "User-Android/imagen"
    
    # デバイスの自動判定
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
    elif torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32

    # モデルをImg2Img用のパイプラインとして直接ロード
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    return pipe

# モデルのロード
with st.spinner("AIモデル（imagen）を準備しています..."):
    try:
        pipe = load_pipeline()
        st.success("AIモデルの準備が完了しました！")
    except Exception as e:
        st.error(f"モデルの読み込みに失敗しました。理由: {e}")
        st.stop()

# ユーザーからの画像アップロード受付
uploaded_file = st.file_uploader("アニメ化したい画像を選択してください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # アップロードされた画像を開いてRGB形式に変換
    init_image = Image.open(uploaded_file).convert("RGB")
    
    # 画像サイズをAIが処理しやすいサイズ（512x512基準）に調整
    init_image.thumbnail((512, 512))
    
    # 画面に元の画像を表示
    st.image(init_image, caption="元の画像", use_container_width=True)
    
    st.write("---")
    
    # 変換の強さ（どれくらい元画像から変化させるか）
    strength = st.slider("変化の強さ (Strength)", min_value=0.1, max_value=0.9, value=0.6, step=0.05)

    # 変換ボタン
    if st.button("✨ 画像をアニメ化する"):
        with st.spinner("imagenモデルで画像を変換中..."):
            try:
                # プロンプトなし（空欄）で、入力画像だけを渡して実行
                output = pipe(
                    prompt="", 
                    image=init_image, 
                    strength=strength,
                    num_inference_steps=30
                )
                
                # 生成されたアニメ画像
                anime_image = output.images[0]
                
                # 結果の表示
                st.subheader("🎉 アニメ化結果")
                st.image(anime_image, caption="アニメ化された画像", use_container_width=True)
                
            except Exception as e:
                st.error(f"変換中にエラーが発生しました: {e}")
