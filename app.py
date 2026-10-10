import streamlit as st
from diffusers import DiffusionPipeline  # 汎用的なパイプラインに変更
import torch

# ページのタイトル設定
st.title("🎨 画像生成AI 検証アプリケーション")
st.write("モデル `User-Android/imagen` の出力をプロンプトなしで検証します。")

# モデルの読み込み（高速化のためにキャッシュ化）
@st.cache_resource
def load_pipeline():
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

    # テキスト不要の汎用的なパイプラインとして読み込みを試行
    pipe = DiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch_dtype,
        use_safetensors=True
    )
    pipe = pipe.to(device)
    return pipe

# モデルのロード
with st.spinner("モデルを読み込んでいます...初回はダウンロードに時間がかかります。"):
    try:
        pipe = load_pipeline()
        st.success("モデルの読み込みに完了しました！")
    except Exception as e:
        st.error(f"モデルの読み込みに失敗しました。このリポジトリの構造が特殊な可能性があります。エラー: {e}")
        st.stop()

# 生成ボタン
submit_button = st.button("プロンプトなしで画像をテスト生成する")

if submit_button:
    with st.spinner("画像を生成中..."):
        try:
            # プロンプト（引数）を何も渡さずにモデルを実行
            # （Unconditionalモデルであれば、これだけでランダムな画像が生成されます）
            output = pipe()
            
            # 生成された画像を取得
            image = output.images[0]
            
            # 画面に画像を表示
            st.image(image, caption="生成されたテスト画像", use_container_width=True)
            
        except Exception as e:
            st.error(f"画像の生成中にエラーが発生しました。\n"
                     f"このモデルはテキストではなく『画像』を入力する必要があるか、"
                     f"あるいはモデルのファイルが未完成な可能性があります。\n"
                     f"エラー詳細: {e}")
