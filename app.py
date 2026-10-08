import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

st.set_page_config(page_title="Qens Chat App", layout="centered")
st.title("🤖 Qens モデル チャットアプリ")
st.write("Hugging Faceの `User-Android/Qens` モデルをStreamlitで動かすデモアプリです。")

# 1. モデルとトークナイザーの読み込み（初回のみキャッシュする）
@st.cache_resource
def load_model():
    model_id = "User-Android/Qens"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # 536MBと軽量なため、GPUがない環境（CPU）でも動作が期待できます
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=dtype,
        device_map=device_map
    )
    return tokenizer, model

try:
    with st.spinner("Hugging Faceからモデルを読み込んでいます（初回はダウンロードに数分かかります）..."):
        tokenizer, model = load_model()
    st.success("モデルの読み込みが完了しました！")
except Exception as e:
    st.error(f"モデルの読み込み中にエラーが発生しました: {e}")
    st.stop()

# 2. ユーザー入力の受付
user_input = st.text_area("質問や指示を入力してください:", value="こんにちは、自己紹介をしてください。", height=100)

# 生成パラメーターの調整（サイドバー）
st.sidebar.header("生成パラメーター")
max_tokens = st.sidebar.slider("最大生成トークン数", min_value=10, max_value=512, value=128, step=10)
temperature = st.sidebar.slider("Temperature (温度)", min_value=0.1, max_value=1.5, value=0.7, step=0.1)
top_p = st.sidebar.slider("Top-p (核サンプリング)", min_value=0.1, max_value=1.0, value=0.9, step=0.05)

if st.button("文章を生成する", type="primary"):
    if user_input.strip():
        with st.spinner("文章を生成中..."):
            try:
                # 入力テキストをトークンに変換
                inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
                
                # テキスト生成
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=max_tokens,
                        do_sample=True,
                        temperature=temperature,
                        top_p=top_p,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                # 結果をデコードして表示
                result = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                st.subheader("生成結果:")
                st.info(result)
                
            except Exception as e:
                st.error(f"生成中にエラーが発生しました: {e}")
    else:
        st.warning("テキストを入力してください。")
