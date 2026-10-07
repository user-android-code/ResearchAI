import streamlit as st
import json
from huggingface_hub import hf_hub_download
from safetensors import safe_open

st.set_page_config(page_title="HF Model Loader", page_icon="🤗", layout="wide")

st.title("🤗 Hugging Faceモデル内部構造 解析器")
st.write("`model.safetensors` の内部にある重みの変数名（キー名）を一覧表示して、モデルの正体を突き止めます。")

# 入力欄
repo_id = st.text_input("Hugging Face リポジトリID", value="User-Android/AI")
filename = st.text_input("ファイル名", value="model.safetensors")

if st.button("モデルの内部（キー名）を解析する"):
    with st.spinner("⏳ モデルファイルをダウンロードして解析中..."):
        try:
            # 1. ファイルのダウンロード
            file_path = hf_hub_download(repo_id=repo_id, filename=filename)
            
            # 2. safetensorsを開いてキー（テンソル名）の一覧を取得
            with safe_open(file_path, framework="pt") as f:
                tensor_keys = f.keys()
            
            st.success(f"✅ ファイルの解析に成功しました！ 内包されている変数の数: {len(tensor_keys)}個")
            
            # 3. LoRAかどうかの簡易判定
            is_lora = any("lora" in k.lower() for k in tensor_keys)
            if is_lora:
                st.info("💡 判定結果: 変数名に 'lora' が含まれています。ベースモデル（親）に被せて使う【LoRAアダプター】の可能性が高いです。")
            else:
                st.warning("💡 判定結果: 'lora' という文字は見当たりません。独自の自作モデルか、特殊な形式の可能性があります。")
            
            # 4. キー名の一覧を表示
            st.subheader("📊 内部の変数名（キー）一覧（先頭50個）")
            st.write("これらの名前のパターンから、ベースになっているAIの種類を推測できます。")
            
            # リスト形式で表示
            keys_to_show = list(tensor_keys)[:50]
            for k in keys_to_show:
                st.code(k)
                
            if len(tensor_keys) > 50:
                st.caption(f"...残り {len(tensor_keys) - 50} 個の変数は省略しました。")
                
        except Exception as e:
            st.error("❌ 解析中にエラーが発生しました")
            st.code(str(e))
