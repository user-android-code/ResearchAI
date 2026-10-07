import streamlit as st
import json
from huggingface_hub import hf_hub_download
from safetensors import safe_open

st.set_page_config(page_title="HF Model Loader", page_icon="🤗", layout="wide")

st.title("🤗 Hugging Faceモデル内部メタデータ解析器")
st.write("`model.safetensors` の内部に隠されている設定（config）を読み取ります。")

# 入力欄
repo_id = st.text_input("Hugging Face リポジトリID", value="User-Android/AI")
filename = st.text_input("ファイル名", value="model.safetensors")

if st.button("モデルの内部を解析する"):
    with st.spinner("⏳ モデルファイルをダウンロードして解析中... (初回は時間がかかります)"):
        try:
            # 1. ファイルのダウンロード
            file_path = hf_hub_download(repo_id=repo_id, filename=filename)
            
            # 2. safetensorsの内部メタデータを取得
            with safe_open(file_path, framework="pt") as f:
                metadata = f.metadata()
            
            # 3. 結果の表示
            if metadata:
                st.success("✅ ファイル内部から設定データ（メタデータ）を検出しました！")
                
                # メタデータの中身を綺麗に表示
                st.subheader("📊 検出された設定情報 (Metadata)")
                
                # もし文字列のJSONが入っていればパースを試みる
                parsed_metadata = {}
                for k, v in metadata.items():
                    try:
                        parsed_metadata[k] = json.loads(v)
                    except:
                        parsed_metadata[k] = v
                
                st.json(parsed_metadata)
                
                # 推定されるアーキテクチャのヒント
                st.info("💡 この設定情報を元に、transformersやllama.cppなどのライブラリに読み込ませてテキスト生成を実行させることができます。")
                
            else:
                st.warning("⚠️ ファイルは読み込めましたが、内部にメタデータ（config）は埋め込まれていませんでした。")
                st.write("通常の重みデータのみの可能性があります。")
                
        except Exception as e:
            st.error("❌ 解析中にエラーが発生しました")
            st.code(str(e))
