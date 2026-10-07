import streamlit as st
from huggingface_hub import list_repo_files

# ページのタイトル設定
st.set_page_config(page_title="Hugging Face リポジトリチェッカー", page_icon="🤗")

st.title("🤗 Hugging Face リポジトリチェッカー")
st.write("指定したリポジトリに `config.json` があるか、どんなファイルがあるかを確認します。")

# 入力フォーム（デフォルトで今回のリポジトリを入力）
repo_id = st.text_input("Hugging Face リポジトリID", value="User-Android/AI")

# 確認ボタン
if st.button("ファイルをチェックする"):
    if not repo_id:
        st.warning("リポジトリIDを入力してください。")
    else:
        with st.spinner("🔍 Hugging Faceからファイル一覧を取得中..."):
            try:
                # ファイル一覧を取得
                file_list = list_repo_files(repo_id=repo_id)
                
                # config.json の有無を判定
                if "config.json" in file_list:
                    st.success("✅ **'config.json' が見つかりました！**")
                    st.info("このモデルは通常の `transformers` ライブラリなどで直接読み込める可能性が高いです。")
                else:
                    st.error("❌ **'config.json' は見つかりませんでした。**")
                    st.warning("設定ファイルがないため、親モデルを指定するLoRAなどの可能性があります。")
                
                # ファイル一覧を綺麗に表示
                st.subheader("📄 リポジトリ内のファイル一覧")
                # マークダウンのリスト形式に変換して表示
                files_markdown = "\n".join([f"- `{file}`" for file in file_list])
                st.markdown(files_markdown)
                
            except Exception as e:
                st.error("❌ エラーが発生しました")
                st.code(str(e))
                st.caption("リポジトリ名が正しいか、または公開（Public）リポジトリであるか確認してください。")
