import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import Qwen2Config, Qwen2ForCausalLM, AutoTokenizer

st.set_page_config(page_title="Custom LLM Chat", page_icon="🤖", layout="wide")

st.title("🤖 自作超小型LLM テキスト生成アプリ")
st.write("`model.safetensors` からモデル構造を自動計算し、文章生成を行います。")

# 1. ハブからダウンロードの設定
REPO_ID = "User-Android/AI"
FILENAME = "model.safetensors"

# キャッシュを使ってモデルの読み込みを1回だけにする
@st.cache_resource
def load_custom_model():
    # ファイルのローカルダウンロード
    file_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    weights = load_file(file_path)
    
    # 重みの形状（Shape）から設計図（Config）のパラメータを自動抽出
    vocab_size, hidden_size = weights["model.embed_tokens.weight"].shape
    intermediate_size = weights["model.layers.0.mlp.up_proj.weight"].shape[0]
    
    # アテンションヘッド数の推測 (通常 hidden_size / 64 or 128)
    # ここでは一般的な16ヘッドとして仮定（ズレていればエラーログから微調整可能）
    num_heads = 16 
    num_kv_heads = 16
    
    # 6レイヤーのQwen2Configを手動で組み立てる
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=6,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False
    )
    
    # 空のモデルを作成して重みを流し込む
    model = Qwen2ForCausalLM(config)
    model.load_state_dict(weights, strict=True)
    model.eval()
    
    # ※トークナイザー（辞書）がリポジトリにないため、
    # 互換性のあるQwen2の標準トークナイザーを代用します
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    return model, tokenizer

# モデルのロード実行
try:
    model, tokenizer = load_custom_model()
    st.success("🎉 モデルの構造解析と読み込みに成功しました！単体で文章生成が可能です。")
except Exception as e:
    st.error("❌ モデルの読み込み中にエラーが発生しました。")
    st.code(str(e))
    st.stop()

# --- UI部分：テキスト生成テスト ---
st.subheader("✍️ 文章を生成させてみる")
prompt = st.text_area("AIへの指示（プロンプト）を入力してください：", value="昔々、あるところに")

col1, col2 = st.columns(2)
with col1:
    max_tokens = st.slider("生成する最大の長さ", min_value=10, max_value=200, value=50)
with col2:
    temperature = st.slider("ランダム度合い (Temperature)", min_value=0.1, max_value=1.5, value=0.7)

if st.button("テキストを生成する"):
    with st.spinner("🤖 思考中..."):
        # 入力テキストをトークンに変換
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # デコードして画面に表示
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        st.subheader("📄 生成された結果:")
        st.info(generated_text)
