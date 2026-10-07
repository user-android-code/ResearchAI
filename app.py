import streamlit as st
import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import Qwen2Config, Qwen2ForCausalLM, AutoTokenizer

st.set_page_config(page_title="Custom LLM Chat", page_icon="🤖", layout="wide")

st.title("🤖 自作超小型LLM テキスト生成アプリ (修正版)")

REPO_ID = "User-Android/AI"
FILENAME = "model.safetensors"

@st.cache_resource
def load_custom_model():
    file_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    weights = load_file(file_path)
    
    # 【修正箇所】[0] や [1] を指定して、正確な数値（int）として取り出します
    vocab_size = weights["model.embed_tokens.weight"].shape[0]
    hidden_size = weights["model.embed_tokens.weight"].shape[1]
    intermediate_size = weights["model.layers.0.mlp.up_proj.weight"].shape[0]
    
    config = Qwen2Config(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=6,  # 変数一覧から全6レイヤーと判定
        num_attention_heads=16,
        num_key_value_heads=16,
        hidden_act="silu",
        max_position_embeddings=2048,
        use_cache=True,
        tie_word_embeddings=False
    )
    
    model = Qwen2ForCausalLM(config)
    model.load_state_dict(weights, strict=True)
    model.eval()
    
    # 互換性のあるQwen2のトークナイザーを読み込み
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    
    return model, tokenizer, vocab_size

try:
    model, tokenizer, model_vocab_size = load_custom_model()
    
    # 画面に現在の語彙数ステータスを表示
    st.success("🎉 モデルの構造解析と読み込みに成功しました！")
    st.metric(label="モデルの実際の語彙数 (vocab_size)", value=model_vocab_size)
    st.write(f"※使用中のトークナイザーの語彙数: {tokenizer.vocab_size}")
    
except Exception as e:
    st.error("❌ モデルの読み込み中にエラーが発生しました。")
    st.code(str(e))
    st.stop()

# --- UI部分 ---
st.subheader("✍️ 文章を生成させてみる")
prompt = st.text_area("AIへの指示（プロンプト）を入力してください：", value="昔々、あるところに")

if st.button("テキストを生成する"):
    with st.spinner("🤖 思考中..."):
        # 1. テキストをトークンIDに変換
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # モデルの最大語彙数を超えるIDを安全なID(0)に置き換える（エラー防止）
        inputs["input_ids"] = torch.where(
            inputs["input_ids"] >= model_vocab_size, 
            torch.tensor(0, device=inputs["input_ids"].device), 
            inputs["input_ids"]
        )
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].to(inputs["input_ids"].device)

        try:
            # 2. 生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # 3. 画面に表示
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.subheader("📄 生成された結果:")
            st.info(generated_text)
            
        except Exception as gen_e:
            st.error("❌ 生成処理中にエラーが発生しました。")
            st.code(str(gen_e))
