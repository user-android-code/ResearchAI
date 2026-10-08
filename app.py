import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # 1. モデルID（またはローカルの保存先パス）の指定
    # ※該当リポジトリには現在tokenizerが同梱されていないため、ベースとなったと推測される
    # 超軽量モデル（例: Qwen/Qwen2.5-0.5B-Instruct や HZ-ZJU/SmolLM2-135M-Instruct など）のトークナイザーを代用するか、
    # もしローカルにダウンロードして使う場合は、ダウンロードしたフォルダのパスを指定してください。
    model_id = "User-Android/AI"
    
    print(f"モデル '{model_id}' を読み込んでいます...")
    
    try:
        # 2. トークナイザーとモデルの読み込み
        # リポジトリにtokenizerの構成ファイルがない場合、AutoTokenizerはエラーになる可能性があります。
        # その場合は `base_model = "Qwen/Qwen2.5-0.5B-Instruct"` などのトークナイザーを別途指定してください。
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True
        )
        
        print("モデルの読み込みが完了しました。チャットを開始します（終了するには 'exit' と入力）。")
        
        # 3. 対話ループ
        while True:
            user_input = input("\nあなた: ")
            if user_input.strip().lower() == "exit":
                print("終了します。")
                break
                
            if not user_input.strip():
                continue
                
            # プロンプトの構築（モデルがInstruct形式のテンプレートを持っている場合）
            # テンプレートがない場合は単にテキストを入力します
            inputs = tokenizer(user_input, return_tensors="pt").to(model.device)
            
            # テキスト生成
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            # 応答のデコードと表示
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            print(f"AI: {response}")
            
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        print("\n[ヒント] 該当のHugging Faceリポジトリには、モデルを動かすために必要な「トークナイザー（tokenizer.jsonなど）」や「設定ファイル（config.json）」が含まれていないようです。")
        print("このコードを動かすには、ベースになった元のモデルのトークナイザーを代わりに指定するか、リポジトリにファイルが揃うのを待つ必要があります。")

if __name__ == "__main__":
    main()
