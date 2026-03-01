import gradio as gr
from openai import OpenAI
import time

client = OpenAI(
    base_url="http://localhost:12434/v1",
    api_key="dummy-key",
)

MODEL = "hf.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF"

def chat_fn(message, history, system_prompt, temperature, max_tokens):
    messages = [{"role": "system", "content": system_prompt}]

    for item in history:
        messages.append({"role": item["role"], "content": item["content"]})

    messages.append({"role": "user", "content": message})

    start = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=int(max_tokens),
        temperature=float(temperature),
    )
    elapsed = time.time() - start
    tokens = resp.usage.completion_tokens
    reply = resp.choices[0].message.content
    reply += (
        f"\n\n---\n*{tokens} tokens · "
        f"{tokens/elapsed:.1f} tok/s · "
        f"M3 Pro · Docker Model Runner*"
    )
    return reply

with gr.Blocks(title="Mistral-7B · Docker Model Runner") as demo:
    gr.Markdown("""
    # Mistral-7B Local Inference
    **Running on Apple Silicon via Docker Model Runner + vLLM-Metal**
    """)

    gr.ChatInterface(
        fn=chat_fn,
        chatbot=gr.Chatbot(height=500),
        additional_inputs=[
            gr.Textbox(
                value="You are a helpful assistant.",
                label="System Prompt"
            ),
            gr.Slider(0.0, 1.0, value=0.3, step=0.1, label="Temperature"),
            gr.Slider(64, 1024, value=512, step=64, label="Max Tokens"),
        ],
        # each example = [message, system_prompt, temperature, max_tokens]
        examples=[
            ["Explain Kubernetes NetworkPolicies in simple terms.", "You are a helpful assistant.", 0.3, 512],
            ["Write a Python function to parse JSON safely.", "You are a helpful assistant.", 0.3, 512],
            ["What is vLLM and why does it matter?", "You are a helpful assistant.", 0.3, 512],
            ["Compare REST and GraphQL APIs.", "You are a helpful assistant.", 0.3, 512],
        ],
    )

demo.launch(theme=gr.themes.Soft(), share=False)
