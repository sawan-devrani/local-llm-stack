from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
import time

client = OpenAI(
    base_url="http://localhost:12434/v1",
    api_key="dummy-key",
)

MODEL = "hf.co/bartowski/Mistral-7B-Instruct-v0.3-GGUF"
console = Console()

def interactive():
    console.print("[bold]Local Mistral-7B Chat[/bold] (type 'quit' to exit)\n")
    history = []

    while True:
        prompt = input("You: ").strip()
        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue

        messages = [{"role": "system", "content": "You are a helpful assistant."}]
        for user, assistant in history:
            messages.append({"role": "user", "content": user})
            messages.append({"role": "assistant", "content": assistant})
        messages.append({"role": "user", "content": prompt})

        start = time.time()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=512,
            temperature=0.3,
        )
        elapsed = time.time() - start
        reply = resp.choices[0].message.content
        tokens = resp.usage.completion_tokens

        console.print(f"\n[bold green]Assistant:[/bold green]")
        console.print(Markdown(reply))
        console.print(
            f"[dim]── {tokens} tokens · {elapsed:.2f}s · "
            f"{tokens/elapsed:.1f} tok/s · M3 Pro[/dim]\n"
        )
        history.append((prompt, reply))

if __name__ == "__main__":
    interactive()
