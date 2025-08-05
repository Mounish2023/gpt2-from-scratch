import torch
import tiktoken
from model import GPT, GPTConfig
import gradio as gr

ckpt_path = "model_19072.pt"  # adjust if needed

# If you hit the PyTorch 2.6 "weights_only" safety error, enable the allowlist:
# from torch.serialization import safe_globals
# with safe_globals({'GPTConfig': GPTConfig}):
#     checkpoint = torch.load(ckpt_path, map_location='cpu')
# Otherwise, this is fine if you trust your own checkpoint:
checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

# Rebuild model from saved config (important: your train.py used vocab_size=50304)
config = checkpoint['config']          # this is your GPTConfig dataclass
model = GPT(config)

# Load weights. Missing attn.bias buffers are OK (they're re-created on init).
state_dict = checkpoint['model']
# (Optional) If you had saved a wrapped DDP model by mistake, strip prefixes:
# state_dict = {k.replace('module.', '', 1) if k.startswith('module.') else k: v
#               for k, v in state_dict.items()}
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("missing:", missing)       # expect things like *.attn.bias (buffers)
print("unexpected:", unexpected) # usually []

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device).eval()

# -------- simple text generation (sampling) ----------
enc = tiktoken.get_encoding("gpt2")

@torch.no_grad()
def generate(model, prompt, max_new_tokens=100, temperature=1.0, top_k=None, device=None):
    device = device or next(model.parameters()).device
    ids = torch.tensor(enc.encode(prompt), dtype=torch.long, device=device)[None, ...]  # (1, T)
    for _ in range(max_new_tokens):
        logits, _ = model(ids)               # (1, T, vocab)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('inf')
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)  # (1, 1)
        ids = torch.cat([ids, next_id], dim=1)
    return enc.decode(ids[0].tolist())

# 3. Create the Gradio interface
iface = gr.Interface(
    fn=generate,
    inputs=[
        gr.components.Textbox(lines=3, label="Prompt"),
        gr.components.Slider(10, 512, step=10, value=100, label="Max New Tokens")
    ],
    outputs="text",
    title="Text Generation Model"
)

if __name__ == "__main__":
    iface.launch()
