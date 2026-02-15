"""
MoE Router Visualizer — streaming expert routing per token.

Loads Trinity Nano, generates token-by-token, hooks the router gate
at the selected layer to capture which experts are active + their scores.
Streams results via SSE to the frontend.

Usage:
    uv run router_vis/app.py
"""

import os
import sys
import json
import time
import gc
import threading

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from flask import Flask, request, render_template, Response
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
MODEL_NAME = "arcee-ai/Trinity-Nano-Preview"

app = Flask(
    __name__,
    template_folder=str(APP_DIR / "templates"),
)

# Global state
MODEL = None
TOKENIZER = None
LOCK = threading.Lock()


def load_model():
    global MODEL, TOKENIZER
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading tokenizer...")
    TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    print(f"Loading {MODEL_NAME} (bf16)...")
    t0 = time.perf_counter()
    MODEL = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    MODEL.eval()
    MODEL = MODEL.to("cuda")
    gc.collect()
    torch.cuda.empty_cache()
    vram = torch.cuda.memory_allocated() / 1024**3
    print(f"  Model loaded in {time.perf_counter()-t0:.1f}s ({vram:.1f}GB VRAM)")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/generate", methods=["POST"])
def generate():
    body = request.get_json(force=True)
    text = body.get("text", "").strip()
    layer = int(body.get("layer", 42))
    max_tokens = min(int(body.get("max_tokens", 64)), 512)

    if not text:
        return Response("data: " + json.dumps({"type": "error", "message": "No text"}) + "\n\n",
                        content_type="text/event-stream")

    def generate_stream():
        with LOCK:
            # Tokenize with chat template
            messages = [{"role": "user", "content": text}]
            chat_text = TOKENIZER.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            input_ids = TOKENIZER.encode(chat_text, add_special_tokens=False, return_tensors="pt").to("cuda")

            # Send prompt tokens
            prompt_tokens = []
            for tid in input_ids[0].tolist():
                prompt_tokens.append(TOKENIZER.decode([tid]))
            yield f"data: {json.dumps({'type': 'prompt_tokens', 'tokens': prompt_tokens})}\n\n"

            # Manual decode loop with router hook
            past = None
            current_ids = input_ids

            for step in range(max_tokens):
                # Set up router hook for the target layer
                router_data = {}

                def router_hook(module, inp, output):
                    # output is (top_scores, selected_experts)
                    # top_scores: (batch*seq, 8), selected_experts: (batch*seq, 8)
                    top_scores, selected_experts = output
                    # Take last token only (for generation)
                    router_data["scores"] = top_scores[-1].detach().float().cpu()
                    router_data["experts"] = selected_experts[-1].detach().cpu()
                    return output

                router_module = MODEL.model.layers[layer].mlp.router
                handle = router_module.register_forward_hook(router_hook)

                try:
                    with torch.no_grad():
                        out = MODEL(
                            input_ids=current_ids,
                            past_key_values=past,
                            use_cache=True,
                        )
                    past = out.past_key_values
                    logits = out.logits[:, -1, :]

                    # Greedy decode (simple for viz)
                    next_token = logits.argmax(dim=-1)
                    token_id = next_token.item()
                    token_text = TOKENIZER.decode([token_id])

                finally:
                    handle.remove()

                # Build expert data from hook
                experts = []
                if "scores" in router_data:
                    scores = router_data["scores"].tolist()
                    expert_ids = router_data["experts"].tolist()
                    for eid, score in zip(expert_ids, scores):
                        experts.append({"id": int(eid), "score": round(float(score), 4)})
                    # Sort by score descending
                    experts.sort(key=lambda e: -e["score"])

                yield f"data: {json.dumps({'type': 'token', 'token': token_text, 'experts': experts})}\n\n"

                # Check for EOS
                if token_id == TOKENIZER.eos_token_id:
                    break

                current_ids = next_token.view(1, 1)

            yield "data: [DONE]\n\n"

    return Response(generate_stream(), content_type="text/event-stream")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=5001)
    p.add_argument("--host", default="127.0.0.1")
    args = p.parse_args()

    load_model()
    print(f"\nRouter visualizer at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
