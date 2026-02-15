# MoE Router Visualizer

Shows which experts fire for each token during generation. Type a prompt, hit enter, watch the routing happen live.

![screenshot](../router-vis.png)

## What it does

Trinity Nano has 128 routed experts per MoE layer (top-8 active at a time, plus a shared expert). This hooks into a layer's router gate during inference and streams the expert selections + scores back to the browser as each token is generated.

The circle is the 128 experts. Blue = active right now. Purple = cumulative heat across the whole response. Click any token on the right to see its routing.

## Running it

```
uv run router_vis/app.py
```

Opens on `localhost:5001`. Needs ~12GB VRAM (bf16 model).

You can pick which layer to watch and set max tokens from the bottom bar. Arrow keys step through tokens, Escape stops generation.

## Files

- `app.py` — Flask server, model loading, router hooking, SSE streaming
- `templates/index.html` — everything frontend (canvas viz, token display, controls)
