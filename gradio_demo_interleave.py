# Copyright 2025 BAAI. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gradio as gr
import tempfile
import os
import shutil
from model_runtime import ModelRuntime

_RUNTIME = ModelRuntime.instance()

CSS = """
.chatbot {
    max-height: 540px;
}

.chatbot .message.user {
    background: #dff7e6 !important;
    margin-left: auto !important;
    text-align: right !important;
    border-radius: 12px 12px 2px 12px !important;
}

.chatbot .message.assistant {
    background: #eef2ff !important;
    margin-right: auto !important;
    text-align: left !important;
    border-radius: 12px 12px 12px 2px !important;
}

.chatbot .message .label {
    display: none !important;
}
"""

def _dup_path(src: str) -> str:
    """Create a new file to avoid rendering issues caused by reusing the same path in multiple messages."""
    _, ext = os.path.splitext(src)
    tmp = tempfile.NamedTemporaryFile(suffix=ext or ".png", delete=False)
    tmp.close()
    shutil.copyfile(src, tmp.name)
    return os.path.abspath(tmp.name)

def startup_initialize(cfg_path: str, save_dir: str, device_str: str | None = None):
    return _RUNTIME.initialize(cfg_path=cfg_path, save_dir=save_dir, device_str=device_str)

def on_submit(text, files, mode, history):
    _RUNTIME.update_sampling_config(mode)

    sample = {"text": text, "images": [f.name for f in files] if files else []}
    _RUNTIME.encode_and_set_prompt(sample)

    # user message
    if files:
        history.append({"role": "user", "content": text})
        history.append({"role": "user", "content": [f.name for f in files]})
    else:
        history.append({"role": "user", "content": text})
    yield history, "", None, history

    # Placeholder assistant message
    assistant_acc = ""
    history.append({"role": "assistant", "content": assistant_acc})
    yield history, "", None, history

    gen_image_middle_msg = "Generating image... Please wait."
    gen_image_middle_msg_time = 0

    # Streaming
    for ev in _RUNTIME.stream_events(text_chunk_tokens=64):
        if ev["type"] == "text":
            assistant_acc += ev["text"]
            assistant_acc = assistant_acc.replace("<|extra_101|><|extra_204|>", "\n\n**Session End.**")
            assistant_acc = assistant_acc.replace("<|extra_60|>", "**Global Thinking**:\n")
            assistant_acc = assistant_acc.replace("<|extra_61|>", "\n**Thinking End**\n\n")
            assistant_acc = assistant_acc.replace("<|extra_50|>", "\n**Step Thinking**: ")
            assistant_acc = assistant_acc.replace("<|extra_51|>", " **Thinking End**\n")
            history[-1] = {"role": "assistant", "content": assistant_acc}
            yield history, "", None, history
        
        elif ev["type"] == "broken_image":
            if "Generating image... Please wait." not in history[-1]["content"]:
                history.append({"role": "assistant", "content": gen_image_middle_msg})
            else:
                history[-1] = {"role": "assistant", "content": gen_image_middle_msg}

            yield history, "", None, history

            gen_image_middle_msg += '.'
            gen_image_middle_msg_time += 1
            if gen_image_middle_msg_time == 6:
                gen_image_middle_msg = "Generating image... Please wait."
                gen_image_middle_msg_time = 0

        elif ev["type"] == "image":
            for ip in ev.get("paths", []):
                echoed = _dup_path(ip)
                if "Generating image... Please wait." not in history[-1]["content"]:
                    history.append({"role": "assistant", "content": [echoed]})
                else:
                    history[-1] = {"role": "assistant", "content": [echoed]}
                    gen_image_middle_msg = "Generating image... Please wait."
                    gen_image_middle_msg_time = 0
                
                yield history, gr.update(value=None), gr.update(value=None), history

            assistant_acc = ""
            history.append({"role": "assistant", "content": assistant_acc})


def clear_chat():
    _RUNTIME.history.clear()
    return [], []

def on_stop():
    _RUNTIME.request_stop()
    return "🛑 The stop signal has been sent (this round of generation will end and the display will be completed as soon as possible)"

def build_ui():
    with gr.Blocks(css=CSS) as demo:
        gr.Markdown("# Emu 3.5-Interleave Gradio Demo")

        with gr.Row():
            with gr.Column(scale=6):
                chat = gr.Chatbot(
                    label="Chat",
                    height=540,
                    elem_classes="chatbot",
                    type="messages"
                )
                state = gr.State([])

                mode = gr.Dropdown(
                    label="Generation Mode",
                    choices=["howto", "story",],
                    value="howto"
                )

                text = gr.Textbox(label="💬 Prompt", placeholder="Enter your prompt...", lines=2)
                files = gr.Files(label="📷 Upload image(s)", file_count="multiple", type="filepath")

                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    stop = gr.Button("Stop")
                    clear = gr.Button("Clear")
                
                status = gr.Markdown("")

        send.click(on_submit, [text, files, mode, state], [chat, text, files, state])

        stop.click(on_stop, outputs=[status])
        clear.click(clear_chat, outputs=[chat, state])

    return demo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    args.cfg = "configs/example_config_visual_guidance.py"
    args.save_dir = "./outputs"
    args.device = None
    print(startup_initialize(args.cfg, args.save_dir, args.device))
    ui = build_ui()
    ui.queue()
    ui.launch(
        server_name=args.host,
        server_port=args.port,
    )

if __name__ == "__main__":
    main()