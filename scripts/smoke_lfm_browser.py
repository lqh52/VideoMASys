"""Opt-in real-weight smoke test of the browser's LFM adapter, using CPU ORT.

Install playwright, huggingface_hub, numpy, onnxruntime into a dev environment.
Downloads ~2.34 GB into the Hugging Face cache. No server GPU is used.
The actual JS processor, prompt template, and generation adapter run in Chromium;
only ONNX session execution is bridged to Python because headless WebGPU may be
unavailable. This checks model correctness, NOT browser GPU speed/compatibility.
Run with the web app already serving: python scripts/smoke_lfm_browser.py
"""

import argparse
import base64
import json
import time

import numpy as np
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from playwright.sync_api import sync_playwright

MODEL = "LiquidAI/LFM2.5-VL-1.6B-ONNX"
FILES = ["embed_tokens_fp16", "embed_images_fp16", "decoder_q4"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8200")
    parser.add_argument("--tokens", type=int, default=32)
    args = parser.parse_args()
    sessions = []
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    for name in FILES:
        print(f"Loading actual weights: {name}", flush=True)
        graph = hf_hub_download(MODEL, f"onnx/{name}.onnx")
        hf_hub_download(MODEL, f"onnx/{name}.onnx_data")
        sessions.append(ort.InferenceSession(graph, options, providers=["CPUExecutionProvider"]))

    def run_graph(payload):
        session = sessions[payload["session"]]
        feeds = {
            name: np.frombuffer(base64.b64decode(v["data"]), dtype=v["type"]).reshape(v["dims"])
            for name, v in payload["feeds"].items()
        }
        result = session.run(None, feeds)
        return {
            info.name: {"type": str(value.dtype), "dims": list(value.shape),
                        "data": base64.b64encode(value.tobytes()).decode()}
            for info, value in zip(session.get_outputs(), result, strict=True)
        }

    with sync_playwright() as driver:
        browser = driver.chromium.launch(args=["--no-sandbox"])
        page = browser.new_page()
        page.expose_function("runActualOnnx", run_graph)
        page.goto(args.url)
        started = time.monotonic()
        result = page.evaluate("""async ({names,maxTokens}) => {
          const hf = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0');
          const {loadLiquidProcessor,LiquidLfmRuntime,LFM_16B_ID} =
            await import('/static/lfm-browser-runtime.js');
          const {renderChatPrompt} = await import('/static/browser-runtime-utils.js');
          const processor = await loadLiquidProcessor(hf,()=>{});
          const config = await (await fetch(`https://huggingface.co/${LFM_16B_ID}/resolve/main/config.json`)).json();
          const runtime = new LiquidLfmRuntime({Tensor:hf.Tensor},config,hf.Tensor);
          const encode = t => {
            const bytes = new Uint8Array(t.data.buffer,t.data.byteOffset,t.data.byteLength);
            let s='';
            for(let i=0;i<bytes.length;i+=8192)
              s+=String.fromCharCode(...bytes.subarray(i,i+8192));
            return {type:t.type,dims:t.dims,data:btoa(s)};
          };
          const decode = t => {
            const bytes=Uint8Array.from(atob(t.data),c=>c.charCodeAt(0));
            const data=t.type==='int64'
              ? new BigInt64Array(bytes.buffer) : new Float32Array(bytes.buffer);
            return new hf.Tensor(t.type,data,t.dims);
          };
          runtime.sessions=names.map((inputNames,session)=>({inputNames,run:async feeds=> {
            const output=await window.runActualOnnx({session,
              feeds:Object.fromEntries(Object.entries(feeds).map(([k,v])=>[k,encode(v)]))});
            return Object.fromEntries(Object.entries(output).map(([k,v])=>[k,decode(v)]));
          }}));
          const sceneData=await (await fetch('/v1/browser-demo/assets/nextqa/scenes')).json();
          const frame=sceneData.scenes[1].frame_urls[0];
          const image=await hf.load_image(new URL(frame,location.href).href);
          const question='Describe what the person is doing in this image in one short sentence.';
          const prompt=renderChatPrompt(processor,[{role:'user',content:[
            {type:'image'},{type:'text',text:question}]}]);
          const inputs=await processor(image,prompt,{add_special_tokens:false});
          const output=await runtime.generate({...inputs,max_new_tokens:maxTokens});
          const generated=output.slice(null,[inputs.input_ids.dims.at(-1),null]);
          return {model:LFM_16B_ID,execution:'Actual JS adapter + CPU ONNX sessions (not WebGPU)',
            frame,question,promptTokens:inputs.input_ids.dims.at(-1),generatedTokens:generated.dims.at(-1),
            answer:processor.batch_decode(generated,{skip_special_tokens:true})[0]};
        }""", {
            "names": [[i.name for i in s.get_inputs()] for s in sessions],
            "maxTokens": args.tokens,
        })
        result["elapsed_seconds_with_cpu_bridge"] = round(time.monotonic() - started, 2)
        print(json.dumps(result, indent=2), flush=True)
        browser.close()


if __name__ == "__main__":
    main()
