import {
  AutoModelForImageTextToText,
  AutoProcessor,
  AutoTokenizer,
  Lfm2VlProcessor,
  Lfm2VlImageProcessor,
  Qwen3_5ForConditionalGeneration,
  Tensor,
  load_image,
  pipeline,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.2.0";
import { createDownloadReporter, renderChatPrompt } from "./browser-runtime-utils.js?v=20260905-16";

const MODEL_CONFIGS = {
  "lfm-1.6b": {
    id: "LiquidAI/LFM2.5-VL-1.6B-ONNX",
    family: "lfm-official",
  },
  "lfm-3b": {
    id: "LiquidAI/LFM2.5-VL-3B-ONNX",
    family: "lfm",
    dtype: {
      embed_tokens: "fp16",
      decoder_model_merged: "q4",
      vision_encoder: "fp16",
    },
  },
  "qwen-4b": {
    id: "onnx-community/Qwen3.5-4B-ONNX-OPT",
    family: "qwen",
    dtype: {
      embed_tokens: "q4",
      decoder_model_merged: "q4",
      vision_encoder: "fp16",
    },
  },
  "lfm-450m": {
    id: "onnx-community/LFM2.5-VL-450M-ONNX",
    family: "lfm",
    dtype: {
      embed_tokens: "fp16",
      decoder_model_merged: "q4",
      vision_encoder: "fp16",
    },
  },
};

let activeKey = null;
let activeConfig = null;
let processor = null;
let model = null;
let embedder = null;

async function disposeModel() {
  if (model?.dispose) await model.dispose();
  if (embedder?.dispose) await embedder.dispose();
  model = null;
  embedder = null;
  processor = null;
  activeKey = null;
  activeConfig = null;
}

async function loadModel(modelKey) {
  const config = MODEL_CONFIGS[modelKey];
  if (!config) throw new Error(`Unknown browser model: ${modelKey}`);
  if (model && activeKey === modelKey) {
    self.postMessage({ type: "ready", modelKey, modelId: config.id, cached: true, loadMs: 0 });
    return;
  }

  await disposeModel();
  const started = performance.now();
  const reportProgress = createDownloadReporter("model", (event) => self.postMessage(event));
  let liquidRuntime = null;
  if (config.family === "lfm-official") {
    liquidRuntime = await import("./lfm-browser-runtime.js?v=20260905-21");
    processor = await liquidRuntime.loadLiquidProcessor({ AutoTokenizer, Lfm2VlProcessor, Lfm2VlImageProcessor }, reportProgress);
  } else {
    processor = await AutoProcessor.from_pretrained(config.id, { progress_callback: reportProgress });
  }
  // Validate the real template before downloading weights or allocating the GPU.
  renderChatPrompt(processor, [{ role: "user", content: [{ type: "text", text: "Hello" }] }]);
  const ModelClass = config.family === "qwen"
    ? Qwen3_5ForConditionalGeneration
    : AutoModelForImageTextToText;
  if (config.family === "lfm-official") {
    const ort = await import("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0-dev.20260416-b7804b056c/dist/ort.webgpu.bundle.min.mjs");
    ort.env.wasm.numThreads = 1;
    model = await liquidRuntime.LiquidLfmRuntime.load({ ort, TensorClass: Tensor, progress: reportProgress });
  } else {
    model = await ModelClass.from_pretrained(config.id, {
      device: "webgpu",
      dtype: config.dtype,
      progress_callback: reportProgress,
    });
  }
  activeKey = modelKey;
  activeConfig = config;
  self.postMessage({
    type: "ready",
    modelKey,
    modelId: config.id,
    cached: false,
    loadMs: Math.round(performance.now() - started),
  });
}

async function infer({ modelKey, imageUrl, prompt, requestId = null, maxNewTokens = 220 }) {
  if (!model || activeKey !== modelKey) await loadModel(modelKey);
  const started = performance.now();
  const image = await load_image(imageUrl);
  const messages = [{
    role: "user",
    content: [
      { type: "image" },
      { type: "text", text: prompt },
    ],
  }];
  const templated = renderChatPrompt(processor, messages);
  const inputs = activeConfig.family === "qwen"
    ? await processor(templated, image)
    : await processor(image, templated, { add_special_tokens: false });
  let outputs = null;
  let generated = null;
  const tokenLimit = Math.max(32, Math.min(900, Number(maxNewTokens) || 220));
  try {
    outputs = await model.generate({
      ...inputs,
      do_sample: false,
      max_new_tokens: tokenLimit,
      repetition_penalty: 1.05,
    });
    const promptLength = inputs.input_ids.dims.at(-1);
    generated = outputs.slice(null, [promptLength, null]);
    const decoded = processor.batch_decode(generated, { skip_special_tokens: true })[0];
    const generatedTokens = generated.dims.at(-1);
    self.postMessage({
      type: "result",
      text: decoded.trim(),
      inferenceMs: Math.round(performance.now() - started),
      modelKey,
      modelId: activeConfig.id,
      requestId,
      generatedTokens,
      tokenLimit,
      hitTokenLimit: generatedTokens >= tokenLimit,
    });
  } finally {
    if (generated !== outputs) generated?.dispose?.();
    outputs?.dispose?.();
    for (const value of Object.values(inputs)) value?.dispose?.();
  }
}

async function embed({ texts, requestId }) {
  if (!Array.isArray(texts) || !texts.length) throw new Error("Embedding input is empty.");
  if (!embedder) {
    const reportProgress = createDownloadReporter("embedding", (event) => self.postMessage(event));
    embedder = await pipeline(
      "feature-extraction",
      "Xenova/all-MiniLM-L6-v2",
      { device: "wasm", dtype: "q8", progress_callback: reportProgress },
    );
  }
  const started = performance.now();
  const output = await embedder(texts, { pooling: "mean", normalize: true });
  const vectors = output.tolist();
  const dimensions = output.dims.at(-1);
  output.dispose?.();
  self.postMessage({
    type: "embeddings",
    requestId,
    vectors,
    dimensions,
    modelId: "Xenova/all-MiniLM-L6-v2",
    inferenceMs: Math.round(performance.now() - started),
  });
}

self.addEventListener("message", async (event) => {
  try {
    if (event.data?.type === "load") await loadModel(event.data.modelKey);
    if (event.data?.type === "infer") await infer(event.data);
    if (event.data?.type === "embed") await embed(event.data);
    if (event.data?.type === "dispose") await disposeModel();
  } catch (error) {
    self.postMessage({
      type: "error",
      operation: event.data?.type || "unknown",
      requestId: event.data?.requestId || null,
      message: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : null,
    });
  }
});
