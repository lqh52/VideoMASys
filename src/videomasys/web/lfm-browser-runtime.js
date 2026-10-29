// Adapter for LiquidAI's three-graph LFM2.5-VL export (not the Transformers.js
// decoder_model_merged format). Uses the real HF processor for image/token layout.
// Graph contract: https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX
export const LFM_16B_ID = "LiquidAI/LFM2.5-VL-1.6B-ONNX";
export const LFM_FILES = ["embed_tokens_fp16", "embed_images_fp16", "decoder_q4"];
const MAX_CONTEXT = 4096;
const PREFILL_CHUNK = 128;

export async function loadLiquidProcessor({ AutoTokenizer, Lfm2VlProcessor, Lfm2VlImageProcessor }, progress) {
  // This export stores the image configuration inside processor_config.json,
  // whereas Transformers.js AutoProcessor expects preprocessor_config.json.
  const base = `https://huggingface.co/${LFM_16B_ID}/resolve/main`;
  const [response, templateResponse, tokenizer] = await Promise.all([
    fetch(`${base}/processor_config.json`),
    fetch(`${base}/chat_template.jinja`),
    AutoTokenizer.from_pretrained(LFM_16B_ID, { progress_callback: progress }),
  ]);
  if (!response.ok || !templateResponse.ok) throw new Error("LFM processor configuration is unavailable.");
  const config = await response.json();
  if (!config.image_processor) throw new Error("LFM image processor configuration is missing.");
  return new Lfm2VlProcessor(config, {
    tokenizer,
    image_processor: new Lfm2VlImageProcessor(config.image_processor),
  }, await templateResponse.text());
}

// Cache before reading into JS; avoid holding an array of download chunks plus
// multiple full copies of gigabyte-sized weights. Cache failures remain nonfatal.
export async function readModelFile(url, progress, fetchFile = fetch, storage = globalThis.caches) {
  let cache = null;
  try {
    cache = await storage?.open("videomasys-lfm-onnx-v1");
    const hit = await cache?.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch (_) { /* Storage is optional. */ }
  const response = await fetchFile(url);
  if (!response.ok) throw new Error(`Model download failed (${response.status}): ${url.split("/").pop()}`);
  let loaded = 0;
  const tracked = new Response(response.body.pipeThrough(new TransformStream({
    transform(chunk, controller) {
      loaded += chunk.byteLength;
      progress({ file: url.split("/").pop(), loaded });
      controller.enqueue(chunk);
    },
  })), { headers: response.headers });
  if (cache) {
    // Await disk caching to keep the tee's unread branch from buffering weights.
    // On failure fetch again: the first response may already have been consumed.
    try {
      await cache.put(url, tracked);
      const saved = await cache.match(url);
      if (saved) return new Uint8Array(await saved.arrayBuffer());
    } catch (_) { /* Quota/private mode: use one direct buffer below. */ }
    const fallback = await fetchFile(url);
    if (!fallback.ok) throw new Error(`Model download failed (${fallback.status}).`);
    return new Uint8Array(await fallback.arrayBuffer());
  }
  return new Uint8Array(await tracked.arrayBuffer());
}

export class LiquidLfmRuntime {
  constructor(ort, config, TensorClass) {
    this.ort = ort;
    this.config = config;
    this.TensorClass = TensorClass;
    this.sessions = [];
  }

  static async load({ ort, TensorClass, progress = () => {}, read = readModelFile }) {
    const base = `https://huggingface.co/${LFM_16B_ID}/resolve/main`;
    const response = await fetch(`${base}/config.json`);
    if (!response.ok) throw new Error(`LFM configuration unavailable (${response.status}).`);
    const runtime = new LiquidLfmRuntime(ort, await response.json(), TensorClass);
    try {
      // Sequential session construction limits simultaneous weight buffers.
      for (const name of LFM_FILES) {
        const graph = await read(`${base}/onnx/${name}.onnx`, progress);
        const data = await read(`${base}/onnx/${name}.onnx_data`, progress);
        runtime.sessions.push(await ort.InferenceSession.create(graph, {
          executionProviders: ["webgpu"],
          externalData: [{ path: `${name}.onnx_data`, data }],
        }));
      }
      return runtime;
    } catch (error) {
      await runtime.dispose();
      throw error;
    }
  }

  async dispose() {
    const sessions = this.sessions.splice(0);
    await Promise.allSettled(sessions.map((session) => session.release()));
  }

  async generate({ input_ids, pixel_values, pixel_attention_mask, spatial_shapes,
    max_new_tokens = 220, repetition_penalty = 1.05 }) {
    const { ort, config } = this;
    const [tokens, vision, decoder] = this.sessions;
    const ids = Array.from(input_ids.data, Number);
    if (ids.length + max_new_tokens > MAX_CONTEXT) {
      throw new Error(`LFM input exceeds the ${MAX_CONTEXT}-token browser budget. Shorten the question or evidence.`);
    }
    const owned = new Set();
    const own = (tensor) => { owned.add(tensor); return tensor; };
    const release = (tensor) => { if (owned.delete(tensor)) tensor.dispose(); };
    const tensor = (type, data, dims) => own(new ort.Tensor(type, data, dims));
    const run = async (session, feeds) => {
      const out = await session.run(feeds);
      Object.values(out).forEach(own);
      return out;
    };
    const text = config.text_config;
    const hidden = text.hidden_size;
    const generated = [];
    const eosIds = new Set([].concat(text.eos_token_id ?? config.eos_token_id ?? 7));
    try {
      const pixels = {};
      for (const [name, value] of Object.entries({ pixel_values, pixel_attention_mask, spatial_shapes })) {
        if (!value) throw new Error(`Missing LFM image input: ${name}`);
        pixels[name] = tensor(value.type, value.data, value.dims);
      }
      const imageOutput = await run(vision, pixels);
      Object.values(pixels).forEach(release);
      const features = imageOutput.image_features;
      const imagePositions = ids.flatMap((id, index) => id === config.image_token_id ? [index] : []);
      if (!features || features.dims.at(-1) !== hidden || features.data.length !== imagePositions.length * hidden) {
        throw new Error("LFM image/token alignment mismatch; refusing to generate from misaligned evidence.");
      }
      const input = tensor("int64", BigInt64Array.from(ids, BigInt), [1, ids.length]);
      const tokenOutput = await run(tokens, { input_ids: input });
      release(input);
      const embeds = tokenOutput.inputs_embeds;
      // These official FP16-weight graphs expose float32 I/O.
      if (embeds.type !== "float32" || features.type !== "float32") {
        throw new Error("Unexpected LFM graph I/O dtype; expected float32 embeddings.");
      }
      const combined = new Float32Array(embeds.data);
      for (let i = 0; i < imagePositions.length; i++) {
        combined.set(features.data.subarray(i * hidden, (i + 1) * hidden), imagePositions[i] * hidden);
      }
      Object.values(tokenOutput).forEach(release);
      Object.values(imageOutput).forEach(release);
      const cache = {};
      for (const name of decoder.inputNames) {
        if (name.startsWith("past_conv")) {
          const shape = [1, hidden, text.conv_L_cache ?? 3];
          cache[name] = tensor("float32", new Float32Array(shape.reduce((a, b) => a * b)), shape);
        } else if (name.startsWith("past_key_values")) {
          cache[name] = tensor("float32", new Float32Array(0),
            [1, text.num_key_value_heads, 0, hidden / text.num_attention_heads]);
        }
      }
      const decode = async (embedding, length) => {
        const mask = tensor("int64", new BigInt64Array(length).fill(1n), [1, length]);
        let output;
        try { output = await run(decoder, { inputs_embeds: embedding, attention_mask: mask, ...cache }); }
        finally { release(mask); release(embedding); }
        for (const [name, value] of Object.entries(output)) {
          const key = name.replace(/^present_conv/, "past_conv").replace(/^present\./, "past_key_values.");
          if (Object.hasOwn(cache, key)) {
            release(cache[key]);
            cache[key] = value;
          } else if (name !== "logits") release(value);
        }
        return output.logits;
      };
      let logits = null;
      // Bound prefill logits to 128 × vocabulary, not the entire prompt.
      for (let offset = 0; offset < ids.length; offset += PREFILL_CHUNK) {
        release(logits);
        const end = Math.min(ids.length, offset + PREFILL_CHUNK);
        const chunk = tensor("float32", combined.slice(offset * hidden, end * hidden), [1, end - offset, hidden]);
        logits = await decode(chunk, end);
      }
      for (let step = 0; step < max_new_tokens; step++) {
        const vocab = logits.dims.at(-1);
        const scores = logits.data.subarray(logits.data.length - vocab);
        for (const id of new Set([...ids, ...generated])) {
          if (id < vocab) scores[id] = scores[id] < 0 ? scores[id] * repetition_penalty : scores[id] / repetition_penalty;
        }
        let next = 0;
        for (let i = 1; i < vocab; i++) if (scores[i] > scores[next]) next = i;
        release(logits);
        generated.push(next);
        if (eosIds.has(next) || generated.length === max_new_tokens) break;
        const token = tensor("int64", BigInt64Array.of(BigInt(next)), [1, 1]);
        const output = await run(tokens, { input_ids: token });
        release(token);
        logits = await decode(output.inputs_embeds, ids.length + generated.length);
      }
      return new this.TensorClass("int64", BigInt64Array.from([...ids, ...generated], BigInt), [1, ids.length + generated.length]);
    } finally {
      for (const value of owned) value.dispose();
    }
  }
}
