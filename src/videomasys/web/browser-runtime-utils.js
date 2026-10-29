// Transformers.js 4.2.0 bundles a Jinja parser without generation-block support.
// These tags mark assistant-token masks for training; they emit no inference text.
// Preserve their whitespace controls by replacing only the tags with empty output.
export function compatibleChatTemplate(template) {
  if (typeof template !== "string") throw new Error("The model has no usable chat template.");
  return template.replace(/\{%(-?)\s*(?:end)?generation\s*(-?)%\}/g, "{{$1 '' $2}}");
}

export function renderChatPrompt(processor, messages) {
  const template = processor.tokenizer.get_chat_template({
    chat_template: processor.chat_template ?? null,
  });
  return processor.apply_chat_template(messages, {
    chat_template: compatibleChatTemplate(template),
    add_generation_prompt: true,
    tokenize: false,
  });
}

// Individual files load concurrently. Report cumulative transferred bytes rather
// than presenting each file's percentage as overall model progress.
export function createDownloadReporter(scope, send, now = () => performance.now()) {
  const downloaded = new Map();
  let lastSent = -Infinity;
  return (event) => {
    const loaded = Number(event?.loaded);
    if (event?.file && Number.isFinite(loaded) && loaded >= 0) {
      const key = `${event.name || ""}/${event.file}`;
      downloaded.set(key, Math.max(downloaded.get(key) || 0, loaded));
    }
    const timestamp = now();
    if (timestamp - lastSent < 150 && event?.status !== "done") return;
    lastSent = timestamp;
    send({
      type: "progress",
      scope,
      downloadedBytes: [...downloaded.values()].reduce((total, bytes) => total + bytes, 0),
    });
  };
}
