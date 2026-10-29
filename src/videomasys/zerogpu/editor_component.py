"""Self-contained Gradio HTML cut editor; canonical validation stays on the server.

The component emits ``commit`` only after an edit, with its complete value as
input. Selection, scrubbing and playback are local and do not invoke the backend.
"""

# Embedded browser code follows the surrounding project's UI-template convention.
# ruff: noqa: E501

from copy import deepcopy


def editor_value(document, *, source_url, duration, thumbnails=None, locked=False, generation=0):
    """Package an already validated document and app-owned media URLs for gr.HTML."""
    return {
        "document": deepcopy(document), "source_url": str(source_url or ""),
        "duration": float(duration or 0), "thumbnails": deepcopy(thumbnails or []),
        "locked": bool(locked), "generation": int(generation),
    }


TEMPLATE = """
<section class="ve" aria-label="Video editor">
  <header class="ve-heading">
    <div><span class="ve-eyebrow">SEQUENCE</span>
      <input class="ve-title" data-field="title" aria-label="Sequence title"
        placeholder="Untitled sequence" maxlength="100" autocomplete="off">
    </div>
    <span class="ve-count" data-ui="count">No clips</span>
  </header>
  <div class="ve-preview">
    <video data-ui="video" playsinline preload="metadata" aria-label="Sequence video preview"></video>
    <div class="ve-empty" data-ui="empty">Choose a video to start editing</div>
    <div class="ve-title-overlay" data-ui="title-overlay" hidden></div>
    <span class="ve-muted-badge" data-ui="muted" hidden>Muted clip</span>
  </div>
  <div class="ve-transport">
    <div class="ve-transport-main">
      <button type="button" data-action="restart" aria-label="Go to sequence start" title="Go to start">↤</button>
      <button type="button" class="ve-play" data-action="play" aria-label="Play sequence">Play</button>
      <output class="ve-clock" data-ui="clock" aria-live="off">0:00.0 / 0:00.0</output>
    </div>
    <span class="ve-source-time" data-ui="source-time">Source —</span>
  </div>
  <div class="ve-toolbar" role="toolbar" aria-label="Clip editing tools">
    <div class="ve-tool-group">
      <button type="button" data-action="split" title="Split selected clip at playhead">Split</button>
      <button type="button" data-action="delete">Delete</button>
      <span class="ve-divider" aria-hidden="true"></span>
      <button type="button" data-action="earlier" aria-label="Move selected clip earlier" title="Move earlier">←</button>
      <button type="button" data-action="later" aria-label="Move selected clip later" title="Move later">→</button>
    </div>
    <div class="ve-tool-group">
      <button type="button" data-action="reset" title="Restore the full source video">Reset</button>
      <button type="button" data-action="undo" title="Undo (Ctrl or Command Z)">Undo</button>
      <button type="button" data-action="redo" title="Redo (Ctrl or Command Shift Z)">Redo</button>
    </div>
  </div>
  <div class="ve-section-heading">
    <span>Source trim</span><output data-ui="trim-range">Select a clip to trim</output>
  </div>
  <div class="ve-scroll" data-ui="scroll">
    <div class="ve-timeline" data-ui="timeline">
      <div class="ve-ruler" data-ui="ruler" aria-hidden="true"></div>
      <div class="ve-source-track" data-ui="source-track" aria-label="Full source video">
        <div class="ve-thumbnails" data-ui="source-thumbnails"></div>
        <div class="ve-excluded ve-excluded-before" data-ui="trim-before" aria-hidden="true"></div>
        <div class="ve-excluded ve-excluded-after" data-ui="trim-after" aria-hidden="true"></div>
        <div class="ve-trim-selection" data-ui="trim-selection" aria-hidden="true"></div>
        <div class="ve-handle" data-edge="start" role="slider" tabindex="0"
          aria-label="Source in" aria-orientation="horizontal" aria-valuemin="0" aria-valuemax="0" aria-valuenow="0"></div>
        <div class="ve-handle" data-edge="end" role="slider" tabindex="0"
          aria-label="Source out" aria-orientation="horizontal" aria-valuemin="0" aria-valuemax="0" aria-valuenow="0"></div>
      </div>
      <div class="ve-playhead" data-ui="playhead" role="slider" tabindex="0"
        aria-label="Source playhead" aria-valuemin="0" aria-valuemax="0" aria-valuenow="0"></div>
    </div>
  </div>
  <div class="ve-hint">Drag the handles to trim. Gray areas stay in the source, outside this clip.</div>
  <div class="ve-section-heading"><span>Sequence</span><span>Drag clips to reorder</span></div>
  <div class="ve-track" data-ui="track" role="list" aria-label="Sequence clips"></div>
  <fieldset class="ve-inspector" data-ui="inspector" disabled>
    <legend data-ui="selected-label">Select a clip</legend>
    <label class="ve-name-field">Clip name
      <input data-field="label" type="text" maxlength="80" autocomplete="off">
    </label>
    <label>Source in <span class="ve-unit">seconds</span>
      <input data-field="start" type="number" min="0" step="0.1" inputmode="decimal">
    </label>
    <label>Source out <span class="ve-unit">seconds</span>
      <input data-field="end" type="number" min="0.1" step="0.1" inputmode="decimal">
    </label>
    <label class="ve-mute-field"><input data-field="mute" type="checkbox"> Mute clip</label>
    <output class="ve-clip-duration" data-ui="clip-duration"></output>
  </fieldset>
  <div class="ve-notice" data-ui="notice" role="status" aria-live="polite"></div>
</section>
"""

# gr.HTML scopes these rules to its own element; all selectors are also namespaced.
CSS = """
.ve {--ve-ink:var(--body-text-color,#172b36); --ve-muted:var(--body-text-color-subdued,#60737e);
 --ve-line:var(--border-color-primary,#dde5e9); --ve-card:var(--block-background-fill,#fff);
 --ve-soft:var(--background-fill-secondary,#f5f8f9); --ve-accent:var(--link-text-color,#106b60);
 --ve-accent-soft:var(--color-accent-soft,#e9f5f1); color:var(--ve-ink); min-width:0;
 background:var(--ve-card); border:1px solid var(--ve-line); border-radius:16px; padding:18px;}
.ve * {box-sizing:border-box;}
.ve [hidden] {display:none !important;}
.ve-heading,.ve-transport,.ve-toolbar,.ve-tool-group,.ve-transport-main {display:flex;
 align-items:center; gap:8px;}
.ve-heading,.ve-transport,.ve-toolbar {justify-content:space-between;}
.ve-heading {margin-bottom:14px; gap:16px;}
.ve-heading > div {min-width:0; flex:1;}
.ve-eyebrow {display:block; font-size:10px; font-weight:750; letter-spacing:1.4px;
 color:var(--ve-muted); margin-bottom:3px;}
.ve .ve-title {border:0; background:transparent; font-family:inherit; font-size:19px;
 font-weight:600; line-height:1.4; width:100%;
 padding:2px 0; color:var(--ve-ink); outline-offset:4px; border-radius:3px;}
.ve-count {white-space:nowrap; font-size:12px; color:var(--ve-muted);}
.ve-preview {position:relative; width:100%; background:#0a1014; border-radius:10px; overflow:hidden;
 aspect-ratio:16 / 9; max-height:320px; display:grid; place-items:center;}
.ve-preview video {display:block; width:100%; height:100%; max-height:320px; object-fit:contain;}
.ve-empty {position:absolute; inset:0; display:grid; place-items:center; padding:24px;
 color:#a4b6c1; text-align:center; font-size:14px; background:#0a1014;}
.ve-title-overlay {position:absolute; bottom:12%; left:8%; right:8%; text-align:center;
 color:white; font-size:clamp(18px,3vw,30px); font-weight:650; line-height:1.25;
 text-shadow:0 2px 8px #000; overflow-wrap:anywhere; pointer-events:none;}
.ve-muted-badge {position:absolute; top:10px; right:10px; border-radius:5px; color:white;
 background:#000a; padding:4px 7px; font-size:11px; pointer-events:none;}
.ve-transport {padding:10px 0 14px; flex-wrap:wrap;}
.ve button {font-family:inherit; font-weight:500; font-size:12px; line-height:1.2;
 border:1px solid var(--ve-line); background:var(--ve-card);
 color:var(--ve-ink); border-radius:7px; min-height:33px; padding:7px 11px; cursor:pointer;}
.ve button:hover:not(:disabled) {border-color:var(--ve-accent); background:var(--ve-accent-soft);}
.ve button:focus-visible,.ve input:focus-visible,.ve-playhead:focus-visible,.ve-handle:focus-visible {outline:2px solid var(--ve-accent); outline-offset:2px;}
.ve button:disabled,.ve input:disabled {opacity:.45; cursor:not-allowed;}
.ve .ve-play {background:var(--ve-accent); color:var(--ve-card); border-color:var(--ve-accent); min-width:64px;}
.ve .ve-play:hover:not(:disabled) {color:var(--ve-ink);}
.ve-clock {font:12px ui-monospace,SFMono-Regular,monospace; margin-left:5px; font-variant-numeric:tabular-nums;}
.ve-source-time,.ve-hint {font-size:11px; color:var(--ve-muted);}
.ve-toolbar {gap:12px; flex-wrap:wrap; padding:0 0 12px;}
.ve-divider {height:18px; border-left:1px solid var(--ve-line); margin:0 3px;}
.ve-section-heading {display:flex; justify-content:space-between; gap:12px; font-size:11px;
 color:var(--ve-muted); padding:0 0 7px; line-height:1.5;}
.ve-section-heading > span:first-child {font-weight:650; color:var(--ve-ink);}
.ve-section-heading output {font:10px ui-monospace,monospace; align-self:center; text-align:right;}
.ve-scroll {overflow:hidden; padding:0;
 border:1px solid var(--ve-line); border-radius:9px; background:var(--ve-soft);}
.ve-timeline {position:relative; width:100%; padding:0 24px 12px;}
.ve-ruler {height:34px; position:relative; cursor:crosshair; touch-action:none;}
.ve-ruler span {position:absolute; top:5px; border-left:1px solid var(--ve-line); padding-left:5px;
 height:17px; font:10px ui-monospace,monospace; color:var(--ve-muted); white-space:nowrap;}
.ve-source-track {position:relative; height:78px; isolation:isolate; cursor:crosshair; touch-action:none;}
.ve-track {display:flex; gap:7px; min-height:46px; overflow-x:auto; overflow-y:hidden;
 scrollbar-width:thin; padding:0 0 10px; margin-bottom:10px; position:relative;}
.ve-clip {position:relative; flex:0 0 170px; height:48px; border:1px solid var(--ve-line);
 border-radius:6px; overflow:hidden; cursor:grab; background:var(--ve-card); min-width:0;}
.ve-clip[data-selected="true"] {border-color:var(--ve-accent); box-shadow:inset 0 0 0 1px var(--ve-accent); background:var(--ve-accent-soft);}
.ve-clip[data-drop="true"] {box-shadow:inset 5px 0 0 var(--ve-accent);}
.ve-thumbnails {position:absolute; inset:0; border-radius:4px; display:flex; overflow:hidden;
 background:linear-gradient(130deg,var(--ve-accent-soft),var(--ve-soft)); pointer-events:none;}
.ve-thumbnails img {height:100%; min-width:0; flex:1 1 0; object-fit:cover; pointer-events:none;}
.ve-excluded {position:absolute; top:0; bottom:0; background:rgba(103,111,121,.7);
 backdrop-filter:grayscale(1); -webkit-backdrop-filter:grayscale(1); z-index:1; pointer-events:none;}
.ve-excluded-before {left:0; border-radius:4px 0 0 4px;}
.ve-excluded-after {right:0; border-radius:0 4px 4px 0;}
.ve-trim-selection {position:absolute; top:0; bottom:0; border:2px solid var(--ve-accent);
 border-left:0; border-right:0; pointer-events:none; z-index:2;}
.ve-clip-caption {position:absolute; left:9px; right:9px; top:7px; display:flex; gap:6px;
 align-items:center; font-size:10px; overflow:hidden; pointer-events:none;}
.ve-clip-caption strong {font-weight:600; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; flex:1;}
.ve-clip-caption span {white-space:nowrap; color:var(--ve-muted);}
.ve-clip-range {position:absolute; left:9px; right:9px; bottom:7px; font:9px ui-monospace,monospace;
 color:var(--ve-muted); white-space:nowrap; pointer-events:none;}
.ve-handle {position:absolute; top:-1px; bottom:-1px; width:22px; cursor:ew-resize; z-index:4;
 background:var(--ve-accent); touch-action:none; user-select:none;}
.ve-handle::after {content:""; position:absolute; top:28px; height:22px; left:9px;
 border-left:2px solid #ffffffb3; border-right:2px solid #ffffffb3; width:5px;}
.ve-handle[data-edge="start"] {transform:translateX(-100%); border-radius:5px 0 0 5px;}
.ve-handle[data-edge="end"] {border-radius:0 5px 5px 0;}
.ve-playhead {position:absolute; top:21px; height:11px; width:14px; margin-left:-7px;
 z-index:5; cursor:ew-resize; touch-action:none; outline-offset:2px;}
.ve-playhead::before {content:""; position:absolute; width:2px; background:#f39b59;
 top:11px; height:82px; left:6px; box-shadow:0 0 1px #000; pointer-events:none;}
.ve-playhead::after {content:""; position:absolute; top:0; left:1px; width:12px; height:11px;
 background:#f39b59; clip-path:polygon(0 0,100% 0,100% 50%,50% 100%,0 50%); pointer-events:none;}
.ve-hint {padding:7px 0 12px; line-height:1.5;}
.ve-inspector {display:grid; grid-template-columns:minmax(110px,1fr) 105px 105px auto auto;
 align-items:end; gap:12px; margin:0; padding:12px; border:1px solid var(--ve-line); border-radius:9px;}
.ve-inspector legend {font-size:11px; color:var(--ve-muted); padding:0 5px;}
.ve-inspector label {font-size:11px; color:var(--ve-muted); min-width:0;}
.ve-inspector input:not([type="checkbox"]) {display:block; width:100%; margin-top:5px; padding:7px;
 background:var(--ve-card); color:var(--ve-ink); border:1px solid var(--ve-line); border-radius:5px;
 font:12px ui-monospace,SFMono-Regular,monospace; min-height:33px;}
.ve-unit {font-size:9px;}
.ve-mute-field {display:flex; align-items:center; gap:6px; min-height:33px; white-space:nowrap;}
.ve-mute-field input {accent-color:var(--ve-accent);}
.ve-clip-duration {font:11px ui-monospace,monospace; color:var(--ve-muted); padding-bottom:10px; white-space:nowrap;}
.ve-notice {font-size:12px; line-height:1.5; color:var(--ve-muted); min-height:0;}
.ve-notice:not(:empty) {padding-top:10px;}
.ve[data-locked="true"] .ve-track,.ve[data-locked="true"] .ve-scroll {opacity:.65; cursor:wait;}
.ve[data-locked="true"] .ve-handle {cursor:not-allowed;}
@media(max-width:650px) {
 .ve {padding:12px;}
 .ve-inspector {grid-template-columns:1fr 1fr; gap:10px;}
 .ve-name-field {grid-column:1/-1;}
 .ve-clip-duration {text-align:right;}
 .ve-source-time {flex-basis:100%;}
 .ve-tool-group {gap:5px;}
 .ve button {padding:7px 9px;}
 .ve-heading {align-items:flex-start;}
}
"""

JS = r"""
if (element.__videoEditorCleanup) element.__videoEditorCleanup();
const controller = new AbortController();
const signal = controller.signal;
const q = (name) => element.querySelector('[data-ui="' + name + '"]');
const field = (name) => element.querySelector('[data-field="' + name + '"]');
const copy = (value) => JSON.parse(JSON.stringify(value));
const clamp = (value, low, high) => Math.max(low, Math.min(high, value));
const rounded = (value) => Math.round(value * 1000) / 1000;
let payload = {document:null, source_url:'', duration:0, thumbnails:[], locked:false, generation:0};
let doc = null, selected = null, position = 0, sourceCursor = 0, sourcePreview = false;
let playing = false, playbackIndex = 0;
let history = [], future = [], localEcho = null, video = null, mediaCleanup = null;
let raf = 0, drag = null, reorder = null, disposed = false, sequenceCounter = 0;
let thumbnailFingerprint = '';
const newId = () => 'clip_' + (globalThis.crypto?.randomUUID?.()
 || Date.now().toString(36) + '_' + (++sequenceCounter)).replaceAll('-', '').slice(0, 24);
const total = () => (doc?.clips || []).reduce((sum, clip) => sum + clip.end - clip.start, 0);
const selectedClip = () => doc?.clips.find((clip) => clip.id === selected);
const signature = (value) => JSON.stringify(value ? {
 version:value.version, source_id:value.source_id, title:value.title || '',
 clips:value.clips.map(({id,start,end,label,mute}) => ({id,start,end,label,mute:!!mute}))
} : null);
const time = (seconds) => {
 const value = Math.max(0, Number(seconds) || 0);
 return Math.floor(value / 60) + ':' + (value % 60).toFixed(1).padStart(4, '0');
};
function notice(message) { const node = q('notice'); if (node) node.textContent = message; }
function safeURL(value) {
 if (typeof value !== 'string' || !value) return '';
 try {
  const url = new URL(value, window.location.href);
  return url.origin === window.location.origin && ['http:', 'https:'].includes(url.protocol)
   && !url.username && !url.password ? url.href : '';
 } catch (_) { return ''; }
}
function valid(value) {
 if (!value || value.version !== '1.0' || !/^[0-9a-f]{64}$/.test(value.source_id)
  || !Array.isArray(value.clips) || value.clips.length > 12
  || typeof value.title !== 'string' || value.title.length > 100 || value.title.includes('\0')) return false;
 const ids = new Set(); let length = 0;
 for (const clip of value.clips) {
  if (!clip || typeof clip.id !== 'string' || !/^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$/.test(clip.id) || ids.has(clip.id)
   || !Number.isFinite(clip.start) || !Number.isFinite(clip.end)
   || clip.start < 0 || clip.end > payload.duration
   || clip.end - clip.start < 0.1 - 1e-9 || typeof clip.label !== 'string'
   || clip.label.length > 80 || typeof clip.mute !== 'boolean') return false;
  ids.add(clip.id); length += clip.end - clip.start;
 }
 return length <= 180 + 1e-9;
}
function locationAt(value) {
 let offset = 0;
 const clips = doc?.clips || [];
 for (let index = 0; index < clips.length; index++) {
  const clip = clips[index], length = clip.end - clip.start;
  if (value < offset + length || index === clips.length - 1)
   return {clip, index, offset, source:clamp(clip.start + value - offset, clip.start, clip.end)};
  offset += length;
 }
 return null;
}
function offsetOf(id) {
 let offset = 0;
 for (const clip of doc?.clips || []) { if (clip.id === id) return offset; offset += clip.end - clip.start; }
 return 0;
}
function pause() {
 playing = false; cancelAnimationFrame(raf); raf = 0;
 if (video) video.pause();
 updatePosition();
}
function bindVideo() {
 const next = q('video');
 if (next === video) return;
 if (mediaCleanup) mediaCleanup();
 video = next;
 if (!video) return;
 const bound = video, mediaEvents = new AbortController();
 bound.addEventListener('loadedmetadata', () => sourcePreview ? seekSource(sourceCursor) : seek(position), {signal:mediaEvents.signal});
 bound.addEventListener('error', () => {
  pause(); notice('This video could not be loaded. Choose it again to refresh the source.');
 }, {signal:mediaEvents.signal});
 bound.addEventListener('ended', () => { if (playing) advance(); }, {signal:mediaEvents.signal});
 mediaCleanup = () => { mediaEvents.abort(); bound.pause(); bound.removeAttribute('src'); bound.load(); };
}
function seek(value, select = false) {
 sourcePreview = false;
 position = clamp(value, 0, total());
 const at = locationAt(position);
 if (at && video) {
  sourceCursor = at.source;
  playbackIndex = at.index; video.muted = at.clip.mute;
  if (select) selected = at.clip.id;
  if (video.readyState > 0) {
   try { video.currentTime = Math.min(at.source, Math.max(at.clip.start, at.clip.end - 0.002)); }
   catch (_) { /* metadata can be replaced during an incoming source update */ }
  }
 }
 updatePosition();
}
function seekSource(value) {
 sourcePreview = true; sourceCursor = clamp(value, 0, payload.duration);
 const chosen = selectedClip();
 if (chosen) position = offsetOf(chosen.id) + clamp(sourceCursor - chosen.start, 0, chosen.end - chosen.start);
 if (video) {
  video.muted = !!chosen?.mute && sourceCursor >= chosen.start && sourceCursor <= chosen.end;
  if (video.readyState > 0) {
   try { video.currentTime = Math.min(sourceCursor, Math.max(0, payload.duration - 0.002)); }
   catch (_) { /* the source can change while media metadata is loading */ }
  }
 }
 updatePosition();
}
function advance() {
 const clips = doc?.clips || [];
 if (!playing || !clips.length) return;
 if (playbackIndex + 1 >= clips.length) { pause(); position = total(); updatePosition(); return; }
 playbackIndex++;
 const clip = clips[playbackIndex]; position = offsetOf(clip.id); selected = clip.id;
 sourceCursor = clip.start; render();
 video.currentTime = clip.start; video.muted = clip.mute;
 video.play().catch(() => { pause(); notice('Press Play to continue the preview.'); });
}
function tick() {
 if (!playing || disposed || !video || !doc) return;
 const clip = doc.clips[playbackIndex];
 if (!clip) { pause(); return; }
 if (!video.seeking && video.currentTime >= clip.end - 0.025) advance();
 else position = clamp(offsetOf(clip.id) + video.currentTime - clip.start, 0, total());
 sourceCursor = video.currentTime;
 updatePosition();
 if (playing) raf = requestAnimationFrame(tick);
}
function play() {
 if (payload.locked || !doc?.clips.length || !safeURL(payload.source_url)) return;
 if (playing) { pause(); return; }
 const chosen = selectedClip();
 if (sourcePreview && chosen) {
  position = offsetOf(chosen.id) + (sourceCursor >= chosen.start && sourceCursor < chosen.end - 0.025
   ? sourceCursor - chosen.start : 0);
 }
 if (position >= total() - 0.025) position = 0;
 seek(position, true); playing = true; render(); notice('');
 video.play().then(() => { if (playing) { cancelAnimationFrame(raf); tick(); } })
  .catch(() => { pause(); notice('Preview is not ready yet. Try Play again.'); });
 updatePosition();
}
function updatePosition() {
 const duration = total(), at = locationAt(position), track = q('source-track');
 if (!q('clock')) return;
 const chosen = selectedClip(), source = sourcePreview ? sourceCursor : at?.source || 0;
 const outside = !chosen || source < chosen.start - 0.001 || source > chosen.end + 0.001;
 q('clock').textContent = time(position) + ' / ' + time(duration);
 q('source-time').textContent = payload.source_url ? 'Source ' + time(source)
  + (outside && chosen ? ' · Outside selected clip' : '') : 'Source —';
 q('source-time').dataset.outside = String(outside);
 const head = q('playhead');
 head.style.left = (24 + (payload.duration ? source / payload.duration : 0) * (track?.offsetWidth || 0)) + 'px';
 head.setAttribute('aria-valuemax', String(rounded(payload.duration)));
 head.setAttribute('aria-valuenow', String(rounded(source)));
 head.setAttribute('aria-valuetext', time(source) + ' of source ' + time(payload.duration));
 head.setAttribute('aria-disabled', String(payload.locked || !payload.duration));
 head.tabIndex = payload.locked || !payload.duration ? -1 : 0;
 head.hidden = !doc;
 const button = element.querySelector('[data-action="play"]');
 button.textContent = playing ? 'Pause' : 'Play'; button.setAttribute('aria-label', playing ? 'Pause sequence' : 'Play sequence');
 q('muted').hidden = !(sourcePreview ? chosen?.mute && !outside : at?.clip.mute);
 q('title-overlay').hidden = !doc?.title || !duration || outside || position >= Math.min(3, duration);
 q('title-overlay').textContent = doc?.title || '';
 const within = chosen ? (sourcePreview ? source - chosen.start : position - offsetOf(chosen.id)) : 0;
 element.querySelector('[data-action="split"]').disabled = payload.locked || !chosen
  || outside || (!sourcePreview && at?.clip.id !== chosen.id)
  || doc.clips.length >= 12 || within < 0.1 || chosen.end - chosen.start - within < 0.1;
}
function render() {
 bindVideo();
 const root = element.querySelector('.ve'), clips = doc?.clips || [], duration = total();
 root.dataset.locked = String(payload.locked);
 if (!clips.some((clip) => clip.id === selected)) selected = clips[0]?.id || null;
 field('title').value = doc?.title || ''; field('title').disabled = !doc || payload.locked;
 q('count').textContent = clips.length ? clips.length + (clips.length === 1 ? ' clip · ' : ' clips · ') + time(duration) : 'No clips';
 const url = safeURL(payload.source_url);
 if (video && video.getAttribute('src') !== url) {
  pause(); video.removeAttribute('src'); video.load();
  if (url) { video.src = url; video.load(); }
 }
 q('empty').hidden = !!url && !!clips.length;
 q('empty').textContent = doc ? (clips.length ? 'Preview source unavailable' : 'Your sequence is empty. Undo or create a new cut.') : 'Choose a video to start editing';
 const track = q('track'), sourceTrack = q('source-track'), width = sourceTrack.clientWidth || 450;
 track.replaceChildren(); q('ruler').replaceChildren();
 const ticks = Math.max(2, Math.min(5, Math.floor(width / 110)));
 for (let n = 0; n <= ticks; n++) {
  const marker = document.createElement('span'); marker.textContent = time(payload.duration * n / ticks);
  marker.style.left = (100 * n / ticks) + '%'; if (n === ticks) marker.style.transform = 'translateX(-100%)';
  q('ruler').append(marker);
 }
 const available = (payload.thumbnails || []).filter((thumb) => Number.isFinite(thumb.time) && safeURL(thumb.url));
 const thumbnailCount = Math.max(3, Math.min(12, Math.ceil(width / 90)));
 const nextThumbnailFingerprint = JSON.stringify([payload.duration, thumbnailCount, available]);
 if (nextThumbnailFingerprint !== thumbnailFingerprint || (available.length && !q('source-thumbnails').children.length)) {
  thumbnailFingerprint = nextThumbnailFingerprint; q('source-thumbnails').replaceChildren();
  for (let n = 0; n < thumbnailCount; n++) {
   const target = payload.duration * (n + 0.5) / thumbnailCount;
   const thumb = available.reduce((best, item) => !best || Math.abs(item.time - target) < Math.abs(best.time - target) ? item : best, null);
   if (thumb) {
    const img = document.createElement('img'); img.src = safeURL(thumb.url); img.alt = ''; img.loading = 'lazy'; img.draggable = false;
    q('source-thumbnails').append(img);
   }
  }
 }
 for (let index = 0; index < clips.length; index++) {
  const clip = clips[index], block = document.createElement('div');
  block.className = 've-clip'; block.dataset.clip = clip.id; block.dataset.selected = String(clip.id === selected);
  block.draggable = !payload.locked; block.tabIndex = payload.locked ? -1 : 0;
  block.setAttribute('role', 'listitem'); block.setAttribute('aria-label', 'Clip ' + (index + 1) + ': ' + clip.label + ', source ' + time(clip.start) + ' to ' + time(clip.end));
  const caption = document.createElement('div'); caption.className = 've-clip-caption';
  const label = document.createElement('strong'); label.textContent = (index + 1) + ' ' + (clip.label || 'Clip');
  const length = document.createElement('span'); length.textContent = (clip.mute ? '◌ ' : '') + time(clip.end - clip.start);
  caption.append(label, length); block.append(caption);
  const range = document.createElement('div'); range.className = 've-clip-range';
  range.textContent = time(clip.start) + ' → ' + time(clip.end); block.append(range);
  track.append(block);
 }
 const chosen = selectedClip(), active = !!chosen && !payload.locked;
 const startFraction = chosen && payload.duration ? chosen.start / payload.duration : 1;
 const endFraction = chosen && payload.duration ? chosen.end / payload.duration : 1;
 q('trim-before').style.width = (startFraction * 100) + '%';
 q('trim-after').style.width = ((1 - endFraction) * 100) + '%';
 q('trim-selection').style.left = (startFraction * 100) + '%';
 q('trim-selection').style.width = ((endFraction - startFraction) * 100) + '%';
 q('trim-selection').hidden = !chosen;
 q('trim-range').textContent = chosen ? 'Clip ' + (clips.indexOf(chosen) + 1) + ' · '
  + time(chosen.start) + ' → ' + time(chosen.end) : 'Select a clip to trim';
 const others = duration - (chosen ? chosen.end - chosen.start : 0);
 for (const edge of ['start', 'end']) {
  const handle = element.querySelector('.ve-handle[data-edge="' + edge + '"]');
  handle.hidden = !chosen; handle.tabIndex = active ? 0 : -1;
  handle.style.left = ((edge === 'start' ? startFraction : endFraction) * 100) + '%';
  handle.setAttribute('aria-disabled', String(!active));
  handle.setAttribute('aria-valuemin', String(rounded(edge === 'start' ? Math.max(0, (chosen?.end || 0) - (180 - others)) : (chosen?.start || 0) + 0.1)));
  handle.setAttribute('aria-valuemax', String(rounded(edge === 'start' ? Math.max(0, (chosen?.end || 0) - 0.1) : Math.min(payload.duration, (chosen?.start || 0) + 180 - others))));
  handle.setAttribute('aria-valuenow', String(rounded(chosen?.[edge] || 0)));
  handle.setAttribute('aria-valuetext', time(chosen?.[edge] || 0) + ' in source video');
  handle.title = (edge === 'start' ? 'Source in: ' : 'Source out: ') + time(chosen?.[edge] || 0);
 }
 q('inspector').disabled = !active;
 q('selected-label').textContent = chosen ? 'Clip ' + (clips.indexOf(chosen) + 1) + ' · source range' : 'Select a clip';
 for (const name of ['label','start','end']) field(name).value = chosen ? (name === 'label' ? chosen[name] : rounded(chosen[name])) : '';
 field('start').max = chosen ? String(rounded(chosen.end - 0.1)) : '0';
 field('end').max = String(payload.duration); field('end').min = chosen ? String(rounded(chosen.start + 0.1)) : '0.1';
 field('mute').checked = !!chosen?.mute;
 q('clip-duration').textContent = chosen ? time(chosen.end - chosen.start) + ' long' : '';
 for (const action of ['restart','play']) element.querySelector('[data-action="' + action + '"]').disabled = !clips.length || !url || payload.locked;
 element.querySelector('[data-action="delete"]').disabled = !active;
 element.querySelector('[data-action="earlier"]').disabled = !active || clips.indexOf(chosen) <= 0;
 element.querySelector('[data-action="later"]').disabled = !active || clips.indexOf(chosen) >= clips.length - 1;
  element.querySelector('[data-action="undo"]').disabled = payload.locked || !history.length;
  element.querySelector('[data-action="redo"]').disabled = payload.locked || !future.length;
 element.querySelector('[data-action="reset"]').disabled = payload.locked || !doc || payload.duration < 0.1;
 updatePosition();
}
function publish() {
 localEcho = signature(doc);
 payload = {...payload, document:copy(doc)};
 props.value = copy(payload);
 trigger('commit');
}
function commit(next, before = doc) {
 if (payload.locked || !valid(next)) { notice('Keep each clip within the source, at least 0.1 seconds long, and the sequence under 3 minutes.'); render(); return; }
 if (signature(next) === signature(before)) { doc = next; render(); return; }
 const source = sourcePreview ? sourceCursor : null;
 pause(); history.push(copy(before)); if (history.length > 50) history.shift(); future = [];
 doc = next; position = clamp(position, 0, total()); notice(''); render();
 if (source === null) seek(position); else seekSource(source);
 publish();
}
function moveClip(fromId, toIndex) {
 const next = copy(doc), from = next.clips.findIndex((clip) => clip.id === fromId);
 if (from < 0 || toIndex < 0 || toIndex >= next.clips.length || from === toIndex) return;
 const [clip] = next.clips.splice(from, 1); next.clips.splice(toIndex, 0, clip);
 selected = fromId; sourcePreview = false; position = next.clips.slice(0, toIndex).reduce((sum, item) => sum + item.end - item.start, 0);
 commit(next);
}
function action(name) {
 if (payload.locked || !doc) return;
 const chosen = selectedClip(), next = copy(doc), index = next.clips.findIndex((clip) => clip.id === selected);
 if (name === 'play') { play(); return; }
 if (name === 'restart') { pause(); seek(0, true); render(); return; }
 if (name === 'reset') {
  next.title = ''; next.clips = [{id:newId(),
   start:0, end:Math.min(payload.duration, 180), label:'Full video', mute:false}];
  selected = next.clips[0].id; position = 0; sourcePreview = false; commit(next); return;
 }
 if (name === 'undo' || name === 'redo') {
  const stack = name === 'undo' ? history : future, other = name === 'undo' ? future : history;
  if (!stack.length) return;
  pause(); other.push(copy(doc)); if (other.length > 50) other.shift(); doc = stack.pop();
  position = clamp(position, 0, total()); render(); seek(position); notice(''); publish(); return;
 }
 if (!chosen) return;
 if (name === 'earlier' || name === 'later') { moveClip(selected, index + (name === 'earlier' ? -1 : 1)); return; }
 if (name === 'delete') {
  next.clips.splice(index, 1); selected = next.clips[Math.min(index, next.clips.length - 1)]?.id || null; commit(next); return;
 }
 if (name === 'split') {
  const source = sourcePreview ? sourceCursor : chosen.start + position - offsetOf(chosen.id);
  if (next.clips.length >= 12 || source - chosen.start < 0.1 || chosen.end - source < 0.1) return;
  const second = {...chosen, id:newId(), start:rounded(source)};
  next.clips[index].end = rounded(source); next.clips.splice(index + 1, 0, second); selected = second.id; commit(next);
 }
}
element.addEventListener('click', (event) => {
 const button = event.target.closest('[data-action]');
 if (button && element.contains(button) && !button.disabled) { action(button.dataset.action); return; }
 if (payload.locked || !doc || event.target.closest('.ve-handle')) return;
 const block = event.target.closest('[data-clip]');
 if (block && element.contains(block)) {
  pause(); selected = block.dataset.clip; seekSource(selectedClip().start); render();
 }
}, {signal});
element.addEventListener('change', (event) => {
 const name = event.target.dataset.field;
 if (!name || !doc || payload.locked) return;
 const next = copy(doc);
 if (name === 'title') next.title = event.target.value;
 else {
  const clip = next.clips.find((item) => item.id === selected); if (!clip) return;
  clip[name] = name === 'mute' ? event.target.checked : ['start','end'].includes(name) ? rounded(Number(event.target.value)) : event.target.value;
 }
 commit(next);
}, {signal});
element.addEventListener('keydown', (event) => {
 if (payload.locked || event.target.matches('input,textarea,select')) return;
 const key = event.key.toLowerCase(), modifier = event.ctrlKey || event.metaKey;
 if (modifier && key === 'z') { event.preventDefault(); action(event.shiftKey ? 'redo' : 'undo'); }
 else if (modifier && key === 'y') { event.preventDefault(); action('redo'); }
 else if (key === ' ' && !event.target.closest('button,[data-clip],.ve-handle')) { event.preventDefault(); action('play'); }
 else if (key === 'delete' || key === 'backspace') { event.preventDefault(); action('delete'); }
 else if (event.target === q('playhead') && ['arrowleft','arrowright','home','end'].includes(key)) {
  event.preventDefault(); pause();
  const current = sourcePreview ? sourceCursor : locationAt(position)?.source || 0;
  seekSource(key === 'home' ? 0 : key === 'end' ? payload.duration : current + (key === 'arrowright' ? 1 : -1) * (event.shiftKey ? 1 : 0.1));
 } else if (event.target.matches('.ve-handle') && ['arrowleft','arrowright','home','end'].includes(key)) {
  event.preventDefault(); const chosen = selectedClip(); if (!chosen) return;
  const next = copy(doc), clip = next.clips.find((item) => item.id === selected), edge = event.target.dataset.edge;
  const low = Number(event.target.getAttribute('aria-valuemin')), high = Number(event.target.getAttribute('aria-valuemax'));
  clip[edge] = clamp(rounded(key === 'home' ? low : key === 'end' ? high
   : clip[edge] + (key === 'arrowright' ? 1 : -1) * (event.shiftKey ? 1 : 0.1)), low, high);
  sourcePreview = true; sourceCursor = clip[edge]; commit(next);
 } else if (event.target.matches('[data-clip]') && ['enter',' '].includes(key)) {
  event.preventDefault(); selected = event.target.dataset.clip; pause(); seekSource(selectedClip().start); render();
 }
}, {signal});
element.addEventListener('pointerdown', (event) => {
 if (payload.locked || !doc || !payload.duration || event.button !== 0) return;
 const handle = event.target.closest('.ve-handle'), scrub = event.target.closest('.ve-playhead,.ve-ruler,.ve-source-track');
 if (!handle && !scrub) return;
 event.preventDefault(); pause();
 if (handle) {
  if (!selectedClip()) return;
  drag = {kind:'trim', id:event.pointerId, edge:handle.dataset.edge, before:copy(doc), x:event.clientX,
   pixelsPerSecond:q('source-track').getBoundingClientRect().width / payload.duration, clipId:selected,
   priorPosition:position, priorSource:sourceCursor, priorSourcePreview:sourcePreview};
 } else drag = {kind:'scrub', id:event.pointerId};
 element.setPointerCapture(event.pointerId);
 if (!handle) { const rect = q('source-track').getBoundingClientRect(); seekSource((event.clientX - rect.left) / rect.width * payload.duration); }
}, {signal});
element.addEventListener('pointermove', (event) => {
 if (!drag || drag.id !== event.pointerId || payload.locked) return;
 if (drag.kind === 'scrub') {
  const rect = q('source-track').getBoundingClientRect(); seekSource((event.clientX - rect.left) / rect.width * payload.duration); return;
 }
 const original = drag.before.clips.find((clip) => clip.id === drag.clipId), next = copy(drag.before);
 const clip = next.clips.find((item) => item.id === drag.clipId), delta = (event.clientX - drag.x) / drag.pixelsPerSecond;
 const others = drag.before.clips.reduce((sum, item) => sum + (item.id === clip.id ? 0 : item.end - item.start), 0);
 if (drag.edge === 'start') clip.start = clamp(rounded(original.start + delta),
  Math.max(0, original.end - (180 - others)), original.end - 0.1);
 else clip.end = clamp(rounded(original.end + delta), original.start + 0.1,
  Math.min(payload.duration, original.start + 180 - others));
 doc = next; position = offsetOf(clip.id) + (drag.edge === 'start' ? 0 : clip.end - clip.start);
 sourceCursor = clip[drag.edge]; sourcePreview = true; render(); seekSource(sourceCursor);
}, {signal});
function finishDrag(event, cancel = false) {
 if (!drag || event.pointerId !== drag.id) return;
 const finished = drag; drag = null;
 if (element.hasPointerCapture(event.pointerId)) element.releasePointerCapture(event.pointerId);
 if (finished.kind === 'trim') {
  const next = copy(doc); doc = finished.before;
  if (cancel || payload.locked) {
   position = finished.priorPosition; sourceCursor = finished.priorSource; sourcePreview = finished.priorSourcePreview;
   render(); if (sourcePreview) seekSource(sourceCursor); else seek(position);
  }
  else commit(next, finished.before);
 } else render();
}
element.addEventListener('pointerup', (event) => finishDrag(event), {signal});
element.addEventListener('pointercancel', (event) => finishDrag(event, true), {signal});
element.addEventListener('lostpointercapture', (event) => finishDrag(event, true), {signal});
element.addEventListener('dragstart', (event) => {
 const block = event.target.closest('[data-clip]');
 if (!block || payload.locked || drag) { event.preventDefault(); return; }
 pause(); selected = block.dataset.clip; reorder = selected;
 event.dataTransfer.effectAllowed = 'move'; event.dataTransfer.setData('text/plain', selected);
}, {signal});
element.addEventListener('dragover', (event) => {
 if (!reorder || payload.locked) return;
 const block = event.target.closest('[data-clip]'); if (!block) return;
 event.preventDefault(); event.dataTransfer.dropEffect = 'move';
 for (const item of element.querySelectorAll('[data-drop]')) delete item.dataset.drop;
 block.dataset.drop = 'true';
}, {signal});
element.addEventListener('drop', (event) => {
 const block = event.target.closest('[data-clip]');
 if (!block || !reorder || payload.locked) return;
 event.preventDefault(); const source = reorder; reorder = null;
 moveClip(source, doc.clips.findIndex((clip) => clip.id === block.dataset.clip));
}, {signal});
element.addEventListener('dragend', () => {
 reorder = null; for (const item of element.querySelectorAll('[data-drop]')) delete item.dataset.drop;
}, {signal});
function incoming() {
 if (disposed) return;
 const incomingValue = props.value || {}, previousSource = doc?.source_id;
 const generation = Number(incomingValue.generation ?? 0);
 if (Number.isFinite(generation) && generation < payload.generation) {
  props.value = copy(payload); return;
 }
 if (drag) {
  const canceled = drag; drag = null;
  if (canceled.before) {
   doc = canceled.before; position = canceled.priorPosition;
   sourceCursor = canceled.priorSource; sourcePreview = canceled.priorSourcePreview;
  }
  if (element.hasPointerCapture(canceled.id)) element.releasePointerCapture(canceled.id);
 }
 payload = {...incomingValue, source_url:incomingValue.source_url || '',
  duration:Number(incomingValue.duration) || 0, thumbnails:incomingValue.thumbnails || [], locked:!!incomingValue.locked,
  generation:Number.isFinite(generation) ? generation : 0};
 const next = payload.document;
 if (next && !valid(next)) { pause(); doc = null; sourcePreview = false; sourceCursor = 0; history = []; future = []; render(); notice('This edit document is invalid. Reload the video or create a new cut.'); return; }
 const fingerprint = signature(next), changed = fingerprint !== signature(doc);
 if (changed && fingerprint !== localEcho) {
  pause(); history = []; future = []; selected = null; position = 0; sourceCursor = 0; sourcePreview = false;
 }
 doc = next ? copy(next) : null;
 if (previousSource !== doc?.source_id) { pause(); position = 0; sourceCursor = 0; sourcePreview = false; history = []; future = []; }
 if (payload.locked) pause();
 render(); if (changed) seek(position);
 if (payload.locked) notice('Editing is paused while this operation finishes.');
 else if (q('notice').textContent === 'Editing is paused while this operation finishes.') notice('');
}
const unwatch = watch('value', incoming);
const resize = new ResizeObserver(() => {
 if (disposed) return;
 if (!element.getClientRects().length) { pause(); return; }
 if (!drag) render();
});
resize.observe(element);
const connectionCheck = setInterval(() => { if (!element.isConnected) cleanup(); }, 1000);
function cleanup() {
 if (disposed) return;
 disposed = true; playing = false; cancelAnimationFrame(raf); clearInterval(connectionCheck);
 controller.abort(); resize.disconnect(); if (mediaCleanup) mediaCleanup();
 if (typeof unwatch === 'function') unwatch();
 delete element.__videoEditorCleanup;
}
element.__videoEditorCleanup = cleanup;
incoming();
"""
