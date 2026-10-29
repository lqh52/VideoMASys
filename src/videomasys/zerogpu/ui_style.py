"""Visual system for the Gradio workspace; no external assets or web fonts."""
# ruff: noqa: E501

import gradio as gr

THEME = gr.themes.Base(
    primary_hue="teal",
    neutral_hue="slate",
    font=[gr.themes.Font("Inter"), gr.themes.Font("system-ui"), gr.themes.Font("sans-serif")],
    font_mono=[gr.themes.Font("SFMono-Regular"), gr.themes.Font("monospace")],
).set(
    body_background_fill="#f5f6f8",
    body_background_fill_dark="#10171c",
    block_background_fill="#ffffff",
    block_background_fill_dark="#182229",
    body_text_color="#172b36",
    body_text_color_dark="#e1eaf0",
    body_text_color_subdued="#60737e",
    body_text_color_subdued_dark="#a4b6c1",
    background_fill_secondary="#f5f8f9",
    background_fill_secondary_dark="#1c2a32",
    border_color_primary="#dde5e9",
    border_color_primary_dark="#33434d",
    link_text_color="#106b60",
    link_text_color_dark="#6cd2be",
    color_accent_soft="#e9f5f1",
    color_accent_soft_dark="#193c35",
    block_border_width="1px",
    block_radius="12px",
    button_primary_background_fill="#106b60",
    button_primary_background_fill_hover="#0a554c",
    button_primary_text_color="#ffffff",
)

STYLE = """
.gradio-container {width:100% !important; max-width:1240px !important; margin:auto; box-sizing:border-box;
 padding:clamp(16px,2vw,26px) clamp(14px,2.3vw,28px) 30px !important;
 --ink:var(--body-text-color); --muted:var(--body-text-color-subdued);
 --line:var(--border-color-primary); --card:var(--block-background-fill);
 --soft:var(--background-fill-secondary); --accent:var(--link-text-color);
 --accent-soft:var(--color-accent-soft); color:var(--ink);}
.gradio-container .main {padding:0 !important;}
.app-header {display:flex; align-items:center; justify-content:space-between; gap:20px;
 padding: 4px 0 24px; border-bottom:1px solid var(--line); margin-bottom:8px;}
.brand {display:flex; align-items:center; gap:12px;}
.brand-mark {display:grid; place-items:center; width:40px; height:40px; border-radius:12px;
 background:#106b60; color:white; font-size:20px; font-weight:700;}
.brand h1 {font-size:22px !important; font-weight:700; letter-spacing:-.8px; margin:0 !important;}
.eyebrow {font-size:10px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase;
 color:var(--muted); margin-bottom:7px;}
.brand .eyebrow {margin:3px 0 0; font-size:9px; letter-spacing:1.1px;}
.model-badge {font-size:12px; background:var(--card); border:1px solid var(--line);
 border-radius:30px; padding:8px 12px; color:var(--muted); white-space:nowrap;}
.model-badge::before {content:""; display:inline-block; height:6px; width:6px; border-radius:50%;
 background:var(--accent); margin-right:8px; vertical-align:middle;}
#task-mode {padding:4px 0;}
#task-mode .wrap {gap:0 !important; width:max-content; padding:4px; border:1px solid var(--line);
 border-radius:12px; background:var(--soft);}
#task-mode label {border:0 !important; border-radius:8px !important; padding:8px 25px !important;
 margin:0 !important; background:transparent !important; box-shadow:none !important;}
#task-mode label:has(input:checked) {background:var(--card) !important; color:var(--accent);}
#task-mode input {position:absolute; opacity:0; width:0;}
#task-mode label:focus-within {outline:2px solid var(--accent);}
#editing-workspace {gap:16px; padding-top:8px;}
#editing-workspace h3 {font-size:17px; font-weight:650; margin:0 0 6px;}
#edit-actions {padding:20px !important; border:1px solid var(--line); border-radius:16px;
 background:var(--card); gap:14px;}
#edit-actions p {font-size:12px; line-height:1.65; color:var(--muted);}
.edit-status {padding:12px 14px; border-radius:9px; font-size:12px; line-height:1.6;
 background:var(--soft); overflow-wrap:anywhere;}
.edit-status[data-state="running"] {border-left:3px solid var(--accent);}
.edit-status[data-state="error"] {border-left:3px solid #c64c40;}
#workspace {gap:22px !important; align-items:stretch; margin-top:8px;}
#video-column, #question-column {min-width:0; background:var(--card); border:1px solid var(--line);
 border-radius:16px; padding:22px !important; gap:14px !important;}
.section-heading {display:flex; align-items:center; gap:10px; margin:0;}
.section-heading .number {color:var(--accent); font:12px ui-monospace,monospace;
 background:var(--accent-soft); padding:5px 7px; border-radius:6px;}
.section-heading h2 {margin:0 !important; font-size:16px !important; font-weight:650;}
#video-input {border:1px dashed var(--line) !important; border-radius:10px !important; overflow:hidden;}
#question-input {border:0 !important; box-shadow:none !important; padding:0 !important;}
#question-input textarea {font-size:16px !important; line-height:1.6 !important; padding:15px !important;
 border:1px solid var(--line) !important; border-radius:10px !important; min-height:145px;}
#choices-panel {border:1px solid var(--line) !important; box-shadow:none !important; border-radius:10px;}
#choices-panel textarea {font-size:13px !important; line-height:1.6;}
#run {min-height:49px; border-radius:10px; font-size:14px; font-weight:650;
 box-shadow:0 3px 8px #106b6016; transition:background .15s,transform .15s;}
#run:hover:not(:disabled) {transform:translateY(-1px);}
#run:disabled {opacity:.7; cursor:not-allowed;}
.small-note {font-size:11px; color:var(--muted); line-height:1.6; margin:0;}
#status {padding:0; min-height:0;}
#status:has(.run-status[data-state="idle"]) {display:none;}
.run-status {display:flex; gap:11px; padding:12px 14px; border-radius:10px;
 background:var(--soft); color:var(--ink); align-items:flex-start;}
.run-status[data-state="idle"] {display:none;}
.run-status strong, .run-status span {display:block;}
.run-status strong {font-size:12px;}
.run-status .detail {font-size:12px; color:var(--muted); margin-top:3px; overflow-wrap:anywhere;}
.run-status .indicator {width:13px; height:13px; margin-top:2px; flex-shrink:0;
 border:2px solid var(--line); border-radius:50%;}
.run-status[data-state="running"] .indicator {border-top-color:var(--accent); animation:spin 1s linear infinite;}
.run-status[data-state="complete"] .indicator {background:var(--accent); border-color:var(--accent);}
.run-status[data-state="error"] .indicator {background:#c64c40; border-color:#c64c40;}
#example-library {padding:12px 0 6px; background:transparent !important; border:0 !important; box-shadow:none !important;}
.form:has(> #example-library) {background:transparent !important; border:0 !important;}
#example-library .wrap:has(> label) {display:grid !important; grid-template-columns:repeat(4,minmax(0,1fr)); gap:10px !important;}
#example-library .wrap > label {margin:0 !important; padding:12px !important; min-width:0;
 border:1px solid var(--line) !important; border-radius:10px !important; background:var(--card);
 align-items:flex-start; transition:border-color .15s,box-shadow .15s;}
#example-library .wrap > label:hover {border-color:var(--accent) !important; box-shadow:0 2px 10px #172b3609;}
#example-library .wrap > label:has(input:checked) {background:var(--accent-soft); border-color:var(--accent) !important;}
#example-library .wrap > label span {font-size:12px !important; line-height:1.5; white-space:pre-line;}
#example-cards {padding:0 !important;}
#example-cards .html-container {padding:0 !important;}
#example-cards .has_children {gap:0 !important;}
#example-library.with-thumbnails .wrap > label {display:grid !important;
 grid-template-columns:16px minmax(0,1fr); column-gap:10px !important; row-gap:10px !important;
 padding:0 12px 12px !important; overflow:hidden; align-content:start;}
#example-library.with-thumbnails .wrap > label::before {content:""; grid-column:1 / -1;
 width:calc(100% + 24px); margin:0 -12px; aspect-ratio:16 / 9;
 background-color:var(--soft); background-image:var(--example-image,none);
 background-position:center; background-size:cover; border-bottom:1px solid var(--line);}
#example-library.with-thumbnails .wrap > label input {margin-top:3px !important;}
#example-library.with-thumbnails .wrap > label:has(input:disabled) {cursor:not-allowed; opacity:.7;}
.gradio-container:has(#task-mode input[value="Edit"]:checked) #example-library.with-thumbnails .wrap > label {
 grid-template-columns:68px minmax(0,1fr); column-gap:10px !important; row-gap:0 !important;
 padding:9px !important; align-items:center;}
.gradio-container:has(#task-mode input[value="Edit"]:checked) #example-library.with-thumbnails .wrap > label::before {
 grid-column:1; grid-row:1; width:68px; height:60px; margin:0; aspect-ratio:auto; border:0; border-radius:6px;}
.gradio-container:has(#task-mode input[value="Edit"]:checked) #example-library.with-thumbnails .wrap > label input {
 position:absolute; opacity:0; width:0;}
.gradio-container:has(#task-mode input[value="Edit"]:checked) #example-library.with-thumbnails .wrap > label span {
 grid-column:2; font-size:11px !important;}
.gradio-container:has(#task-mode input[value="Edit"]:checked) #example-library.with-thumbnails .wrap > label:focus-within {
 outline:2px solid var(--accent);}
#results {border:0; background:transparent; margin-top:12px; gap:18px;}
.result-card {padding:22px; border:1px solid var(--line); border-radius:14px; background:var(--card);}
.result-card p {line-height:1.7; font-size:15px; overflow-wrap:anywhere; margin:12px 0;}
.result-card ul, .flow-body ul {list-style:disc; padding-left:20px; margin:10px 0;}
.result-card li + li, .flow-body li + li {margin-top:4px;}
.result-card .answer-text {font-size:19px; line-height:1.65;}
.badge {display:inline-block; border:1px solid var(--line); border-radius:20px; padding:4px 9px;
 font-size:10px; line-height:1.4; font-weight:650; color:var(--muted); margin:3px 5px 3px 0;}
.badge-success {background:var(--accent-soft); color:var(--accent); border-color:transparent;}
.badge-warning {background:#fff4df; color:#805510; border-color:transparent;}
.metric-grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(135px,1fr)); gap:1px;
 border:1px solid var(--line); background:var(--line); border-radius:12px; overflow:hidden; margin:10px 0 0;}
.metric {background:var(--card); padding:13px 16px; font-size:12px; color:var(--muted); min-width:0;}
.metric strong {display:block; font-size:18px; color:var(--ink); font-weight:600; margin-bottom:4px;}
#evidence-section {border:1px solid var(--line); border-radius:14px; padding:20px; background:var(--card); gap:14px;}
#evidence-gallery {border:0 !important; background:var(--soft); border-radius:10px;}
#evidence-gallery img {object-fit:contain !important;}
#evidence-gallery .grid-wrap {height:auto !important; min-height:0 !important; max-height:none !important;}
#evidence-gallery .gallery-container:has(> .preview) {min-height:min(480px,60vh);}
#evidence-gallery .grid-container {grid-template-rows:none !important; grid-auto-rows:auto !important;}
#evidence-gallery .thumbnail-lg {aspect-ratio:16 / 11 !important; height:auto !important;}
#evidence-gallery .caption-label {font-size:10px !important;}
.evidence-grid {display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr)); gap:12px;}
.evidence-timeline {margin-top:12px; padding:14px 16px; background:var(--card); border:1px solid var(--line); border-radius:12px;}
.timeline-track {position:relative; height:14px; background:var(--soft); border:1px solid var(--line); border-radius:5px; overflow:hidden; margin:10px 0 6px;}
.timeline-window {position:absolute; height:100%; background:var(--accent); opacity:.8; border-right:2px solid var(--card);}
.timeline-labels {display:flex; justify-content:space-between; font:10px ui-monospace,monospace; color:var(--muted);}
.evidence-card {min-width:0; border:1px solid var(--line); border-radius:10px; padding:14px; font-size:13px; line-height:1.6;}
.evidence-card p, .flow-body p {margin:8px 0;}
#run-details {margin-top:12px; border:1px solid var(--line); border-radius:14px; background:var(--card); padding:8px;}
.flow-controls {display:flex; flex-wrap:wrap; gap:8px; padding:8px 10px;}
.flow-controls button {border:1px solid var(--line); border-radius:7px; padding:6px 10px;
 background:var(--card); color:var(--accent); font-size:12px; cursor:pointer;}
.flow-controls button:hover {background:var(--accent-soft);}
.flow-step {border-bottom:1px solid var(--line); color:var(--ink);}
.flow-step:last-child {border-bottom:0;}
.flow-heading {padding:13px 10px; cursor:pointer; font-size:13px; font-weight:600; overflow-wrap:anywhere;}
.flow-body {padding:0 12px 16px; font-size:13px; line-height:1.7; overflow-wrap:anywhere;}
.flow-body pre, .flow-body code {font:12px/1.65 ui-monospace,monospace; white-space:pre-wrap; overflow-wrap:anywhere;}
.flow-body pre {padding:12px; background:var(--soft); border-radius:8px; max-height:none; overflow:visible;}
.evidence-need {padding:10px 12px; border-left:2px solid var(--accent); background:var(--soft); margin:8px 0;}
.muted {color:var(--muted); font-size:12px;}
.mono {font-family:ui-monospace,monospace;}
#raw-trace {max-height:480px; overflow:auto;}
.app-footer {color:var(--muted); font-size:10px; padding-top:18px; border-top:1px solid var(--line); margin-top:20px;
 display:flex; justify-content:space-between; gap:12px;}
button:focus-visible, summary:focus-visible, label:has(input:focus-visible) {outline:2px solid var(--accent) !important; outline-offset:3px;}
@keyframes spin {to {transform:rotate(360deg);}}
@media(max-width:760px) {
 .app-header {align-items:flex-start; gap:10px; padding-bottom:17px;}
 .brand h1 {font-size:20px !important;} .brand-mark {height:34px; width:34px;}
 .model-badge {font-size:10px; padding:6px 9px; white-space:normal; max-width:145px;}
 #workspace {flex-direction:column !important; gap:14px !important;}
 #video-column, #question-column {padding:16px !important; width:100% !important; min-width:0 !important;}
 #example-library .wrap:has(> label) {grid-template-columns:repeat(2,minmax(0,1fr));}
 .metric-grid {grid-template-columns:repeat(2,minmax(0,1fr));}
 #evidence-gallery .grid-container {grid-template-columns:repeat(2,minmax(0,1fr)) !important;}
 .evidence-grid {grid-template-columns:1fr;} .result-card {padding:17px;}
 .app-footer {flex-direction:column; gap:5px;}
}
@media(prefers-reduced-motion:reduce) {* {animation:none !important; transition:none !important;}}
"""
