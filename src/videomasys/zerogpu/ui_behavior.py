"""Client-only interaction state; never changes pipeline data or model outputs."""

EMBED_HEIGHT_JS = """
let resizePending = false;
let layoutRevision = 0;
function refreshEmbedHeight() {
    if (window.parent === window || resizePending || !element.isConnected) return;
    resizePending = true;
    requestAnimationFrame(() => {
        resizePending = false;
        // Gradio bounds consecutive iframe growth to prevent resize loops.
        // A new result/native details toggle is real content growth, just like
        // opening a Gradio Accordion: reset its guard through the same event.
        trigger('gradio_expand');
        // Wake the layout observer AFTER that reset, even if restoring an
        // already-closed step didn't change any DOM attributes.
        element.dataset.layoutRevision = String(++layoutRevision);
    });
}
if (window.parent !== window) {
    let lastWidth = window.innerWidth;
    window.addEventListener('resize', () => {
        // The parent's height response also fires resize. Only width changes
        // represent a new wrapping layout; never reset the guard for height.
        if (window.innerWidth === lastWidth) return;
        lastWidth = window.innerWidth;
        refreshEmbedHeight();
    });
}
"""

FLOW_JS = EMBED_HEIGHT_JS + """
// Gradio patches HTML attributes on each streamed value, including `open`.
// Keep the reader's choices outside that HTML and restore them after the patch.
const states = new Map();
const steps = () => element.querySelectorAll('details[data-step-key]');
function restoreSteps() {
    if (!props.value) {
        states.clear();
        refreshEmbedHeight();
        return;
    }
    for (const step of steps()) {
        const key = step.dataset.stepKey;
        if (states.has(key)) step.open = states.get(key);
        else states.set(key, step.open);
    }
    refreshEmbedHeight();
}
element.addEventListener('toggle', (event) => {
    const step = event.target;
    if (step.matches('details[data-step-key]') && element.contains(step)) {
        states.set(step.dataset.stepKey, step.open);
        refreshEmbedHeight();
    }
}, true);
element.addEventListener('click', (event) => {
    const button = event.target.closest('button[data-flow-action]');
    if (!button || !element.contains(button)) return;
    const open = button.dataset.flowAction === 'expand';
    for (const step of steps()) {
        states.set(step.dataset.stepKey, open);
        step.open = open;
    }
    refreshEmbedHeight();
});
watch('value', restoreSteps);
restoreSteps();
"""

EVIDENCE_JS = EMBED_HEIGHT_JS + """
// Cached-caption disclosures are native details elements, too.
element.addEventListener('toggle', refreshEmbedHeight, true);
watch('value', refreshEmbedHeight);
refreshEmbedHeight();
"""
