(() => {
  "use strict";

  const clamp = (value, low, high) => Math.max(low, Math.min(high, Number(value)));

  function optionsText(options) {
    return options?.length
      ? options.map((option, index) => `${index}: ${option}`).join("\n")
      : "Open-ended question";
  }

  function plannerPrompt(question, options, durationSeconds) {
    return `You are the reasoning planner in a retrieval-first video QA system.
You do not answer the question and you do not choose tools. Decide which independently
observable facts must be established. For causal, purpose, before/after, or multi-step
questions, separate the anchor event from the outcome and encode their dependency.
Do not assume any answer choice is correct. Keep the visual budget small.

Video duration: ${durationSeconds.toFixed(1)} seconds
Question: ${question}
Choices:
${optionsText(options)}

Return only one JSON object, with no markdown, using exactly this shape:
{
  "objective":"specific reasoning objective",
  "evidence_needs":[
    {"id":"anchor","description":"observable fact to find","temporal_role":"anchor","depends_on":[]}
  ],
  "temporal_constraints":["plain-language constraint"],
  "option_strategy":"how evidence will discriminate choices",
  "max_intervals":3,
  "max_inspected_seconds":60,
  "stop_condition":"when evidence is sufficient",
  "reasoning":"why these facts and ordering are necessary"
}
Prefer 1-2 evidence needs; use at most 4. A dependency may reference only an earlier need id.
temporal_role is one of: anchor, before, after, outcome, context.
Keep each description and rationale under 20 words. Use simple sentences without quoted speech.`;
  }

  function toolProgramPrompt(question, options, plan) {
    return `You are the tool-sequence agent. Compile a validated reasoning plan into
typed retrieval calls. Do not answer the question.

Question: ${question}
Choices:
${optionsText(options)}
Reasoning plan:
${JSON.stringify(plan, null, 2)}

Available functions:
- locate(id, query, limit, evidence_need_ids): vector search over cached scene captions.
- relate(id, source, relation, horizon_seconds, limit, evidence_need_ids): traverse
  before, after, during, or around the scenes returned by an earlier call.
- inspect(sources, evidence_need_ids): materialize a bounded union for visual inspection.

Return only one JSON object, with no markdown:
{
  "version":"2.0",
  "compiler":"model",
  "rationale":"why these calls implement the plan",
  "max_intervals":3,
  "max_inspected_seconds":60,
  "locate":[{"id":"l1","query":"standalone visual search query","limit":4,"evidence_need_ids":["need_id"]}],
  "relate":[{"id":"r1","source":"l1","relation":"after","horizon_seconds":60,"limit":3,"evidence_need_ids":["need_id"]}],
  "inspect":{"sources":["l1","r1"],"evidence_need_ids":["need_id"]}
}
Use sequential l1..l4 and r1..r6 ids. Every evidence need must be covered by locate
and inspect. Every relation source must reference an earlier output. Respect both
budgets from the reasoning plan. Queries must describe visible content, not option ids.
Keep queries and rationale under 20 words each. Use the fewest calls needed.`;
  }

  function sceneCaptionPrompt(sceneMap, labels) {
    return `You are building a reusable video scene cache. The contact sheet contains
chronological frames labelled ${labels.join(", ")}. These frame-to-scene groups apply:
${sceneMap.map((item) => `scene ${item.index}: ${item.labels.join(", ")} (${item.start.toFixed(1)}-${item.end.toFixed(1)}s)`).join("\n")}

For every listed scene, write a literal retrieval caption describing visible people,
objects, actions, state changes, and interaction order. Do not answer any downstream
question and do not infer intent. Mention uncertainty when sparse frames cannot prove
motion. Return only JSON:
{"scenes":[{"index":0,"caption":"literal chronological description","entities":["entity"],"actions":["action"]}]}`;
  }

  function inspectorPrompt(question, options, selectedScenes, labels) {
    return `You are the visual evidence inspector. The contact sheet contains only
vector-retrieved video frames, in chronological panel order, labelled ${labels.join(", ")}.
Panel groups:
${selectedScenes.map((item) => `scene ${item.index}: ${item.labels.join(", ")} (${item.start_seconds.toFixed(1)}-${item.end_seconds.toFixed(1)}s), cache hint: ${item.caption}`).join("\n")}

Question: ${question}
Choices:
${optionsText(options)}

Inspect pixels rather than trusting cache hints. For each scene, state only relevant
visible evidence. Compare state changes and discriminate all choices. Do not compose
the final answer. Return only JSON:
{"evidence":[{"scene_index":0,"relevant":true,"observation":"grounded chronological observation","confidence":0.0}]}`;
  }

  function answerPrompt(question, options, evidence) {
    return `You are the evidence-grounded answer composer. Use only the observations
below. Do not rely on world knowledge to invent unseen events. For why/purpose questions,
prefer a later visible outcome over guessing intent. For multiple choice, compare all
choices and return the zero-based index whose wording is best supported.

Question: ${question}
Choices:
${optionsText(options)}
Evidence:
${evidence.map((item) => `scene ${item.scene_index} (${item.start_seconds.toFixed(1)}-${item.end_seconds.toFixed(1)}s): ${item.observation}`).join("\n")}

Return only JSON:
{"answer":"direct answer","selected_option":null,"confidence":0.0,"supporting_scene_indices":[0],"limitations":[]}`;
  }

  function verifierPrompt(question, options, evidence, answer) {
    return `You are the final grounding verifier. Check the draft against every supplied
observation and answer choice. Reject unsupported causal claims, wrong option/text
pairings, and claims outside the retrieved evidence. Do not reward confident wording.

Question: ${question}
Choices:
${optionsText(options)}
Evidence:
${evidence.map((item) => `scene ${item.scene_index}: ${item.observation}`).join("\n")}
Draft:
${JSON.stringify(answer, null, 2)}

Return only JSON:
{"accepted":true,"issues":[],"confidence":0.0,"corrected_answer":null,"corrected_selected_option":null}`;
  }

  function parseModelJson(text, hitTokenLimit = false) {
    const cleaned = String(text || "").replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
    const fenced = cleaned.match(/```(?:json)?\s*([\s\S]*?)```/i)?.[1];
    const candidate = (fenced || cleaned).trim();
    const start = candidate.indexOf("{");
    if (start < 0) throw new Error("Model did not return a JSON object.");
    // Find a complete object without treating braces in strings as delimiters.
    // Never ask syntax repair to finish a truncated response or invent its tail.
    const stack = [];
    let quoted = false;
    let escaped = false;
    let end = -1;
    for (let index = start; index < candidate.length; index += 1) {
      const char = candidate[index];
      if (quoted) {
        if (escaped) escaped = false;
        else if (char === "\\") escaped = true;
        else if (char === '"') quoted = false;
      } else if (char === '"') quoted = true;
      else if (char === "{" || char === "[") stack.push(char);
      else if (char === "}" || char === "]") {
        if (stack.pop() !== (char === "}" ? "{" : "[")) break;
        if (!stack.length) { end = index + 1; break; }
      }
    }
    if (end < 0) throw new Error(hitTokenLimit
      ? "Model response reached its token limit before completing JSON."
      : "Model returned an incomplete JSON object.");
    if (/^[\s,]*[\[{]/.test(candidate.slice(end))) throw new Error("Model returned multiple JSON objects.");
    const source = candidate.slice(start, end);
    try {
      return { data: JSON.parse(source), repaired: false, normalized: source };
    } catch (error) {
      if (!hitTokenLimit && window.JSONRepair?.jsonrepair) {
        try {
          const normalized = window.JSONRepair.jsonrepair(source);
          const data = JSON.parse(normalized);
          if (data && typeof data === "object" && !Array.isArray(data)) {
            return { data, repaired: true, normalized };
          }
        } catch (_) { /* use model regeneration when local syntax repair fails */ }
      }
      throw new Error(`Model returned invalid JSON: ${error.message}`);
    }
  }

  function parseJson(text) {
    return parseModelJson(text).data;
  }

  function validatePlan(raw, durationSeconds) {
    if (!raw || typeof raw.objective !== "string" || !Array.isArray(raw.evidence_needs)) {
      throw new Error("Planner output is missing objective or evidence_needs.");
    }
    if (raw.evidence_needs.length < 1 || raw.evidence_needs.length > 4) {
      throw new Error("Planner must return 1-4 evidence needs.");
    }
    const ids = new Set();
    const idMap = new Map();
    const normalizedIds = raw.evidence_needs.map((need, index) => {
      const original = String(need.id ?? index + 1);
      let id = original.toLowerCase().replace(/[^a-z0-9_]+/g, "_").replace(/^_+|_+$/g, "");
      if (!/^[a-z]/.test(id)) id = `evidence_${index + 1}`;
      id = id.slice(0, 32) || `evidence_${index + 1}`;
      const base = id;
      let suffix = 2;
      while (ids.has(id)) {
        id = `${base.slice(0, 28)}_${suffix}`;
        suffix += 1;
      }
      ids.add(id);
      // Preserve an explicit model id exactly. Numeric fallbacks are resolved only
      // after all ids are known, avoiding the 0-based/1-based collision at "1".
      if (!idMap.has(original)) idMap.set(original, id);
      return id;
    });
    const known = new Set();
    const needs = raw.evidence_needs.map((need, index) => {
      const id = normalizedIds[index];
      const dependsOn = (Array.isArray(need.depends_on) ? need.depends_on : [])
        .map((dependency) => {
          const text = String(dependency);
          if (idMap.has(text)) return idMap.get(text);
          const numeric = Number(text);
          return Number.isInteger(numeric) ? normalizedIds[numeric - 1] || text : text;
        })
        .filter((dependency) => known.has(dependency));
      known.add(id);
      return {
        id,
        description: String(need.description || "").trim(),
        temporal_role: ["anchor", "before", "after", "outcome", "context"].includes(need.temporal_role)
          ? need.temporal_role : "context",
        depends_on: dependsOn,
      };
    });
    if (needs.some((need) => need.description.length < 3)) {
      throw new Error("Every planner evidence need requires a concrete description.");
    }
    return {
      objective: raw.objective.trim(),
      evidence_needs: needs,
      temporal_constraints: Array.isArray(raw.temporal_constraints) ? raw.temporal_constraints.map(String) : [],
      option_strategy: raw.option_strategy == null ? null : String(raw.option_strategy),
      max_intervals: Math.round(clamp(raw.max_intervals || 3, 1, 6)),
      max_inspected_seconds: clamp(raw.max_inspected_seconds || 60, 5, Math.min(180, durationSeconds)),
      stop_condition: String(raw.stop_condition || "").trim(),
      reasoning: String(raw.reasoning || "").trim(),
    };
  }

  function validateProgram(raw, plan) {
    if (!raw || !Array.isArray(raw.locate) || !raw.inspect) {
      throw new Error("Tool agent output is missing locate or inspect calls.");
    }
    const needIdList = plan.evidence_needs.map((need) => need.id);
    const needIds = new Set(needIdList);
    const normalizeNeedId = (value, fallbackIndex) => {
      const text = String(value);
      if (needIds.has(text)) return text;
      const numeric = Number(text);
      if (Number.isInteger(numeric)) {
        return needIdList[numeric - 1] || needIdList[numeric] || needIdList[fallbackIndex];
      }
      return needIdList[fallbackIndex];
    };
    const known = new Set();
    const outputIdMap = new Map();
    const controllerRepairs = [];
    const locate = raw.locate.slice(0, 4).map((call, index) => {
      const id = `l${index + 1}`;
      const rawNeedIds = Array.isArray(call.evidence_need_ids) && call.evidence_need_ids.length
        ? call.evidence_need_ids : [needIdList[index % needIdList.length]];
      const evidenceNeedIds = [...new Set(rawNeedIds.map((value) => normalizeNeedId(value, index % needIdList.length)))];
      if (!evidenceNeedIds.length || evidenceNeedIds.some((value) => !needIds.has(value))) {
        throw new Error(`${id} references invalid evidence needs.`);
      }
      known.add(id);
      outputIdMap.set(String(call.id ?? id), id);
      outputIdMap.set(id, id);
      return {
        id,
        query: String(call.query || "").trim(),
        limit: Math.round(clamp(call.limit || 4, 2, 8)),
        evidence_need_ids: evidenceNeedIds,
      };
    });
    if (!locate.length || locate.some((call) => call.query.length < 3)) {
      throw new Error("Tool agent must provide at least one concrete locate query.");
    }
    const covered = new Set(locate.flatMap((call) => call.evidence_need_ids));
    for (const need of plan.evidence_needs) {
      if (covered.has(need.id)) continue;
      if (locate.length >= 4) {
        const target = locate.at(-1);
        target.query = `${target.query}; ${need.description}`;
        target.evidence_need_ids.push(need.id);
        covered.add(need.id);
        controllerRepairs.push(`Extended ${target.id} because the model omitted locate coverage for ${need.id}.`);
        continue;
      }
      const id = `l${locate.length + 1}`;
      locate.push({ id, query: need.description, limit: 4, evidence_need_ids: [need.id] });
      covered.add(need.id);
      known.add(id);
      outputIdMap.set(id, id);
      controllerRepairs.push(`Added ${id} because the model omitted locate coverage for ${need.id}.`);
    }
    const relate = [];
    for (const call of (Array.isArray(raw.relate) ? raw.relate : []).slice(0, 6)) {
      const id = `r${relate.length + 1}`;
      const source = outputIdMap.get(String(call.source || "")) || String(call.source || "");
      if (!known.has(source)) {
        controllerRepairs.push(`Dropped relation with unknown source ${source || "(empty)"}.`);
        continue;
      }
      known.add(id);
      outputIdMap.set(String(call.id ?? id), id);
      outputIdMap.set(id, id);
      relate.push({
        id,
        source,
        relation: ["before", "after", "during", "around"].includes(call.relation) ? call.relation : "around",
        horizon_seconds: clamp(call.horizon_seconds || 60, 5, 180),
        limit: Math.round(clamp(call.limit || 3, 1, 8)),
        evidence_need_ids: [...new Set((call.evidence_need_ids || []).map((value) => normalizeNeedId(value, 0)))],
      });
    }

    // Deterministically restore temporal traversals required by planner dependencies.
    const locateForNeed = new Map();
    for (const call of locate) {
      for (const needId of call.evidence_need_ids) {
        if (!locateForNeed.has(needId)) locateForNeed.set(needId, call.id);
      }
    }
    for (const need of plan.evidence_needs) {
      for (const dependency of need.depends_on) {
        if (relate.length >= 6) break;
        const alreadyCovered = relate.some((call) => call.evidence_need_ids.includes(need.id));
        const source = locateForNeed.get(dependency);
        if (alreadyCovered || !source) continue;
        const id = `r${relate.length + 1}`;
        const relation = need.temporal_role === "before" ? "before" : "after";
        relate.push({ id, source, relation, horizon_seconds: 60, limit: 3, evidence_need_ids: [need.id] });
        known.add(id);
        outputIdMap.set(id, id);
        controllerRepairs.push(`Added ${id} to enforce ${dependency} → ${need.id}.`);
      }
    }
    const requestedSources = (Array.isArray(raw.inspect.sources) ? raw.inspect.sources : [])
      .map((source) => outputIdMap.get(String(source)) || String(source))
      .filter((source) => known.has(source));
    const sources = [...new Set(requestedSources)];
    const outputsByPreference = [...relate, ...locate];
    for (const needId of needIds) {
      const alreadyCovered = sources.some((source) =>
        outputsByPreference.find((call) => call.id === source)?.evidence_need_ids.includes(needId)
      );
      if (alreadyCovered) continue;
      const output = outputsByPreference.find((call) => call.evidence_need_ids.includes(needId));
      if (output) sources.push(output.id);
    }
    if (!requestedSources.length) controllerRepairs.push("Rebuilt inspect sources from validated retrieval outputs.");
    return {
      version: "2.0",
      compiler: controllerRepairs.length ? "model_repaired" : "model",
      rationale: String(raw.rationale || "Model supplied no rationale."),
      max_intervals: Math.min(plan.max_intervals, Math.round(clamp(raw.max_intervals || plan.max_intervals, 1, 6))),
      max_inspected_seconds: Math.min(plan.max_inspected_seconds, clamp(raw.max_inspected_seconds || plan.max_inspected_seconds, 5, 180)),
      locate,
      relate,
      inspect: { sources, evidence_need_ids: [...needIds] },
      controller_repairs: controllerRepairs,
    };
  }

  function tokens(text) {
    const stop = new Set(["the", "a", "an", "is", "are", "was", "were", "to", "of", "in", "on", "and", "or", "with", "what", "why", "did", "does", "do", "after", "before"]);
    return String(text || "").toLowerCase().match(/[a-z0-9]+/g)?.filter((token) => token.length > 1 && !stop.has(token)) || [];
  }

  function vectorize(text, documentFrequency, documentCount) {
    const counts = new Map();
    const values = tokens(text);
    const terms = [...values, ...values.slice(0, -1).map((token, index) => `${token}_${values[index + 1]}`)];
    for (const term of terms) counts.set(term, (counts.get(term) || 0) + 1);
    const vector = new Map();
    for (const [term, count] of counts) {
      const idf = Math.log((documentCount + 1) / ((documentFrequency.get(term) || 0) + 1)) + 1;
      vector.set(term, (1 + Math.log(count)) * idf);
    }
    const norm = Math.sqrt([...vector.values()].reduce((total, value) => total + value * value, 0)) || 1;
    for (const [term, value] of vector) vector.set(term, value / norm);
    return vector;
  }

  function cosine(left, right) {
    let score = 0;
    for (const [term, value] of left) score += value * (right.get(term) || 0);
    return score;
  }

  function executeRetrieval(program, scenes, denseVectors = null) {
    const documentFrequency = new Map();
    for (const scene of scenes) {
      for (const term of new Set(tokens(scene.caption))) {
        documentFrequency.set(term, (documentFrequency.get(term) || 0) + 1);
      }
    }
    const sceneVectors = scenes.map((scene, index) => ({
      scene,
      vector: denseVectors?.sceneVectors?.[index]
        || vectorize(scene.caption, documentFrequency, scenes.length),
    }));
    const similarity = denseVectors
      ? (left, right) => left.reduce((total, value, index) => total + value * (right[index] || 0), 0)
      : cosine;
    const outputs = new Map();
    const trace = [];
    for (const [callIndex, call] of program.locate.entries()) {
      const queryVector = denseVectors?.queryVectors?.[callIndex]
        || vectorize(call.query, documentFrequency, scenes.length);
      const hits = sceneVectors.map(({ scene, vector }) => ({
        scene,
        score: similarity(queryVector, vector),
        source: call.id,
      })).sort((left, right) => right.score - left.score).slice(0, call.limit);
      outputs.set(call.id, hits);
      trace.push({ call: call.id, tool: "locate", query: call.query, hits: hits.map((hit) => ({ scene: hit.scene.index, score: hit.score })) });
    }
    for (const call of program.relate) {
      const sourceHits = outputs.get(call.source) || [];
      const related = [];
      for (const candidate of scenes) {
        let best = -1;
        for (const hit of sourceHits) {
          const distance = call.relation === "after"
            ? candidate.start_seconds - hit.scene.end_seconds
            : call.relation === "before"
              ? hit.scene.start_seconds - candidate.end_seconds
              : Math.abs((candidate.start_seconds + candidate.end_seconds) / 2 - (hit.scene.start_seconds + hit.scene.end_seconds) / 2);
          const valid = ["during", "around"].includes(call.relation)
            ? distance <= call.horizon_seconds
            : distance >= 0 && distance <= call.horizon_seconds;
          if (valid) best = Math.max(best, 1 - distance / Math.max(call.horizon_seconds, 1));
        }
        if (best >= 0) related.push({ scene: candidate, score: best, source: call.id });
      }
      related.sort((left, right) => right.score - left.score);
      const hits = related.slice(0, call.limit);
      outputs.set(call.id, hits);
      trace.push({ call: call.id, tool: `relate(${call.relation})`, source: call.source, hits: hits.map((hit) => ({ scene: hit.scene.index, score: hit.score })) });
    }
    const union = new Map();
    for (const source of program.inspect.sources) {
      for (const hit of outputs.get(source) || []) {
        const current = union.get(hit.scene.index);
        if (!current || hit.score > current.score) union.set(hit.scene.index, hit);
      }
    }
    const ranked = [...union.values()].sort((left, right) => right.score - left.score);
    const selected = [];
    let seconds = 0;
    for (const hit of ranked) {
      const duration = hit.scene.end_seconds - hit.scene.start_seconds;
      if (selected.length >= program.max_intervals) break;
      if (selected.length && seconds + duration > program.max_inspected_seconds) continue;
      selected.push(hit.scene);
      seconds += duration;
    }
    selected.sort((left, right) => left.start_seconds - right.start_seconds);
    trace.push({ call: "inspect", tool: "budgeted_union", hits: selected.map((scene) => ({ scene: scene.index })) });
    return {
      selected,
      trace,
      index: {
        kind: denseVectors ? "in-browser MiniLM semantic vector index" : "in-browser TF-IDF vector index",
        model: denseVectors?.modelId || null,
        dimensions: denseVectors?.dimensions || documentFrequency.size,
        scene_count: scenes.length,
      },
    };
  }

  window.VideoMASysLive = {
    answerPrompt,
    executeRetrieval,
    inspectorPrompt,
    parseJson,
    parseModelJson,
    plannerPrompt,
    sceneCaptionPrompt,
    toolProgramPrompt,
    validatePlan,
    validateProgram,
    verifierPrompt,
  };
})();
