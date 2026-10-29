"""Small browser-runtime regressions without model downloads or a GPU."""

from pathlib import Path
from urllib.parse import urlsplit

import pytest

playwright = pytest.importorskip("playwright.sync_api")
WEB_ROOT = Path(__file__).parents[1] / "src" / "videomasys" / "web"


@pytest.fixture
def page():
    with playwright.sync_playwright() as driver:
        try:
            browser = driver.chromium.launch(args=["--no-sandbox"])
        except playwright.Error as error:
            pytest.skip(f"Chromium unavailable: {error}")
        tab = browser.new_page()

        def respond(route):
            path = urlsplit(route.request.url).path
            if path.startswith("/static/"):
                route.fulfill(path=WEB_ROOT / path.removeprefix("/static/"))
            elif path == "/v1/browser-demo/examples":
                route.fulfill(json={"videos": [], "examples": []})
            else:
                route.fulfill(path=WEB_ROOT / "index.html", content_type="text/html")

        tab.route("https://videomasys.test/**", respond)
        tab.goto("https://videomasys.test/")
        yield tab
        browser.close()


def test_template_compatibility_preserves_content_and_whitespace_controls(page):
    result = page.evaluate("""async () => {
      const {compatibleChatTemplate, renderChatPrompt} =
        await import('/static/browser-runtime-utils.js');
      const input = 'A{%- generation -%}{{ message }}{% endgeneration %}B';
      const normalized = compatibleChatTemplate(input);
      let options;
      const processor = {
        chat_template: input,
        tokenizer: {get_chat_template: ({chat_template}) => chat_template},
        apply_chat_template: (messages, opts) => { options = opts; return messages[0].content; },
      };
      const content = 'Literal {% generation %} inside the user prompt';
      const output = renderChatPrompt(processor, [{role:'user', content}]);
      return {
        normalized, output, content, options,
        unchanged: compatibleChatTemplate('plain {{ generation_count }}'),
        idempotent: compatibleChatTemplate(normalized) === normalized,
      };
    }""")
    assert result["normalized"] == "A{{- '' -}}{{ message }}{{ '' }}B"
    assert result["output"] == result["content"]
    assert result["options"]["chat_template"] == result["normalized"]
    assert result["options"]["add_generation_prompt"] is True
    assert result["options"]["tokenize"] is False
    assert result["unchanged"] == "plain {{ generation_count }}"
    assert result["idempotent"] is True


def test_lfm_intermediate_selection_requires_confirmation_before_loading(page):
    page.on("dialog", lambda dialog: dialog.dismiss())
    result = page.evaluate("""async () => {
      const selected = elements.browserModel.value;
      state.webgpuReady = true;
      let calls = 0;
      ensureBrowserWorker = () => ({postMessage: () => calls++});
      let error;
      try { await ensureBrowserModelLoaded(); } catch (e) { error = e.message; }
      return {selected, fallback:elements.browserModel.value, calls, error};
    }""")
    assert result["selected"] == "lfm-1.6b"
    assert result["fallback"] == "lfm-450m"
    assert result["calls"] == 0
    assert "cancelled" in result["error"]


def test_lfm_adapter_chunks_prefill_aligns_pixels_and_releases_tensors(page):
    result = page.evaluate("""async () => {
      const {LiquidLfmRuntime} = await import('/static/lfm-browser-runtime.js');
      const allocated = [];
      class Tensor {
        constructor(type,data,dims) {
          Object.assign(this,{type,data,dims,disposed:false}); allocated.push(this);
        }
        dispose() { if (this.disposed) throw Error('double disposal'); this.disposed = true; }
      }
      const config = {image_token_id:3, text_config:{hidden_size:2,
        num_attention_heads:1,num_key_value_heads:1,eos_token_id:7}};
      const runtime = new LiquidLfmRuntime({Tensor},config,Tensor);
      const chunks = [], masks = [], aligned = [];
      let calls = 0;
      const decoder = {inputNames:['past_conv.0','past_key_values.1.key'],run:async (feed) => {
        chunks.push(feed.inputs_embeds.dims[1]); masks.push(feed.attention_mask.dims[1]);
        aligned.push(Array.from(feed.inputs_embeds.data.slice(0,2)));
        const scores = new Float32Array(8); scores[++calls === 4 ? 7 : 4] = 10;
        return {logits:new Tensor('float32',scores,[1,1,8]),
          'present_conv.0':new Tensor('float32',new Float32Array(6),[1,2,3]),
          'present.1.key':new Tensor('float32',new Float32Array(0),[1,1,0,2])};
      }};
      runtime.sessions = [
        {run:async ({input_ids}) => ({inputs_embeds:new Tensor('float32',
          new Float32Array(input_ids.dims[1]*2),[1,input_ids.dims[1],2])})},
        {run:async () => ({image_features:
          new Tensor('float32',Float32Array.of(9,8),[1,2])})}, decoder,
      ];
      const inputs = {
        input_ids:{data:BigInt64Array.from([3,...Array(259).fill(1)],BigInt),dims:[1,260]},
        pixel_values:{type:'float32',data:new Float32Array(1),dims:[1]},
        pixel_attention_mask:{type:'int64',data:BigInt64Array.of(1n),dims:[1]},
        spatial_shapes:{type:'int64',data:BigInt64Array.of(1n,1n),dims:[1,2]},
      };
      const output = await runtime.generate({...inputs,max_new_tokens:8});
      const answer = Array.from(output.data.slice(260),Number);
      const leaked = allocated.filter(v=>v !== output && !v.disposed).length;
      output.dispose();
      inputs.input_ids.data[0] = 1n;
      let mismatch;
      try { await runtime.generate(inputs); } catch(e) { mismatch = e.message; }
      const leaksAfterFailure = allocated.filter(v=>!v.disposed).length;
      let budget;
      try { await runtime.generate({...inputs,max_new_tokens:4096}); }
      catch(e) { budget=e.message; }
      return {chunks,masks,aligned,answer,leaked,mismatch,leaksAfterFailure,budget};
    }""")
    assert result["chunks"] == [128, 128, 4, 1]
    assert result["masks"] == [128, 256, 260, 261]
    assert result["aligned"][0] == [9, 8]
    assert result["answer"] == [4, 7]
    assert result["leaked"] == result["leaksAfterFailure"] == 0
    assert "alignment mismatch" in result["mismatch"]
    assert "4096-token" in result["budget"]


def test_lfm_download_caching_and_quota_failure(page):
    result = page.evaluate("""async () => {
      const {readModelFile} = await import('/static/lfm-browser-runtime.js');
      const events=[], entries=new Map(); let fetches=0;
      const fetchFile=async () => { fetches++; return new Response(new Uint8Array([1,2,3])); };
      const cache = {match:async u=>entries.get(u)?.clone(),
        put:async (u,r)=>entries.set(u,new Response(await r.arrayBuffer()))};
      const storage={open:async()=>cache};
      const first=await readModelFile('https://test/model',e=>events.push(e),fetchFile,storage);
      const second=await readModelFile('https://test/model',()=>{},fetchFile,storage);
      const cachedFetches=fetches;
      const quota={open:async()=>({match:async()=>null,put:async()=>{throw Error('quota');}})};
      const fallback=await readModelFile('https://test/quota',()=>{},fetchFile,quota);
      return {first:[...first],second:[...second],fallback:[...fallback],
        cachedFetches,fetches,events};
    }""")
    assert result["first"] == result["second"] == result["fallback"] == [1, 2, 3]
    assert result["cachedFetches"] == 1
    assert result["fetches"] == 3
    assert result["events"][0]["loaded"] == 3


def test_downloads_are_monotonic_and_progress_has_readable_stages(page):
    result = page.evaluate("""async () => {
      const {createDownloadReporter} = await import('/static/browser-runtime-utils.js');
      const events = [];
      let clock = 0;
      const report = createDownloadReporter('model', (event) => events.push(event), () => clock);
      report({file:'large_decoder_model.onnx',loaded:1048576,progress:0.1});
      clock = 200;
      report({file:'vision_encoder.onnx',loaded:2097152,progress:1});
      clock = 400;
      report({file:'large_decoder_model.onnx',loaded:1,progress:0.001});
      handleBrowserWorkerMessage({data:events.at(-1)});
      const download = {
        label: document.querySelector('#browserProgressLabel').textContent,
        detail: document.querySelector('#browserProgressDetail').textContent,
        value: document.querySelector('#browserProgressValue').textContent,
      };
      setBrowserProgress('Reading the video', 'Scenes 1–3 of 9', 'Step 3 of 6');
      const step = document.querySelector('#browserProgressValue').textContent;
      setBrowserProgress('Analysis stopped', '', '', false);
      const active = document.querySelector('#browserProgress')
        .classList.contains('indeterminate');
      return {events, download, step, active};
    }""")
    assert [event["downloadedBytes"] for event in result["events"]] == [
        1048576, 3145728, 3145728
    ]
    assert all("file" not in event for event in result["events"])
    assert result["download"] == {
        "label": "Loading model", "detail": "3 MB downloaded", "value": ""
    }
    assert result["step"] == "Step 3 of 6"
    assert result["active"] is False


def test_json_recovery_preserves_values_and_rejects_incomplete_output(page):
    result = page.evaluate(r'''() => {
      const parse = window.VideoMASysLive.parseModelJson;
      const missingComma = parse('{"answer":"Open the gift" "selected_option":2}');
      const quotes = parse('{"answer":"He opens the "gift".","selected_option":2}');
      const trailingComma = parse('{"answer":"Open the gift","selected_option":2,}');
      const valid = parse('Preface\n```json\n{"answer":"A {gift}","selected_option":2}\n```');
      const rejected = [];
      for (const text of [
        '{"answer":"unfinished',
        '{"evidence":[{"index":0}',
        '{"answer":"one"} {"answer":"two"}',
      ]) {
        try { parse(text); rejected.push(false); } catch (_) { rejected.push(true); }
      }
      return {missingComma, quotes, trailingComma, valid, rejected};
    }''')
    assert result["missingComma"]["data"] == {
        "answer": "Open the gift", "selected_option": 2
    }
    assert result["quotes"]["data"]["answer"] == 'He opens the "gift".'
    assert result["trailingComma"]["data"]["selected_option"] == 2
    assert result["valid"]["data"]["answer"] == "A {gift}"
    assert result["missingComma"]["repaired"] is True
    assert result["valid"]["repaired"] is False
    assert result["rejected"] == [True, True, True]


def test_agent_recovers_locally_then_bounds_regeneration_and_keeps_diagnostics(page):
    result = page.evaluate(r'''async () => {
      let calls = [];
      let queued = [];
      inferBrowserAgent = async (prompt, image, limit) => {
        calls.push({prompt,image,limit});
        return {text:queued.shift(), inferenceMs:7};
      };
      state.currentTask = {steps:[{status:'running',details:{}}]};
      queued = ['{"answer":"visible" "selected_option":0}'];
      const local = await inferJsonAgent('Return answer and selected_option.', 'test-image', 100);
      const localCalls = calls.length;
      calls = [];
      queued = ['{"answer":"cut off', '{"answer":"complete","selected_option":0}'];
      const retry = await inferJsonAgent('Return answer and selected_option.', 'test-image', 100);
      const retryCalls = calls;
      calls = [];
      queued = ['not JSON', 'still not JSON', 'not JSON again'];
      let failure = null;
      try { await inferJsonAgent('Return JSON.', 'test-image', 100); }
      catch (error) { failure = error.message; }
      return {local,localCalls,retry,retryCalls,failure,failedCalls:calls.length,
        records:state.currentTask.steps[0].details.agent_responses};
    }''')
    assert result["localCalls"] == 1
    assert result["local"]["repaired"] is True
    assert result["retry"]["data"]["answer"] == "complete"
    assert len(result["retryCalls"]) == 2
    assert all(call["image"] == "test-image" for call in result["retryCalls"])
    assert all(call["limit"] == 100 for call in result["retryCalls"])
    assert "after 3 attempts" in result["failure"]
    assert result["failedCalls"] == 3
    assert len(result["records"]) == 6
    assert result["records"][0]["syntax_repaired"] is True
    assert result["records"][-1]["raw_output"] == "not JSON again"
    assert result["records"][-1]["validation"] == "failed"


def test_retry_reuses_completed_stage_outputs(page):
    result = page.evaluate('''async () => {
      const task = {steps:[],request:{question:'test'},plan:null};
      state.currentTask = task;
      let plannerCalls = 0;
      let toolCalls = 0;
      const plan = () => {plannerCalls++; return {value:{objective:'test'}};};
      await runLiveStep(task,'plan','planner',plan);
      try {
        await runLiveStep(task,'tool_sequence','tool_sequence',() => {
          toolCalls++; throw new Error('test failure');
        });
      } catch (_) {}
      const recoveredPlan = await runLiveStep(task,'plan','planner',plan);
      await runLiveStep(task,'tool_sequence','tool_sequence',() => {
        toolCalls++; return {value:{locate:[]}};
      });
      return {plannerCalls,toolCalls,recoveredPlan,statuses:task.steps.map(step=>step.status)};
    }''')
    assert result["plannerCalls"] == 1
    assert result["toolCalls"] == 2
    assert result["recoveredPlan"] == {"objective": "test"}
    assert result["statuses"] == ["succeeded", "succeeded"]


def test_pipeline_button_resumes_after_exhausted_tool_json_retries(page):
    # Exercise the real orchestration with isolated model/media test doubles.
    result = page.evaluate('''async () => {
      state.currentVideo = {id:'test-video',status:'ready',duration_seconds:10};
      state.browserDemo = true;
      state.webgpuReady = true;
      elements.question.value = 'Test question';
      ensureBrowserModelLoaded = async () => {};
      getBrowserScenes = async () => [{index:0,start_seconds:0,end_seconds:10,
        caption:'test scene',frame_urls:['test-frame']}];
      captionScenes = async scenes => ({scenes,cacheHit:false,modelMs:0});
      embedBrowserTexts = async () => {throw new Error('test lexical fallback');};
      buildEvidenceContactSheet = async () => ({imageUrl:'test-image',labels:['T1']});
      const calls = {planner:0,tool:0,inspect:0,answer:0,verify:0};
      let allowTool = false;
      inferBrowserAgent = async prompt => {
        let data;
        if (prompt.includes('You are the reasoning planner')) {
          calls.planner++;
          data = {objective:'test',evidence_needs:[
            {id:'anchor',description:'test scene',temporal_role:'anchor',depends_on:[]}]};
        } else if (prompt.includes('You are the tool-sequence agent')) {
          calls.tool++;
          if (!allowTool) return {text:'invalid test output',inferenceMs:1};
          data = {locate:[{id:'l1',query:'test scene',evidence_need_ids:['anchor']}],
            relate:[],inspect:{sources:['l1']}};
        } else if (prompt.includes('You are the visual evidence inspector')) {
          calls.inspect++;
          data = {evidence:[{scene_index:0,relevant:true,observation:'test observation'}]};
        } else if (prompt.includes('You are the evidence-grounded answer composer')) {
          calls.answer++;
          data = {answer:'test answer',selected_option:null,supporting_scene_indices:[0]};
        } else if (prompt.includes('You are the final grounding verifier')) {
          calls.verify++;
          data = {accepted:true,confidence:0,issues:[]};
        } else throw new Error('unexpected test prompt');
        return {text:JSON.stringify(data),inferenceMs:1};
      };
      await runLiveBrowserPipeline();
      const firstId = state.currentTask.id;
      const paused = state.currentTask.status;
      const retryLabel = elements.run.textContent.trim();
      const failedResponses = state.currentTask.steps[1].details.agent_responses.length;
      allowTool = true;
      await runLiveBrowserPipeline();
      return {paused,retryLabel,failedResponses,calls,sameTask:firstId===state.currentTask.id,
        status:state.currentTask.status,stepCount:state.currentTask.steps.length};
    }''')
    assert result["paused"] == "failed"
    assert result["retryLabel"] == "Retry failed stage"
    assert result["failedResponses"] == 3
    assert result["calls"] == {"planner": 1, "tool": 4, "inspect": 1, "answer": 1, "verify": 1}
    assert result["sameTask"] is True
    assert result["status"] == "succeeded"
    assert result["stepCount"] == 6
