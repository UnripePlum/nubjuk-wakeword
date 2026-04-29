INDEX_HTML = """
<!doctype html>
<html lang="ko" data-theme="dark">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Wakeword Studio</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Serif:wght@400;500&display=swap" rel="stylesheet" />
  <style>
    :root {
      --bg: #0a0c0f;
      --bg-1: #0d1014;
      --bg-2: #11151a;
      --bg-3: #161b22;
      --line: #1c232c;
      --line-2: #262f3a;
      --ink: #e6edf3;
      --ink-2: #b6c2cf;
      --ink-3: #8a96a3;
      --ink-4: #5a6470;
      --ink-5: #3a424c;
      --accent: #22d3ee;
      --accent-glow: rgba(34, 211, 238, .15);
      --ok: #4ade80;
      --warn: #fbbf24;
      --err: #f87171;
    }
    [data-theme="light"] {
      --bg: #f7f7f4;
      --bg-1: #ffffff;
      --bg-2: #fafaf7;
      --bg-3: #f0f0eb;
      --line: #e4e4dc;
      --line-2: #d4d4cb;
      --ink: #0d1014;
      --ink-2: #2a2f36;
      --ink-3: #4a525c;
      --ink-4: #7a8390;
      --ink-5: #b0b6bd;
      --accent: #0891b2;
      --accent-glow: rgba(8, 145, 178, .12);
      --ok: #16a34a;
      --warn: #d97706;
      --err: #dc2626;
    }
    * { box-sizing: border-box; }
    html, body { margin: 0; min-height: 100%; }
    body {
      font-family: "IBM Plex Sans", system-ui, sans-serif;
      background: var(--bg);
      color: var(--ink);
      font-size: 14px;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }
    .mono { font-family: "JetBrains Mono", ui-monospace, monospace; }
    .serif { font-family: "IBM Plex Serif", Georgia, serif; }
    button, input, textarea, select { font: inherit; }
    button { cursor: pointer; }
    button:disabled { cursor: not-allowed; opacity: .42; }
    input:focus, textarea:focus, button:focus-visible {
      outline: 3px solid var(--accent-glow);
      outline-offset: 2px;
    }
    .app {
      display: grid;
      grid-template-columns: 240px minmax(560px, 1fr) 320px;
      grid-template-rows: 48px 1fr;
      height: 100vh;
      width: 100vw;
      overflow: hidden;
    }
    .topbar {
      grid-column: 1 / -1;
      display: flex;
      align-items: center;
      gap: 16px;
      padding: 0 16px;
      background: var(--bg-1);
      border-bottom: 1px solid var(--line);
      font-family: "JetBrains Mono", monospace;
      font-size: 12px;
    }
    .logo { display: flex; align-items: center; gap: 10px; font-weight: 700; letter-spacing: .02em; }
    .logo-dot {
      width: 8px; height: 8px; border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 8px var(--accent);
      animation: pulse 2s ease-in-out infinite;
    }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: .4; } }
    .crumbs { display: flex; align-items: center; gap: 6px; color: var(--ink-3); }
    .sep { color: var(--ink-5); }
    .topbar-right { margin-left: auto; display: flex; align-items: center; gap: 14px; color: var(--ink-3); }
    .meta-k { color: var(--ink-4); }
    .meta-v { color: var(--ink-2); }
    .theme-toggle {
      border: 1px solid var(--line-2);
      background: var(--bg-2);
      color: var(--ink-2);
      border-radius: 3px;
      padding: 5px 8px;
      font-family: "JetBrains Mono", monospace;
      font-size: 11px;
    }
    .rail {
      background: var(--bg-1);
      border-right: 1px solid var(--line);
      overflow-y: auto;
      padding: 16px 0;
    }
    .rail-title, .rail-section, .aside-title, .card-title, .label {
      font-family: "JetBrains Mono", monospace;
      font-size: 10px;
      letter-spacing: .16em;
      text-transform: uppercase;
      color: var(--ink-4);
    }
    .rail-title { padding: 0 16px 12px; }
    .stage {
      display: flex;
      align-items: center;
      gap: 12px;
      padding: 10px 16px;
      color: var(--ink-3);
      font-size: 13px;
      border-left: 2px solid transparent;
    }
    .stage.current {
      color: var(--ink);
      background: var(--bg-2);
      border-left-color: var(--accent);
    }
    .stage.done { color: var(--ink-2); }
    .stage.locked { color: var(--ink-5); }
    .stage-num {
      width: 22px; height: 22px;
      display: inline-flex; align-items: center; justify-content: center;
      border: 1px solid var(--line-2);
      border-radius: 50%;
      color: var(--ink-4);
      font-family: "JetBrains Mono", monospace;
      font-size: 11px;
      flex-shrink: 0;
    }
    .stage.done .stage-num { border-color: var(--ok); color: var(--ok); background: rgba(74, 222, 128, .08); }
    .stage.current .stage-num { border-color: var(--accent); color: var(--accent); background: var(--accent-glow); }
    .rail-divider { height: 1px; background: var(--line); margin: 16px; }
    .rail-section { padding: 8px 16px; }
    .rail-link { display: flex; align-items: center; gap: 10px; padding: 8px 16px; color: var(--ink-3); font-size: 13px; }
    .rail-link.active { color: var(--ink); background: var(--bg-2); }
    .main { overflow-y: auto; background: var(--bg); }
    .main-inner { max-width: 920px; margin: 0 auto; padding: 32px 40px 80px; }
    .page-header { margin-bottom: 28px; }
    .page-eyebrow {
      font-family: "JetBrains Mono", monospace;
      font-size: 11px;
      letter-spacing: .18em;
      color: var(--accent);
      text-transform: uppercase;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .page-eyebrow::before { content: ""; width: 16px; height: 1px; background: var(--accent); }
    .page-title { font-size: 28px; font-weight: 600; letter-spacing: -.01em; margin: 0 0 10px; }
    .page-sub { color: var(--ink-3); font-size: 15px; max-width: 62ch; }
    .aside {
      background: var(--bg-1);
      border-left: 1px solid var(--line);
      overflow-y: auto;
      display: flex;
      flex-direction: column;
    }
    .aside-section { padding: 16px; border-bottom: 1px solid var(--line); }
    .aside-title { margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }
    .kv-grid { display: grid; grid-template-columns: auto 1fr; gap: 8px 12px; font-size: 12px; }
    .kv-grid .k { color: var(--ink-4); font-family: "JetBrains Mono", monospace; font-size: 11px; text-transform: uppercase; }
    .kv-grid .v { color: var(--ink-2); font-family: "JetBrains Mono", monospace; word-break: break-all; }
    .card { background: var(--bg-1); border: 1px solid var(--line); border-radius: 4px; overflow: hidden; margin-bottom: 16px; }
    .card-header { padding: 14px 18px; border-bottom: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .card-body { padding: 18px; }
    .btn {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 9px 16px;
      background: var(--bg-2);
      border: 1px solid var(--line-2);
      color: var(--ink);
      font-family: "JetBrains Mono", monospace;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: .02em;
      border-radius: 3px;
      text-transform: uppercase;
    }
    .btn:hover:not(:disabled) { border-color: var(--ink-4); background: var(--bg-3); }
    .btn.primary { background: var(--accent); border-color: var(--accent); color: #001218; }
    .btn.danger { border-color: rgba(248, 113, 113, .4); color: var(--err); }
    .btn.ghost { background: transparent; border-color: transparent; color: var(--ink-3); }
    .btn.sm { padding: 6px 10px; font-size: 11px; }
    .input, .textarea, .select {
      width: 100%;
      background: var(--bg);
      border: 1px solid var(--line-2);
      color: var(--ink);
      padding: 10px 12px;
      border-radius: 3px;
      font-family: "JetBrains Mono", monospace;
      font-size: 13px;
    }
    .input.lg { font-size: 18px; padding: 14px 16px; }
    .field { margin-bottom: 18px; }
    .label { display: block; margin-bottom: 6px; }
    .help { color: var(--ink-4); font-size: 12px; margin-top: 6px; }
    .field-row { display: flex; gap: 16px; }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 3px 8px;
      border-radius: 3px;
      font-family: "JetBrains Mono", monospace;
      font-size: 10px;
      letter-spacing: .06em;
      text-transform: uppercase;
      font-weight: 700;
    }
    .badge.ok { background: rgba(74, 222, 128, .12); color: var(--ok); }
    .badge.warn { background: rgba(251, 191, 36, .12); color: var(--warn); }
    .badge.err { background: rgba(248, 113, 113, .12); color: var(--err); }
    .badge.run { background: var(--accent-glow); color: var(--accent); }
    .badge.idle { background: var(--bg-3); color: var(--ink-4); border: 1px solid var(--line-2); }
    .chip, .word-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 10px 6px 12px;
      background: var(--bg-2);
      border: 1px solid var(--line-2);
      border-radius: 3px;
      font-family: "JetBrains Mono", monospace;
      font-size: 12px;
      color: var(--ink-2);
    }
    .word-list { display: flex; flex-wrap: wrap; gap: 8px; }
    .word-chip { cursor: pointer; min-height: 34px; }
    .word-chip input { width: auto; accent-color: var(--accent); }
    .word-chip:has(input:checked) { border-color: var(--accent); color: var(--ink); background: var(--accent-glow); }
    .alert { padding: 12px 14px; border-radius: 3px; border-left: 2px solid; background: var(--bg-2); font-size: 13px; color: var(--ink-2); }
    .alert.info { border-left-color: var(--accent); }
    .alert.warn { border-left-color: var(--warn); }
    .alert.err { border-left-color: var(--err); }
    .alert-title { font-family: "JetBrains Mono", monospace; font-size: 11px; text-transform: uppercase; letter-spacing: .08em; margin-bottom: 4px; font-weight: 700; }
    .alert.info .alert-title { color: var(--accent); }
    .alert.warn .alert-title { color: var(--warn); }
    .alert.err .alert-title { color: var(--err); }
    .metric-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); border: 1px solid var(--line); border-radius: 4px; overflow: hidden; background: var(--bg-1); }
    .metric { padding: 14px 16px; border-right: 1px solid var(--line); }
    .metric:last-child { border-right: none; }
    .m-lbl { font-family: "JetBrains Mono", monospace; font-size: 10px; letter-spacing: .1em; text-transform: uppercase; color: var(--ink-4); margin-bottom: 4px; }
    .m-val { font-family: "JetBrains Mono", monospace; font-size: 22px; font-weight: 500; color: var(--ink); }
    .unit { font-size: 12px; color: var(--ink-4); margin-left: 4px; }
    .progress { height: 4px; background: var(--bg-3); border-radius: 2px; overflow: hidden; }
    .progress > .bar { height: 100%; background: var(--accent); transition: width .2s; }
    .page-nav { display: flex; gap: 12px; margin-top: 32px; padding-top: 24px; border-top: 1px solid var(--line); }
    .page-nav .spacer { flex: 1; }
    .div-dotted { height: 1px; background-image: linear-gradient(to right, var(--line-2) 50%, transparent 50%); background-size: 6px 1px; margin: 16px 0; }
    .evlog { font-family: "JetBrains Mono", monospace; font-size: 11px; background: var(--bg); border: 1px solid var(--line); border-radius: 3px; max-height: 280px; overflow-y: auto; }
    .evlog .row { display: grid; grid-template-columns: 80px 60px 1fr; gap: 12px; padding: 5px 12px; border-bottom: 1px solid var(--line); color: var(--ink-2); }
    .evlog .t { color: var(--ink-4); }
    .evlog .lvl.info, .evlog .lvl.progress { color: var(--accent); }
    .evlog .lvl.ok, .evlog .lvl.job_succeeded { color: var(--ok); }
    .evlog .lvl.warn { color: var(--warn); }
    .evlog .lvl.err, .evlog .lvl.job_failed { color: var(--err); }
    .file-row { display: flex; align-items: center; gap: 12px; padding: 10px 12px; border-bottom: 1px solid var(--line); font-family: "JetBrains Mono", monospace; font-size: 11px; color: var(--ink-2); }
    .file-row:last-child { border-bottom: none; }
    .file-row .fname { color: var(--ink); flex: 1; }
    .file-row .fsize { color: var(--ink-4); }
    .viz { background: var(--bg); border: 1px solid var(--line); border-radius: 3px; position: relative; overflow: hidden; }
    .waveform { display: flex; align-items: center; gap: 2px; height: 72px; padding: 8px; }
    .waveform span { flex: 1; min-width: 2px; background: var(--accent); opacity: .82; }
    .spectrogram { display: grid; grid-template-columns: repeat(48, 1fr); grid-template-rows: repeat(8, 1fr); height: 92px; gap: 1px; padding: 6px; }
    .spectrogram span { background: rgba(34, 211, 238, .08); }
    .level-meter { display: flex; gap: 2px; height: 18px; align-items: end; }
    .level-meter .seg { width: 4px; background: var(--bg-3); border-radius: 1px; }
    .status-row { display: flex; align-items: center; gap: 12px; padding: 10px 14px; background: var(--bg-2); border: 1px solid var(--line); border-radius: 3px; font-family: "JetBrains Mono", monospace; font-size: 12px; }
    .empty { font-family: "JetBrains Mono", monospace; font-size: 11px; color: var(--ink-5); }
    @media (max-width: 1080px) {
      .app { grid-template-columns: 220px 1fr; }
      .aside { display: none; }
    }
  </style>
</head>
<body>
  <div class="app">
    <header class="topbar">
      <div class="logo"><span class="logo-dot"></span><span>WAKEWORD&nbsp;STUDIO</span><span id="app-version" style="color: var(--ink-5)">v...</span></div>
      <div class="crumbs"><span>runs</span><span class="sep">/</span><span id="crumb-run">new-run</span><span class="sep">/</span><span id="crumb-stage" style="color: var(--accent)">Wakeword setup</span></div>
      <div class="topbar-right">
        <span><span class="meta-k">host</span> <span id="app-host" class="meta-v">local</span></span>
        <span><span class="meta-k">mode</span> <span class="meta-v">local</span></span>
        <button id="theme-toggle" class="theme-toggle">theme: dark</button>
      </div>
    </header>

    <nav class="rail" aria-label="Pipeline">
      <div class="rail-title">Pipeline</div>
      <div id="stage-list"></div>
      <div class="rail-divider"></div>
      <div class="rail-section">Workspace</div>
      <div class="rail-link active">⌂ All runs</div>
      <div class="rail-link">□ MCU handoff</div>
      <div class="rail-link">⚙ Settings</div>
    </nav>

    <main class="main">
      <div class="main-inner">
        <div class="page-header">
          <div class="page-eyebrow" id="page-eyebrow">Step 01 / 09</div>
          <h1 class="page-title" id="page-title">Define the wakeword</h1>
          <div class="page-sub" id="page-sub">Pick the phrase your device will listen for. Approve the required near-miss words before seed generation.</div>
        </div>

        <section class="card">
          <div class="card-header">
            <div class="card-title">Wakeword</div>
            <span id="run-badge" class="badge idle">not started</span>
          </div>
          <div class="card-body">
            <div class="field">
              <label class="label" for="wake-word">Phrase</label>
              <input class="input lg" id="wake-word" placeholder="e.g. hey jarvis" />
              <div class="help">Use at least two spoken characters. A carrier word like hey, ok, or hi usually helps.</div>
            </div>
            <div class="field-row">
              <div class="field" style="flex: 1">
                <label class="label" for="language">Language</label>
                <select class="select" id="language">
                  <option value="ko">Korean</option>
                  <option value="en">English</option>
                </select>
              </div>
              <div class="field" style="flex: 1">
                <label class="label" for="hint">Pronunciation hint</label>
                <input class="input" id="hint" placeholder="hey JAR-vis" />
              </div>
            </div>
            <div class="div-dotted"></div>
            <div id="message" class="alert info" role="status"><div class="alert-title">Ready</div>Create a run to generate slug, model name, and near-miss candidates.</div>
          </div>
        </section>

        <section id="near-card" class="card" hidden>
          <div class="card-header">
            <div class="card-title">Near-miss review</div>
            <span id="near-count" class="badge warn">0 approved</span>
          </div>
          <div class="card-body">
            <div class="alert info">
              <div class="alert-title">Why this matters</div>
              These are words that sound close to the wakeword. Training against them reduces false wakes.
            </div>
            <div class="div-dotted"></div>
            <div id="near-miss" class="word-list"></div>
          </div>
        </section>

        <section id="seed-panel" class="card" hidden></section>

        <div class="page-nav">
          <button id="create-run" class="btn primary">New wakeword run</button>
          <button id="save-near-miss" class="btn" disabled>Save near-miss</button>
          <div class="spacer"></div>
          <span id="action-reason" class="mono" style="font-size: 11px; color: var(--ink-4); align-self: center">Create a run first.</span>
          <button id="generate-seed" class="btn primary" disabled>Generate seed sample</button>
          <button id="approve-seed" class="btn primary" disabled>Approve seed</button>
          <button id="reject-seed" class="btn danger" disabled>Sounds wrong</button>
        </div>
      </div>
    </main>

    <aside class="aside">
      <section class="aside-section">
        <div class="aside-title">Run context</div>
        <div class="kv-grid" id="run-summary">
          <div class="k">word</div><div class="v">-</div>
          <div class="k">stage</div><div class="v">not started</div>
        </div>
      </section>
      <section class="aside-section" style="flex: 1; min-height: 220px">
        <div class="aside-title">Event timeline <span id="event-count" class="mono">0 events</span></div>
        <div id="events-empty" class="empty">// no events yet</div>
        <div id="events" class="evlog" hidden></div>
      </section>
      <section class="aside-section">
        <div class="aside-title">Artifacts <span id="artifact-count" class="mono">0</span></div>
        <div id="artifact-list" class="empty">// none yet</div>
      </section>
    </aside>
  </div>

  <script>
    const stages = [
      ["setup", "Wakeword setup"],
      ["nearmiss", "Near-miss review"],
      ["seed", "Seed sample"],
      ["plan", "Data plan"],
      ["datagen", "Data generation"],
      ["training", "Training"],
      ["eval", "Evaluation"],
      ["export", "Export"],
      ["mictest", "Host mic test"],
    ];
    const backendStageIndex = {
      draft: 0,
      suitability_checked: 0,
      near_miss_review: 1,
      seed_generating: 2,
      seed_review: 2,
      data_plan_review: 3,
      data_generating: 4,
      qc_review: 4,
      feature_preparing: 5,
      training: 5,
      evaluating: 6,
      exported: 7,
      host_mic_testing: 8,
    };
    let currentRun = null;
    let eventsSource = null;
    let eventRows = [];
    let minNearMissRequired = null;

    const $ = (id) => document.getElementById(id);
    const escapeHtml = (value) => String(value ?? "").replace(/[&<>"']/g, (ch) => ({
      "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[ch]));
    const api = async (url, options = {}) => {
      const response = await fetch(url, {
        ...options,
        headers: { "content-type": "application/json", ...(options.headers || {}) },
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(data.detail || response.statusText);
      return data;
    };
    const requiredNearMiss = () => Number.isFinite(minNearMissRequired) ? minNearMissRequired : null;
    const requiredNearMissLabel = () => requiredNearMiss() ?? "...";

    function setMessage(title, text, kind = "info") {
      $("message").className = `alert ${kind}`;
      $("message").innerHTML = `<div class="alert-title">${escapeHtml(title)}</div>${escapeHtml(text)}`;
    }

    function checkedNearMiss() {
      return [...document.querySelectorAll("#near-miss input:checked")].map((el) => el.value);
    }

    function renderStages(activeIndex) {
      $("stage-list").innerHTML = stages.map(([id, label], index) => {
        const status = index < activeIndex ? "done" : index === activeIndex ? "current" : "locked";
        const num = status === "done" ? "✓" : status === "locked" ? "•" : String(index + 1);
        return `<div class="stage ${status}" data-stage="${id}">
          <span class="stage-num">${num}</span>
          <span class="stage-label">${escapeHtml(label)}</span>
        </div>`;
      }).join("");
      $("crumb-stage").textContent = stages[activeIndex]?.[1] || "Wakeword setup";
      $("page-eyebrow").textContent = `Step ${String(activeIndex + 1).padStart(2, "0")} / 09`;
    }

    function renderRunSummary(run) {
      if (!run) return;
      $("crumb-run").textContent = run.target_slug;
      $("run-summary").innerHTML = `
        <div class="k">word</div><div class="v" style="color: var(--ink)">"${escapeHtml(run.wake_word)}"</div>
        <div class="k">lang</div><div class="v">${escapeHtml(run.language)}</div>
        <div class="k">slug</div><div class="v">${escapeHtml(run.target_slug)}</div>
        <div class="k">model</div><div class="v">${escapeHtml(run.model_name)}</div>
        <div class="k">stage</div><div class="v" style="color: var(--accent)">${escapeHtml(run.stage)}</div>
        <div class="k">updated</div><div class="v">${escapeHtml(run.updated_at)}</div>
      `;
      $("run-badge").className = "badge run";
      $("run-badge").textContent = "in progress";
    }

    function updateNearMissCount() {
      const count = checkedNearMiss().length;
      const required = requiredNearMiss();
      const ready = currentRun && required !== null && count >= required && ["near_miss_review", "seed_review"].includes(currentRun.stage);
      $("near-count").className = `badge ${ready ? "ok" : "warn"}`;
      $("near-count").textContent = `${count} / ${requiredNearMissLabel()} approved`;
      $("save-near-miss").disabled = !currentRun || count === 0;
      $("generate-seed").disabled = !ready;
      $("action-reason").textContent = !currentRun
        ? "Create a run first."
        : required === null
          ? "Loading app policy."
        : ready
          ? "Ready to generate exactly one seed sample."
          : `${Math.max(0, required - count)} more near-miss words required.`;
    }

    function renderNearMiss(run) {
      $("near-card").hidden = false;
      const approved = new Set(run.near_miss_approved || []);
      $("near-miss").innerHTML = (run.near_miss_candidates || []).map((item) => `
        <label class="word-chip">
          <input type="checkbox" value="${escapeHtml(item)}" ${approved.has(item) ? "checked" : ""} />
          <span>${escapeHtml(item)}</span>
        </label>
      `).join("");
      document.querySelectorAll("#near-miss input").forEach((input) => {
        input.addEventListener("change", updateNearMissCount);
      });
      updateNearMissCount();
    }

    function renderArtifacts(run) {
      const entries = Object.entries(run?.artifact_urls || {});
      $("artifact-count").textContent = entries.length;
      $("artifact-list").className = entries.length ? "" : "empty";
      $("artifact-list").innerHTML = entries.length ? entries.map(([key, url]) => `
        <a class="file-row" href="${url}" target="_blank" rel="noreferrer">
          <span>□</span><span class="fname">${escapeHtml(key)}</span><span class="fsize">artifact</span>
        </a>
      `).join("") : "// none yet";
    }

    function renderSeedPanel(run) {
      const seedUrl = run.artifact_urls?.seed_wav;
      const qc = run.seed_qc_summary || {};
      const qcPassed = qc.keep === true;
      const qcKnown = qc.keep === true || qc.keep === false;
      const qcLabel = qcKnown ? (qcPassed ? "PASS" : "CHECK") : "UNKNOWN";
      const qcClass = qcKnown ? (qcPassed ? "ok" : "warn") : "warn";
      const sampleRate = Number.isFinite(qc.sample_rate) ? Math.round(qc.sample_rate / 1000) : null;
      const duration = Number.isFinite(qc.duration_s) ? qc.duration_s.toFixed(2) : null;
      $("seed-panel").hidden = !(seedUrl || run.stage === "seed_generating" || run.stage === "seed_review");
      $("approve-seed").disabled = !(run.stage === "seed_review" && seedUrl);
      $("reject-seed").disabled = run.stage !== "seed_review";
      if (run.stage === "seed_generating") {
        $("seed-panel").innerHTML = `
          <div class="card-header"><div class="card-title">Seed sample</div><span class="badge run">generating</span></div>
          <div class="card-body">
            <div class="mono" style="font-size: 12px; color: var(--ink-3); margin-bottom: 12px">qwen · synthesizing "${escapeHtml(run.wake_word)}"</div>
            <div class="progress"><div class="bar" style="width: 66%"></div></div>
            <div class="help">Loading model, generating audio, then running QC.</div>
          </div>`;
      } else if (seedUrl) {
        $("seed-panel").innerHTML = `
          <div class="card-header"><div class="card-title">Seed sample</div><span class="badge warn">awaiting approval</span></div>
          <div class="card-body">
            <audio controls src="${seedUrl}?t=${Date.now()}" style="width: 100%; margin-bottom: 12px"></audio>
            <div class="label">Waveform preview</div>
            <div class="viz waveform" id="waveform"></div>
            <div class="help">Visual placeholder until browser-side audio analysis is added.</div>
            <div class="div-dotted"></div>
            <div class="label">Mel spectrogram preview</div>
            <div class="viz spectrogram" id="spectrogram"></div>
            <div class="help">QC numbers below come from the generated manifest.</div>
            <div class="div-dotted"></div>
            <div class="metric-strip">
              <div class="metric"><div class="m-lbl">QC</div><div class="m-val" style="color: var(--${qcClass})">${qcLabel}</div></div>
              <div class="metric"><div class="m-lbl">Sample rate</div><div class="m-val">${sampleRate ?? "-"}<span class="unit">kHz</span></div></div>
              <div class="metric"><div class="m-lbl">Duration</div><div class="m-val">${duration ?? "-"}<span class="unit">s</span></div></div>
            </div>
          </div>`;
        renderAudioViz();
      }
    }

    function renderAudioViz() {
      const wave = $("waveform");
      const spec = $("spectrogram");
      if (wave) {
        wave.innerHTML = Array.from({ length: 72 }, (_, index) => {
          const h = 12 + Math.round(Math.abs(Math.sin(index * .35) * Math.cos(index * .09)) * 54);
          return `<span style="height:${h}px"></span>`;
        }).join("");
      }
      if (spec) {
        spec.innerHTML = Array.from({ length: 384 }, (_, index) => {
          const value = Math.abs(Math.sin(index * .17) * Math.cos(index * .031));
          const alpha = (0.08 + value * 0.86).toFixed(2);
          return `<span style="background: rgba(34, 211, 238, ${alpha})"></span>`;
        }).join("");
      }
    }

    function renderEvents() {
      $("event-count").textContent = `${eventRows.length} events`;
      $("events-empty").hidden = eventRows.length > 0;
      $("events").hidden = eventRows.length === 0;
      $("events").innerHTML = eventRows.slice(-20).reverse().map((row) => {
        const time = row.time ? new Date(row.time).toLocaleTimeString("en-GB", { hour12: false }) : "--:--:--";
        const level = row.event || "info";
        return `<div class="row"><span class="t">${time}</span><span class="lvl ${level}">${escapeHtml(level).toUpperCase()}</span><span>${escapeHtml(row.message)}</span></div>`;
      }).join("");
    }

    function renderRun(run) {
      currentRun = run;
      const activeIndex = backendStageIndex[run.stage] ?? 0;
      renderStages(activeIndex);
      renderRunSummary(run);
      renderNearMiss(run);
      renderSeedPanel(run);
      renderArtifacts(run);
      $("page-title").textContent = activeIndex === 1 ? "Approve near-miss words" : activeIndex === 2 ? "Generate one seed sample" : activeIndex === 3 ? "Seed approved. Data plan is next." : "Define the wakeword";
      $("page-sub").textContent = activeIndex === 1
        ? `Approve at least ${requiredNearMissLabel()} words that sound like "${run.wake_word}".`
        : activeIndex === 2
          ? "Produce a single sample first. If it sounds wrong, fix the phrase before generating thousands more."
          : `Pick the phrase your device will listen for. Approve at least ${requiredNearMissLabel()} near-miss words before seed generation.`;
      if (run.last_error) setMessage("Last error", run.last_error, "err");
    }

    async function saveNearMiss(showMessage = true) {
      if (!currentRun) return null;
      const approved = checkedNearMiss();
      const run = await api(`/api/runs/${currentRun.run_id}/near-miss`, {
        method: "POST",
        body: JSON.stringify({ approved }),
      });
      const required = requiredNearMiss();
      if (showMessage) setMessage("Saved", `${approved.length} near-miss words approved.`, required !== null && approved.length >= required ? "info" : "warn");
      renderRun(run);
      return run;
    }

    async function loadAppConfig() {
      $("app-host").textContent = window.location.host || "local";
      try {
        const config = await api("/api/config");
        minNearMissRequired = Number(config.min_approved_near_miss);
        $("app-version").textContent = `v${config.app_version || "dev"}`;
        if (currentRun) renderRun(currentRun);
      } catch (error) {
        setMessage("Cannot load app config", error.message, "err");
      }
    }

    function watchEvents(runId) {
      if (eventsSource) eventsSource.close();
      eventRows = [];
      renderEvents();
      eventsSource = new EventSource(`/api/runs/${runId}/events`);
      eventsSource.onmessage = (event) => {
        eventRows.push(JSON.parse(event.data));
        renderEvents();
        refreshRun();
      };
    }

    async function refreshRun() {
      if (!currentRun) return;
      renderRun(await api(`/api/runs/${currentRun.run_id}`));
    }

    $("create-run").onclick = async () => {
      try {
        const run = await api("/api/runs", {
          method: "POST",
          body: JSON.stringify({
            wake_word: $("wake-word").value,
            language: $("language").value || "ko",
            pronunciation_hint: $("hint").value,
          }),
        });
        setMessage("Run created", `Review and approve at least ${requiredNearMissLabel()} near-miss words.`, "info");
        renderRun(run);
        watchEvents(run.run_id);
      } catch (error) {
        setMessage("Cannot create run", error.message, "err");
      }
    };
    $("save-near-miss").onclick = () => saveNearMiss(true).catch((error) => setMessage("Cannot save", error.message, "err"));
    $("generate-seed").onclick = async () => {
      if (!currentRun) return;
      try {
        await saveNearMiss(false);
        const job = await api(`/api/runs/${currentRun.run_id}/seed`, { method: "POST", body: "{}" });
        setMessage("Seed generation started", `Job ${job.job_id} is running.`, "info");
        watchEvents(currentRun.run_id);
        await refreshRun();
      } catch (error) {
        setMessage("Cannot generate seed", error.message, "err");
      }
    };
    $("approve-seed").onclick = async () => {
      if (!currentRun) return;
      try {
        renderRun(await api(`/api/runs/${currentRun.run_id}/seed/approve`, { method: "POST", body: "{}" }));
        setMessage("Seed approved", "Data plan is next.", "info");
      } catch (error) {
        setMessage("Cannot approve", error.message, "err");
      }
    };
    $("reject-seed").onclick = async () => {
      if (!currentRun) return;
      try {
        renderRun(await api(`/api/runs/${currentRun.run_id}/seed/reject`, {
          method: "POST",
          body: JSON.stringify({ reason: "User rejected seed sample." }),
        }));
        setMessage("Seed rejected", "Regenerate or edit pronunciation hint.", "warn");
      } catch (error) {
        setMessage("Cannot reject", error.message, "err");
      }
    };
    $("theme-toggle").onclick = () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      $("theme-toggle").textContent = `theme: ${next}`;
    };

    renderStages(0);
    renderEvents();
    loadAppConfig();
  </script>
</body>
</html>
"""
