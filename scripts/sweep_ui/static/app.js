/* Sweep browser front end.
 *
 * Hand-written SVG rather than a charting library, and no CDN reference: a cluster node may have
 * no egress, and a build step for four chart types would be worse than the ~400 lines here. The
 * charts deliberately echo the plotnine figures they replace -- Set1 series colours, a dashed
 * pretrained anchor, a dashed warmup marker -- so a screenshot from this page and a PDF from
 * plots/ are recognisably the same measurement.
 */
'use strict';

const SET1 = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf'];
const $ = (s) => document.querySelector(s);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElementNS(
    tag === 'svg' || SVG_TAGS.has(tag) ? 'http://www.w3.org/2000/svg' : 'http://www.w3.org/1999/xhtml',
    tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') n.setAttribute('class', v);
    else if (k === 'text') n.textContent = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  // anything that is not already a Node becomes text: passing a NUMBER child (an axis tick) threw
  // "parameter 1 is not of type 'Node'" and took the whole chart loop down with it, leaving a
  // legend and no panels
  for (const k of kids) {
    if (k == null) continue;
    n.appendChild(k.nodeType ? k : document.createTextNode(String(k)));
  }
  return n;
};
const SVG_TAGS = new Set(['g', 'rect', 'line', 'path', 'text', 'polyline', 'circle', 'clipPath']);

const TABS = ['grid', 'curves', 'gens', 'table'];
const state = {
  runs: [], orphans: [], collapseRatio: 1.5, runningWindow: 7200,
  selected: new Set(), traces: new Map(),
  sort: { key: 'off_target', dir: -1 },
  /* which metric each grid is showing, keyed by its group title. Per grid and not global because
   * the available keys are the organism's: a casing grid has no `target_frac` to switch to. */
  gridMetric: new Map(),
  /* the grid's side panel: {group, run}, null when closed */
  panel: null, panelSplit: '',
  /* All-runs rows whose attached mask/GRPO/restricted runs are unfolded, and each of those runs'
   * own metric pick */
  expanded: new Set(), attachedMetric: new Map(),
  /* the tab lives in the URL hash so a reload keeps the view and a link carries it */
  tab: TABS.includes(location.hash.slice(1)) ? location.hash.slice(1) : 'grid',
};

/* ---------- formatting ---------- */
const fmtLr = (lr) => {
  if (lr == null) return '—';
  const e = Math.floor(Math.log10(lr));
  const m = +(lr / 10 ** e).toFixed(3);
  const sup = String(e).replace(/-/g, '⁻').replace(/\d/g, (d) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+d]);
  return `${m}×10${sup}`;
};
const fmt = (v, d = 3) => (v == null ? '—' : Number(v).toFixed(d));
const pct = (v) => (v == null ? '—' : (100 * v).toFixed(0) + '%');
/* metrics are mostly fractions, but the dropdown also reaches counts (`n`) and losses, so a
   fixed 2dp would print `64.00` for a sample size */
const fmtMetric = (v) => (v == null ? '—' : Math.abs(v) >= 10 ? String(Math.round(v))
  : Number(v).toFixed(2));
const fmtAge = (s) => {
  if (s == null) return '—';
  if (s < 90) return `${Math.round(s)}s`;
  if (s < 5400) return `${Math.round(s / 60)}m`;
  if (s < 172800) return `${Math.round(s / 3600)}h`;
  return `${Math.round(s / 86400)}d`;
};
/* `pending` splits into running/idle on the age of the newest artifact, which is a hint and not a
   fact -- see RUNNING_WINDOW_S in sweep_ui.py. Nothing on disk knows whether the job is alive. */
const statusOf = (r) => (r.status !== 'pending' ? r.status
  : (r.idle_s != null && r.idle_s > state.runningWindow ? 'idle' : 'running'));
const statusText = (r) => {
  if (r.status !== 'pending') return r.status;
  const bits = [statusOf(r)];
  if (r.last_step != null) bits.push(`step ${r.last_step}`);
  if (r.idle_s != null) bits.push(`${fmtAge(r.idle_s)} ago`);
  return bits.join(' · ');
};

/* white -> red ramp, matching the plotnine rate scale; grey when the run collapsed.
   `domain` is [0,1] for a fraction -- so two grids' colours mean the same thing -- and the
   observed range for anything else, which is why the header prints it. */
const tileFill = (v, collapsed, domain = [0, 1]) => {
  if (collapsed) return 'var(--grey)';
  if (v == null) return 'transparent';
  const [d0, d1] = domain;
  const t = Math.max(0, Math.min(1, (v - d0) / (d1 - d0 || 1)));
  return `rgb(${253 - 25 * t}, ${234 - 208 * t}, ${234 - 208 * t})`;
};

/* ---------- data ---------- */
async function load(refresh = false) {
  $('#status').textContent = 'loading…';
  const r = await fetch('/api/runs' + (refresh ? '?refresh=1' : ''));
  const d = await r.json();
  state.runs = d.runs;
  state.orphans = d.orphans || [];
  state.collapseRatio = d.collapse_ratio;
  state.runningWindow = d.running_window_s ?? state.runningWindow;
  $('#ratio-note').textContent = d.collapse_ratio;
  $('#status').textContent = `${d.runs.length} finetunes · ${d.root}`;
  fillFilters();
  render();
}

function fillFilters() {
  const uniq = (f) => [...new Set(state.runs.map(f).filter(Boolean))].sort();
  for (const [sel, f] of [['#f-model', (r) => r.model_short], ['#f-task', (r) => r.task],
                          ['#f-method', (r) => r.method]]) {
    const node = $(sel), keep = node.value;
    node.innerHTML = '';
    node.appendChild(el('option', { value: '' }, 'all'));
    for (const v of uniq(f)) node.appendChild(el('option', { value: v }, v));
    node.value = keep;
  }
}

const filtered = () => state.runs.filter((r) => {
  const q = $('#f-search').value.trim().toLowerCase();
  return (!$('#f-model').value || r.model_short === $('#f-model').value)
    && (!$('#f-task').value || r.task === $('#f-task').value)
    && (!$('#f-method').value || r.method === $('#f-method').value)
    && (!q || r.name.toLowerCase().includes(q))
    && (!$('#f-healthy').checked || !r.collapsed);
});

/* ---------- heatmap grids, one per (task, model) ---------- */

/* The grouping every view shares: one block per (task, model). Two organisms must never share an
 * axis or a colour scale, and the model is part of the identity because a 1B and an 8B cell at the
 * same lr are not the same experiment. */
function groupRuns(runs) {
  const groups = new Map();
  for (const r of runs) {
    const k = `${r.task} · ${r.model_short}`;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(r);
  }
  return groups;
}

/* Every (metric, split) pair at least one run in this group reported, grouped by split with
 * off_target first and the group's headline at the top of it.
 *
 * Keyed on the split as well as the metric, and not filtered to off_target, because two of the
 * most informative evals have no off-target notion at all: `mmlu` reports one split of its own
 * (the capability-retention control — a finetune that moved MMLU is not localised, whatever its
 * behaviour metric says) and `sft_loss` reports train/test. Filtering on an `off_target` key
 * silently dropped both. The optgroups are what keep ~30 options readable.
 *
 * Union rather than intersection: a metric only some cells have is still worth offering (a
 * two-phase eval that failed on one cell leaves exactly that hole) and its tiles read as blank
 * rather than as zero. */
const SPLIT_ORDER = ['off_target', 'in_dist'];
/* An option is `<eval>.<metric>@<split>`; it is shown without the `@split`, because the optgroup
   it sits in is the split. Split on the LAST `@` -- no metric name contains one today, but the
   round trip should not depend on that. */
const optId = (id, split) => `${id}@${split}`;
const parseOpt = (opt) => { const i = (opt || '').lastIndexOf('@'); return [opt.slice(0, i), opt.slice(i + 1)]; };
const optLabel = (opt) => parseOpt(opt)[0];

function metricIds(rs) {
  const bySplit = new Map();
  for (const r of rs) {
    for (const [id, per] of Object.entries(r.metrics || {})) {
      for (const [split, v] of Object.entries(per)) {
        if (v == null) continue;
        if (!bySplit.has(split)) bySplit.set(split, new Set());
        bySplit.get(split).add(id);
      }
    }
  }
  const heads = new Set([...new Set(rs.map((r) => r.headline_id))].filter(Boolean)
    .map((id) => optId(id, 'off_target')));
  const rank = (s) => (SPLIT_ORDER.indexOf(s) + 1 || SPLIT_ORDER.length + 1);
  const splits = [...bySplit.keys()].sort((a, b) => rank(a) - rank(b) || a.localeCompare(b));
  const groups = splits.map((split) => [split, [...bySplit.get(split)].sort(
    /* the headline first inside its own split, so the default is also the top entry */
    (a, b) => (heads.has(optId(b, split)) - heads.has(optId(a, split))) || a.localeCompare(b),
  ).map((id) => optId(id, split))]);
  return { groups, ids: groups.flatMap(([, o]) => o), heads };
}

/* [0,1] for a fraction, so the colour of a tile means the same thing in every grid; the observed
 * range for a count or a loss, where a fixed [0,1] would paint every tile the same red. */
function domainOf(values) {
  if (!values.length) return [0, 1];
  const lo = Math.min(...values), hi = Math.max(...values);
  return (lo >= 0 && hi <= 1) ? [0, 1] : [Math.min(0, lo), hi || 1];
}

function renderGrids(runs) {
  const host = $('#grids');
  host.innerHTML = '';
  /* Baselines are excluded: nothing trained, so "method x learning rate" has no meaning for them
   * and a tile would read as a finetune of whatever dataset their config names as a placeholder. */
  const bases = runs.filter((r) => r.baseline);
  runs = runs.filter((r) => !r.baseline);
  const groups = groupRuns(runs);
  if (bases.length) {
    host.appendChild(el('p', { class: 'note' },
      `${bases.length} pretrained-anchor run(s) hidden here (epochs: 0 — nothing trained, so there `
      + `is no learning rate to place them at). They are in the All runs tab, tagged baseline.`));
  }
  if (!groups.size) { host.appendChild(el('p', { class: 'note' }, 'nothing matches the filters')); return; }

  for (const [title, rs] of [...groups].sort()) {
    const lrs = [...new Set(rs.map((r) => r.lr))].sort((a, b) => a - b);
    const methods = [...new Set(rs.map((r) => r.method))]
      .sort((a, b) => (a === 'Full SFT') - (b === 'Full SFT')
        || (parseInt(a.replace(/\D/g, '')) || 0) - (parseInt(b.replace(/\D/g, '')) || 0));
    const { groups: opts, ids, heads } = metricIds(rs);
    /* the remembered pick, unless the filters removed the metric it named */
    let opt = state.gridMetric.get(title);
    if (!ids.includes(opt)) opt = ids.find((id) => heads.has(id)) || ids[0];
    const [mid, mSplit] = parseOpt(opt || '@');
    const valueOf = (r) => (opt ? r.metrics?.[mid]?.[mSplit] : r.off_target) ?? null;
    const domain = domainOf(rs.filter((r) => !r.collapsed).map(valueOf)
      .filter((v) => v != null));

    const box = el('div', { class: 'grid' });
    const label = rs.find((r) => r.metric_label)?.metric_label;
    const sel = el('select', {
      onchange: (ev) => { state.gridMetric.set(title, ev.target.value); render(); },
    });
    for (const [split, ids_] of opts) {
      const g = el('optgroup', { label: split });
      for (const id of ids_) {
        g.appendChild(el('option', { value: id },
          optLabel(id) + (heads.has(id) ? `  (headline${label ? ': ' + label : ''})` : '')));
      }
      sel.appendChild(g);
    }
    sel.value = opt || '';
    const head = el('div', { class: 'gridhead' },
      el('h3', {}, title, el('span', {}, opt ? ` — ${mSplit} split` : '')));
    if (ids.length) head.appendChild(el('label', {}, 'metric ', sel));
    else head.appendChild(el('span', { class: 'muted' }, 'no metric reported yet'));
    if (domain[0] !== 0 || domain[1] !== 1) {
      head.appendChild(el('span', { class: 'muted' },
        `colour scaled to ${fmtMetric(domain[0])}–${fmtMetric(domain[1])} (not a fraction)`));
    }
    box.appendChild(head);

    const tbl = el('table', { class: 'tiles' });
    const hrow = el('tr', {}, el('th', {}, ''));
    for (const lr of lrs) hrow.appendChild(el('th', {}, fmtLr(lr)));
    tbl.appendChild(el('thead', {}, hrow));
    const body = el('tbody');
    for (const m of methods) {
      const tr = el('tr', {}, el('th', {}, m));
      for (const lr of lrs) {
        /* All matches, not the first: two runs can share (task, model, method, lr) -- a cell
         * resubmitted under a second output directory, say -- and `.find()` would draw one of
         * them with nothing saying the other existed. Which one gets drawn is then decided
         * rather than left to directory order: results beat stale results beat no results. */
        const rank = (r) => (r.status === 'done' ? 0 : r.status === 'stale' ? 1 : 2);
        const cell = rs.filter((r) => r.method === m && r.lr === lr)
          .sort((a, b) => rank(a) - rank(b) || a.name.localeCompare(b.name));
        if (!cell.length) { tr.appendChild(el('td', { class: 'empty' })); continue; }
        const run = cell[0];
        const v = valueOf(run);
        const tip = [run.name,
          run.status === 'pending'
            ? `NO RESULTS YET (${statusText(run)})`
            : `${mid} — ${mSplit} ${fmtMetric(v)}`
              + Object.entries(run.metrics?.[mid] || {}).filter(([s]) => s !== mSplit)
                .map(([s, x]) => `, ${s} ${fmtMetric(x)}`).join(''),
          `headline ${run.headline_id || '—'} off-target ${fmt(run.off_target)}`,
          `test loss ${fmt(run.test_loss)} (anchor ${fmt(run.anchor_test_loss)}, `
            + `ratio ${fmt(run.loss_ratio, 2)})`,
          run.collapsed ? 'COLLAPSED' : null,
          run.stale ? 'STALE: config newer than evals' : null,
          cell.length > 1 ? `${cell.length} runs in this cell: ${cell.map((r) => r.name).join(', ')}`
            : null,
        ].filter(Boolean).join('\n');
        const td = el('td', {
          title: tip,
          /* opens the side panel rather than leaving the grid: the comparison a tile invites is
           * with the tiles beside it, and switching tabs threw that context away */
          onclick: () => openPanel(title, run.name),
          /* A collapsed run keeps its number, on grey. The grey is the warning; blanking the
           * value would hide `undetermined_frac`, which is the metric you switch to precisely
           * BECAUSE the cell collapsed -- it is how you tell a broken model from a drifted one. */
        }, run.status === 'pending' ? '⋯' : fmtMetric(v));
        td.style.background = tileFill(run.status === 'pending' ? null : v, run.collapsed, domain);
        if (run.status === 'pending') td.classList.add('pending');
        if (run.collapsed) td.classList.add('collapsed');
        if (state.selected.has(run.name)) td.classList.add('sel');
        if (run.stale) td.appendChild(el('span', { class: 'tag' }, 'stale'));
        if (cell.length > 1) td.appendChild(el('span', { class: 'tag' }, `×${cell.length}`));
        tr.appendChild(td);
      }
      body.appendChild(tr);
    }
    tbl.appendChild(body);
    box.appendChild(tbl);
    host.appendChild(box);
  }
  renderPanel();
}

/* ---------- the grid's side panel ---------- */
function openPanel(group, name) {
  state.panel = { group, run: name };
  toggle(name, true);            // re-renders; the panel is drawn from renderGrids
}

function closePanel() {
  state.panel = null;
  state.selected.clear();
  render();
}

/* One run, in place: its trajectory against every run in the same grid, and its own generations.
 *
 * The two halves answer different questions and are deliberately scoped differently. A curve is
 * only meaningful against the cells it should be read with, so the chart keeps the whole group and
 * highlights this run; text is only meaningful as this run's own output, so the generations are
 * filtered to it. Both live beside the grid rather than on another tab, because the click that
 * opened them was a click on a comparison.
 */
async function renderPanel() {
  const host = $('#panel');
  if (!host) return;
  const sel = state.panel;
  host.hidden = !sel;
  if (!sel) { host.innerHTML = ''; return; }
  const run = state.runs.find((r) => r.name === sel.run);
  if (!run) { state.panel = null; host.hidden = true; host.innerHTML = ''; return; }
  const group = groupRuns(filtered().filter((r) => !r.baseline)).get(sel.group) || [run];

  host.innerHTML = '';
  host.appendChild(el('div', { class: 'panelhead' },
    el('h3', {}, run.name),
    el('button', { class: 'close', title: 'close (Esc)', onclick: closePanel }, '×')));
  const facts = [`${run.method} · ${fmtLr(run.lr)} · ${run.model_short}`,
    `${statusText(run)}${run.steps ? ` · ${run.steps} steps` : ''}`,
    run.headline_id ? `${run.headline_id}: off-target ${fmtMetric(run.off_target)}, `
      + `in-dist ${fmtMetric(run.in_dist)}` : null,
    run.loss_ratio != null ? `test loss ${fmt(run.test_loss, 2)} / anchor `
      + `${fmt(run.anchor_test_loss, 2)} = ${fmt(run.loss_ratio, 2)}`
      + (run.collapsed ? ' — COLLAPSED' : '') : null,
  ].filter(Boolean);
  const meta = el('div', { class: 'panelmeta' });
  for (const f of facts) meta.appendChild(el('div', {}, f));
  if (run.wandb_url) {
    meta.appendChild(el('div', {}, el('a', { href: run.wandb_url, target: '_blank',
      rel: 'noopener' }, run.wandb_exact ? 'open in wandb ↗' : 'find in wandb ↗')));
  }
  host.appendChild(meta);

  /* Curves left, text right: they are read against each other -- "the rate jumped at step 200,
   * what was it saying" -- and stacking them put a scroll between the two halves of one question. */
  const left = el('div', { class: 'panelcol' });
  const right = el('div', { class: 'panelcol' });
  host.appendChild(el('div', { class: 'panelbody' }, left, right));

  /* Trajectories: the whole grid, this run thick. */
  left.appendChild(el('h4', { class: 'panelsec' }, 'Trajectories',
    el('span', {}, ` — all ${group.length} run(s) in ${sel.group}, this one highlighted`)));
  const withHistory = group.filter((r) => r.n_eval_points > 1);
  if (!withHistory.length) {
    left.appendChild(el('p', { class: 'note' }, 'no run in this grid has a trajectory yet'));
  } else {
    left.appendChild(legendFor(withHistory));
    const slot = el('div', {});
    left.appendChild(slot);
    await ensureTraces(withHistory);
    /* the panel may have been closed or moved on while the fetches were in flight */
    if (state.panel !== sel || !slot.isConnected) return;
    slot.appendChild(curvesRow(withHistory, { highlight: new Set([run.name]), compact: true }));
  }

  /* Generations: this run only. */
  right.appendChild(el('h4', { class: 'panelsec' }, 'Generations',
    el('span', {}, ' — this run only, last eval point first')));
  if (!run.has_generations) {
    right.appendChild(el('p', { class: 'note' }, 'this run wrote no generations '
      + '(its evals are forward-only, or it has not reached an eval point)'));
    return;
  }
  const bar = el('div', { class: 'bar' });
  const gsplit = el('select', { onchange: () => renderPanel() });
  bar.appendChild(el('label', {}, 'Split ', gsplit));
  const count = el('span', { class: 'muted' });
  bar.appendChild(count);
  right.appendChild(bar);
  const list = el('div', { class: 'genlist' });
  right.appendChild(list);
  const q = new URLSearchParams({ split: state.panelSplit || '', limit: 12 });
  const d = await (await fetch(`/api/run/${encodeURIComponent(run.name)}/generations?${q}`)).json();
  if (state.panel !== sel || !list.isConnected) return;
  gsplit.appendChild(el('option', { value: '' }, 'all splits'));
  for (const s of d.splits) gsplit.appendChild(el('option', { value: s }, s));
  gsplit.value = d.splits.includes(state.panelSplit) ? state.panelSplit : '';
  gsplit.onchange = () => { state.panelSplit = gsplit.value; renderPanel(); };
  count.textContent = `${d.total} record(s)${d.total > d.records.length ? `, showing ${d.records.length}` : ''}`;
  for (const rec of d.records) list.appendChild(genCard(rec));
}

/* ---------- line charts ---------- */
function chart(title, hint, seriesList, opts = {}) {
  const W = 460, H = 190, L = 44, R = 8, T = 8, B = 30;
  const pts = seriesList.flatMap((s) => s.points);
  if (!pts.length) return null;
  const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
  let y0 = opts.yRange ? opts.yRange[0] : Math.min(...ys);
  let y1 = opts.yRange ? opts.yRange[1] : Math.max(...ys);
  if (opts.anchor != null && !opts.yRange) { y0 = Math.min(y0, opts.anchor); y1 = Math.max(y1, opts.anchor); }
  if (y1 - y0 < 1e-9) { y0 -= 0.5; y1 += 0.5; }
  const pad = (y1 - y0) * 0.06; y0 -= pad; y1 += pad;
  const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1;
  const sx = (v) => L + ((v - x0) / (x1 - x0 || 1)) * (W - L - R);
  const sy = (v) => T + (1 - (v - y0) / (y1 - y0)) * (H - T - B);

  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, preserveAspectRatio: 'xMidYMid meet' });
  const ticks = (lo, hi, n) => {
    const step = (hi - lo) / n, out = [];
    for (let i = 0; i <= n; i++) out.push(lo + i * step);
    return out;
  };
  for (const t of ticks(y0, y1, 4)) {
    svg.appendChild(el('line', { x1: L, x2: W - R, y1: sy(t), y2: sy(t), stroke: 'var(--line)', 'stroke-width': 0.5 }));
    svg.appendChild(el('text', { x: L - 5, y: sy(t) + 3, 'text-anchor': 'end', 'font-size': 9, fill: 'var(--muted)' }, t.toFixed(opts.yDigits ?? 2)));
  }
  for (const t of ticks(x0, x1, 4)) {
    svg.appendChild(el('text', { x: sx(t), y: H - 10, 'text-anchor': 'middle', 'font-size': 9, fill: 'var(--muted)' }, Math.round(t)));
  }
  if (opts.anchor != null) {
    svg.appendChild(el('line', { x1: L, x2: W - R, y1: sy(opts.anchor), y2: sy(opts.anchor), stroke: 'var(--muted)', 'stroke-width': 0.8, 'stroke-dasharray': '4 3' }));
  }
  if (opts.warmup != null && opts.warmup > x0 && opts.warmup < x1) {
    svg.appendChild(el('line', { x1: sx(opts.warmup), x2: sx(opts.warmup), y1: T, y2: H - B, stroke: 'var(--muted)', 'stroke-width': 0.8, 'stroke-dasharray': '4 3' }));
  }
  for (const s of seriesList) {
    svg.appendChild(el('polyline', {
      points: s.points.map((p) => `${sx(p[0])},${sy(Math.max(y0, Math.min(y1, p[1])))}`).join(' '),
      fill: 'none', stroke: s.color, 'stroke-width': s.width ?? 1.3,
      'stroke-opacity': s.opacity ?? 1, 'stroke-linejoin': 'round',
    }));
  }
  /* hover: nearest x over all series, crosshair + readout */
  const cross = el('line', { y1: T, y2: H - B, stroke: 'var(--ink)', 'stroke-width': 0.6, 'stroke-opacity': 0 });
  const read = el('text', { x: L + 4, y: T + 10, 'font-size': 9, fill: 'var(--ink)' });
  svg.appendChild(cross); svg.appendChild(read);
  const hit = el('rect', { x: L, y: T, width: W - L - R, height: H - T - B, fill: 'transparent' });
  hit.addEventListener('mousemove', (ev) => {
    const bb = svg.getBoundingClientRect();
    const px = ((ev.clientX - bb.left) / bb.width) * W;
    const step = x0 + ((px - L) / (W - L - R)) * (x1 - x0);
    cross.setAttribute('x1', sx(step)); cross.setAttribute('x2', sx(step));
    cross.setAttribute('stroke-opacity', 0.5);
    const near = seriesList.map((s) => {
      let best = null;
      for (const p of s.points) if (!best || Math.abs(p[0] - step) < Math.abs(best[0] - step)) best = p;
      return best ? `${s.label} ${best[1].toFixed(opts.yDigits ?? 3)}` : null;
    }).filter(Boolean);
    read.textContent = `step ${Math.round(step)} · ${near.slice(0, 4).join('  ')}`;
  });
  hit.addEventListener('mouseleave', () => { cross.setAttribute('stroke-opacity', 0); read.textContent = ''; });
  svg.appendChild(hit);

  const box = el('div', { class: 'chart' }, el('h4', {}, title));
  if (hint) box.appendChild(el('div', { class: 'hint' }, hint));
  box.appendChild(svg);
  return box;
}

/* One /api/run per run, memoised: the trajectory tab and the grid's side panel want the same
 * series, and a run's history does not change between them. */
async function ensureTraces(rs) {
  await Promise.all(rs.map(async (r) => {
    if (!state.traces.has(r.name)) {
      state.traces.set(r.name, await (await fetch(`/api/run/${encodeURIComponent(r.name)}`)).json());
    }
  }));
}

const colorFor = (rs) => {
  const lrs = [...new Set(rs.map((r) => r.lr))].sort((a, b) => a - b);
  return (lr) => SET1[lrs.indexOf(lr) % SET1.length];
};

function legendFor(rs) {
  const colorOf = colorFor(rs);
  const legend = el('div', { class: 'legend' });
  for (const lr of [...new Set(rs.map((r) => r.lr))].sort((a, b) => a - b)) {
    legend.appendChild(el('span', {}, el('i', { style: `background:${colorOf(lr)}` }), fmtLr(lr)));
  }
  return legend;
}

/* The four panels for one group of runs, coloured by learning rate. `highlight` is a Set of run
 * names to draw thick with the rest faded -- the same treatment for a table selection and for the
 * run whose tile opened the side panel, because it answers the same question either way: where
 * does THIS cell sit among the cells it should be read against. */
function curvesRow(rs, { highlight = new Set(), compact = false } = {}) {
  const colorOf = colorFor(rs);
  /* Only highlight what is actually drawn. A pending run has no series, so a highlight naming it
   * would fade every curve in the panel to 0.25 and emphasise nothing -- the panel would look
   * broken rather than empty. */
  const hl = new Set([...highlight].filter((n) => rs.some((r) => r.name === n)));
  /* Loss panels are capped at 1.5x the largest step-0 held-out loss in the group -- the loss
   * before any update, so above it a run has made the model worse than it started. A collapsed
   * cell then leaves the panel instead of flattening every healthy curve into the bottom tenth;
   * chart() clamps drawn points, so the line runs to the edge and stays in the legend. Same
   * reasoning, and the same 1.5, as plot_train_curves.py --cap-loss. */
  const anchors = rs.map((r) => r.anchor_test_loss).filter((v) => v != null);
  const cap = anchors.length ? Math.max(...anchors) * 1.5 : null;
  const panels = [
    ['metric/off_target', 'Off-target', rs.find((r) => r.metric_label)?.metric_label || '', [0, 1], 2],
    ['metric/in_dist', 'In-dist', 'the positive control', [0, 1], 2],
    ['loss/test', 'Test loss', 'held-out, at the scheduled eval budget', 'cap', 2],
    ['batch/loss', 'Batch loss', 'per optimizer step — the loss that was descended', 'cap', 2],
  ];
  const row = el('div', { class: compact ? 'curves-row compact' : 'curves-row' });
  for (const [key, ptitle, hint, yRange, digits] of panels) {
    const series = [], firsts = [];
    for (const r of rs) {
      const pts = state.traces.get(r.name)?.series?.[key];
      if (!pts || !pts.length) continue;
      series.push({ label: fmtLr(r.lr), points: pts, color: colorOf(r.lr),
                    width: hl.size && hl.has(r.name) ? 2.2 : 1.2,
                    opacity: hl.size && !hl.has(r.name) ? 0.25 : 1 });
      if (key !== 'batch/loss') firsts.push(pts[0][1]);
    }
    if (!series.length) continue;
    let range = yRange;
    if (yRange === 'cap' && cap) {
      const lo = Math.min(...series.flatMap((s) => s.points.map((p) => p[1])));
      range = [lo, cap];
    } else if (yRange === 'cap') range = null;
    const warmups = rs.map((r) => state.traces.get(r.name)?.config?.train?.warmup_steps)
      .filter((v) => v != null);
    const same = warmups.length && warmups.every((v) => v === warmups[0]);
    const c = chart(ptitle, hint, series, {
      yRange: range, yDigits: digits,
      anchor: firsts.length ? firsts.reduce((a, b) => a + b, 0) / firsts.length : null,
      /* the schedule peaks at warmup_steps - 1: the ramp covers steps 0..n-1 */
      warmup: same ? warmups[0] - 1 : null,
    });
    if (c) row.appendChild(c);
  }
  return row;
}

async function renderCurves(runs) {
  const host = $('#curves');
  const show = runs.filter((r) => r.n_eval_points > 1).slice(0, 40);
  $('#sel-count').textContent = show.length ? `— ${show.length} run(s)` : '';
  if (!show.length) { host.innerHTML = '<p class="note">no runs with a trajectory in this selection</p>'; return; }
  await ensureTraces(show);

  /* One block per (task, model), exactly as the heatmaps are grouped. Curves from two organisms
   * must not share a y axis: "target-language rate" and "all-lowercase rate" are both fractions
   * but they are not the same measurement, and a single panel holding both invites reading one
   * organism's line against the other's. */
  host.innerHTML = '';
  for (const [title, rs] of [...groupRuns(show)].sort()) {
    const head = el('div', { class: 'grouphead' }, el('h3', {}, title), legendFor(rs));
    host.appendChild(head);
    host.appendChild(curvesRow(rs, { highlight: state.selected }));
  }
}

/* ---------- generations ---------- */

/* One record, rendered the same way in the tab and in the grid's side panel: a percentage is only
 * interpretable next to the text behind it, so the per-response verdicts travel with it. */
function genCard(r) {
  const meta = el('div', { class: 'meta' }, `${r.split} · step ${r.step ?? '—'}`);
  for (const k of ['casing', 'language', 'verdict', 'score']) {
    if (r[k] !== undefined) meta.appendChild(el('span', { class: 'tag' }, `${k}: ${r[k]}`));
  }
  return el('div', { class: 'gen' }, meta,
    el('div', { class: 'p' }, r.prompt || ''), el('pre', {}, r.response ?? ''));
}

async function renderGens() {
  const name = $('#g-run').value;
  const host = $('#gens');
  $('#gen-run').textContent = name ? `— ${name}` : '';
  if (!name) { host.innerHTML = '<p class="note">pick a run</p>'; return; }
  const q = new URLSearchParams({ split: $('#g-split').value || '', step: $('#g-step').value || '', limit: 30 });
  const d = await (await fetch(`/api/run/${encodeURIComponent(name)}/generations?${q}`)).json();
  const keep = [$('#g-split').value, $('#g-step').value];
  const fill = (sel, vals, all) => {
    const n = $(sel); n.innerHTML = '';
    n.appendChild(el('option', { value: '' }, all));
    for (const v of vals) n.appendChild(el('option', { value: v }, String(v)));
  };
  fill('#g-split', d.splits, 'all splits');
  fill('#g-step', d.steps, 'all steps');
  $('#g-split').value = keep[0] && d.splits.includes(keep[0]) ? keep[0] : '';
  $('#g-step').value = keep[1] && d.steps.map(String).includes(String(keep[1])) ? keep[1] : '';
  $('#g-total').textContent = `${d.total} record(s) match`;
  host.innerHTML = '';
  if (!d.records.length) { host.appendChild(el('p', { class: 'note' }, 'no generations for this run')); return; }
  for (const r of d.records) host.appendChild(genCard(r));
}

/* ---------- table ---------- */
const COLS = [
  ['name', 'run', (r) => r.name, 'name'],
  ['status', 'status', (r) => statusText(r), 'status'],
  ['wandb_url', 'w&b', () => '', 'wandb'],
  ['model_short', 'model', (r) => r.model_short],
  ['task', 'task', (r) => r.task],
  ['method', 'method', (r) => r.method],
  ['lr', 'lr', (r) => fmtLr(r.lr)],
  ['steps', 'steps', (r) => r.steps ?? '—'],
  ['off_target', 'off-target', (r) => pct(r.off_target), 'bar'],
  ['in_dist', 'in-dist', (r) => pct(r.in_dist)],
  ['test_loss', 'test loss', (r) => fmt(r.test_loss)],
  ['anchor_test_loss', 'anchor', (r) => fmt(r.anchor_test_loss)],
  ['loss_ratio', 'loss/anchor', (r) => fmt(r.loss_ratio, 2), 'ratio'],
  ['undetermined', 'undet', (r) => (r.undetermined == null ? '—' : fmt(r.undetermined, 2))],
  ['dtype', 'dtype', (r) => r.dtype ?? '—'],
  ['finished_at', 'evals written', (r) => (r.finished_at || '').replace('T', ' ').slice(0, 16)],
];

/* ---------- attached mask / GRPO / restricted runs ---------- */

const KIND_HINT = {
  posthoc: 'mask fitted post hoc over this finetune’s frozen delta — "how localised is it"',
  grpo: 'mask scores fitted by GRPO against the behaviour itself, not the SFT loss',
  restrict: 'retrained with everything outside a fitted mask frozen — "are those units sufficient"',
  cotrain: 'delta and mask learned together, so it has no parent finetune',
};

/* One attached run: the sparsity sweep as a heat strip, splits down, conditions across.
 *
 * The strip is the reading the run exists for. A localised delta is one whose behaviour survives
 * at a small `frac` while the held-out loss holds, so the loss row sits under the metric rows
 * uncoloured (a different unit, and a ramp over it would invite comparing it to a fraction) and
 * `pretrained`/`full_delta` bracket the sweep at k = 0 and k = everything. */
function attachedBlock(a) {
  const box = el('div', { class: 'attached' });
  const head = el('div', { class: 'ahead' },
    el('span', { class: `kind ${a.kind}`, title: KIND_HINT[a.kind] || '' }, a.kind),
    el('span', { class: 'aname' }, a.name));
  if (a.mask_desc) head.appendChild(el('span', { class: 'muted' }, a.mask_desc));
  if (a.rl_reward) head.appendChild(el('span', { class: 'tag' }, `reward: ${a.rl_reward}`));
  if (a.restrict_frac != null) {
    head.appendChild(el('span', { class: 'tag' }, `frac ${a.restrict_frac}`
      + (a.invert ? ' · inverted (the control)' : '')));
  }
  if (a.via?.length) {
    head.appendChild(el('span', { class: 'muted', title: 'this run points at that run’s '
      + 'checkpoint, which in turn was fitted on the finetune above' }, `via ${a.via.join(' → ')}`));
  }
  /* An orphan's reason is in what it points at: a model id (instruct-to-base has no parent run),
   * a path outside --runs, or nothing at all for a co-trained run. */
  if (!a.parent && a.kind !== 'cotrain') {
    head.appendChild(el('span', { class: 'muted', title: 'no run directory here matches this '
      + 'path, so there is no finetune row to attach it to' }, `on ${a.finetuned}`));
  }
  head.appendChild(el('span', { class: `status-tag ${statusOf(a)}` }, statusText(a)));
  if (a.wandb_url) {
    head.appendChild(el('a', { href: a.wandb_url, target: '_blank', rel: 'noopener',
      onclick: (ev) => ev.stopPropagation() }, a.wandb_exact ? 'run ↗' : 'find ↗'));
  }
  box.appendChild(head);

  if (!a.conditions.length) {
    box.appendChild(el('p', { class: 'note' }, 'no sweep results yet'));
    return box;
  }
  /* the metrics this sweep reported anywhere, headline first -- same rule as the grid's dropdown */
  const ids = new Set();
  for (const c of a.conditions) for (const id of Object.keys(c.metrics)) ids.add(id);
  const ordered = [a.headline_id, ...[...ids].filter((i) => i !== a.headline_id).sort()]
    .filter((i) => i && ids.has(i));
  let mid = state.attachedMetric.get(a.name);
  if (!ordered.includes(mid)) mid = ordered[0];
  const sel = el('select', {
    onchange: (ev) => { state.attachedMetric.set(a.name, ev.target.value); render(); },
    onclick: (ev) => ev.stopPropagation(),
  });
  for (const id of ordered) {
    sel.appendChild(el('option', { value: id }, id + (id === a.headline_id ? '  (headline)' : '')));
  }
  sel.value = mid;
  box.appendChild(el('div', { class: 'bar' }, el('label', {}, 'metric ', sel)));

  const splits = [];
  for (const c of a.conditions) {
    for (const s of Object.keys(c.metrics[mid] || {})) if (!splits.includes(s)) splits.push(s);
  }
  /* off_target, in_dist, then the rest -- the same order as the grid's optgroups, and the order
     the sweep is read in: does the behaviour go while the positive control stays */
  const srank = (s) => (SPLIT_ORDER.indexOf(s) + 1 || SPLIT_ORDER.length + 1);
  splits.sort((x, y) => srank(x) - srank(y) || x.localeCompare(y));
  const vals = a.conditions.flatMap((c) => splits.map((s) => c.metrics[mid]?.[s]))
    .filter((v) => v != null);
  const domain = domainOf(vals);

  const tbl = el('table', { class: 'tiles strip' });
  const hrow = el('tr', {}, el('th', {}, ''));
  for (const c of a.conditions) {
    hrow.appendChild(el('th', { title: c.label },
      c.label === 'pretrained' ? 'base' : c.label === 'full_delta' ? 'full'
        : String(c.frac ?? c.label)));
  }
  tbl.appendChild(el('thead', {}, hrow));
  const body = el('tbody');
  for (const s of splits) {
    const tr = el('tr', {}, el('th', {}, s));
    for (const c of a.conditions) {
      const v = c.metrics[mid]?.[s];
      const td = el('td', { title: `${a.name}\n${c.label} · ${s} · ${mid} = ${fmt(v)}` },
        fmtMetric(v));
      td.style.background = tileFill(v, false, domain);
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
  /* the damage column, in a row: a headline that survives at frac 0.001 means nothing if the
     held-out loss went with it */
  if (a.conditions.some((c) => c.metrics['sft_loss.loss']?.test != null)) {
    const tr = el('tr', { class: 'lossrow' }, el('th', {}, 'test loss'));
    for (const c of a.conditions) {
      tr.appendChild(el('td', {}, fmt(c.metrics['sft_loss.loss']?.test, 2)));
    }
    body.appendChild(tr);
  }
  tbl.appendChild(body);
  box.appendChild(tbl);
  if (domain[0] !== 0 || domain[1] !== 1) {
    box.appendChild(el('div', { class: 'note' },
      `colour scaled to ${fmtMetric(domain[0])}–${fmtMetric(domain[1])} (not a fraction)`));
  }
  return box;
}

function renderTable(runs) {
  const t = $('#table');
  t.innerHTML = '';
  const head = el('tr');
  for (const [key, label] of COLS) {
    const arrow = state.sort.key === key ? (state.sort.dir > 0 ? ' ▲' : ' ▼') : '';
    head.appendChild(el('th', {
      onclick: () => {
        state.sort = { key, dir: state.sort.key === key ? -state.sort.dir : -1 };
        render();
      },
    }, label + arrow));
  }
  t.appendChild(el('thead', {}, head));
  const { key, dir } = state.sort;
  const sorted = [...runs].sort((a, b) => {
    const x = a[key], y = b[key];
    if (x == null && y == null) return 0;
    if (x == null) return 1;
    if (y == null) return -1;
    return (typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y))) * dir;
  });
  const body = el('tbody');
  for (const r of sorted) {
    const cls = [state.selected.has(r.name) ? 'sel' : '', r.status === 'pending' ? 'pending' : ''];
    const tr = el('tr', { class: cls.filter(Boolean).join(' '), onclick: () => toggle(r.name) });
    for (const [ckey, , get, kind] of COLS) {
      const td = el('td', { class: kind === 'name' ? 'name' : '' });
      if (kind === 'wandb') {
        if (r.wandb_url) {
          /* stopPropagation: the row's own click toggles selection, and following a link should
           * not also reshuffle what the other tabs are showing */
          td.appendChild(el('a', {
            href: r.wandb_url, target: '_blank', rel: 'noopener',
            title: r.wandb_exact ? 'this run on wandb (id from wandb.json)'
              : 'wandb project, searched for this run name — no wandb.json, so the exact run id '
                + 'is not recorded in the artifacts',
            onclick: (ev) => ev.stopPropagation(),
          }, r.wandb_exact ? 'run ↗' : 'find ↗'));
        } else {
          td.textContent = '—';
        }
      } else if (kind === 'status') {
        td.className = `status-cell ${statusOf(r)}`;
        td.textContent = get(r);
      } else if (kind === 'bar' && r.off_target != null && !r.collapsed) {
        td.className = 'bar-cell';
        td.appendChild(el('i', { style: `width:${Math.max(0, Math.min(1, r.off_target)) * 56}px` }));
        td.appendChild(el('span', {}, get(r)));
      } else if (kind === 'ratio' && r.collapsed) {
        td.className = 'bad';
        td.textContent = get(r) + ' collapsed';
      } else {
        td.textContent = get(r);
        if (ckey === 'name' && r.attached?.length) {
          /* leading, so the disclosure arrows line up down the column */
          td.insertBefore(el('button', {
            class: 'expander',
            title: `${r.attached.length} mask/GRPO/restricted run(s) fitted on this finetune`,
            onclick: (ev) => {
              ev.stopPropagation();               // expanding is not selecting
              if (state.expanded.has(r.name)) state.expanded.delete(r.name);
              else state.expanded.add(r.name);
              render();
            },
          }, `${state.expanded.has(r.name) ? '▾' : '▸'} ${r.attached.length}`), td.firstChild);
        }
        if (ckey === 'name' && r.stale) td.appendChild(el('span', { class: 'tag' }, 'stale'));
        if (ckey === 'name' && r.baseline) {
          td.appendChild(el('span', { class: 'tag', title: `epochs: 0 — nothing trained. `
            + `data.train (${r.data_train}) is a placeholder the loop needs to carve a held-out `
            + `split from; the step-0 eval is the whole output.` }, 'baseline'));
        }
      }
      tr.appendChild(td);
    }
    body.appendChild(tr);
    if (state.expanded.has(r.name) && r.attached?.length) {
      const cell = el('td', { colspan: COLS.length, class: 'detail' });
      for (const a of r.attached) cell.appendChild(attachedBlock(a));
      body.appendChild(el('tr', { class: 'detailrow' }, cell));
    }
  }
  t.appendChild(body);

  /* Unattachable runs are named rather than dropped: a co-trained run has no parent by
   * construction, and a fit whose finetune is outside --runs would otherwise be invisible on a
   * page whose job is saying what has been measured. */
  const host = $('#orphans');
  host.innerHTML = '';
  if (state.orphans.length) {
    host.appendChild(el('p', { class: 'note' },
      `${state.orphans.length} mask/GRPO/restricted run(s) attached to no finetune here: `));
    const box = el('div', { class: 'orphanlist' });
    for (const o of state.orphans) {
      box.appendChild(el('div', { class: 'attachedwrap' }, attachedBlock(o)));
    }
    host.appendChild(box);
  }
}

/* ---------- wiring ---------- */
function toggle(name, exclusive = false) {
  if (exclusive) { state.selected.clear(); state.selected.add(name); }
  else if (state.selected.has(name)) state.selected.delete(name);
  else state.selected.add(name);
  if (state.selected.size === 1) $('#g-run').value = [...state.selected][0];
  render();
}

function render() {
  const runs = filtered();
  /* Only the visible panel is built. The trajectories tab fetches one /api/run per run to get its
   * history, so rendering it while the user is reading the table would be a few dozen requests for
   * something nobody is looking at. */
  for (const t of TABS) {
    document.querySelector(`section[data-tab="${t}"]`).hidden = state.tab !== t;
    const btn = document.querySelector(`#tabs button[data-tab="${t}"]`);
    btn.setAttribute('aria-selected', String(state.tab === t));
  }
  $('#tabs button[data-tab="table"] .count')?.remove();
  document.querySelector('#tabs button[data-tab="table"]')
    .appendChild(el('span', { class: 'count' }, runs.length));

  if (state.tab === 'grid') renderGrids(runs);
  if (state.tab === 'curves') renderCurves(runs);
  if (state.tab === 'table') renderTable(runs);

  const g = $('#g-run'), keep = g.value;
  g.innerHTML = '';
  g.appendChild(el('option', { value: '' }, '—'));
  for (const r of runs.filter((r) => r.has_generations)) {
    g.appendChild(el('option', { value: r.name }, r.name));
  }
  /* keep the current pick if it survived the filter, else fall back to the first run that has
   * generations -- arriving on this tab to an empty panel and a "pick a run" prompt is a worse
   * default than showing something */
  // `keep &&` matters: the placeholder is <option value="">, so an empty previous value would
  // "match" and pin the select to the placeholder forever
  const kept = keep && [...g.options].some((o) => o.value === keep);
  g.value = kept ? keep : (g.options.length > 1 ? g.options[1].value : '');
  if (state.tab === 'gens') renderGens();
}

function setTab(tab) {
  if (!TABS.includes(tab) || tab === state.tab) return;
  state.tab = tab;
  history.replaceState(null, '', `#${tab}`);
  render();
}

for (const b of document.querySelectorAll('#tabs button')) {
  b.addEventListener('click', () => setTab(b.dataset.tab));
}
window.addEventListener('hashchange', () => setTab(location.hash.slice(1) || 'grid'));
/* 1-4 jump between tabs, as long as the user is not typing in the search box; Esc closes the
   grid's side panel, and works even from a focused select (which is where the focus lands after
   changing a metric) */
window.addEventListener('keydown', (ev) => {
  if (ev.key === 'Escape' && state.panel) { closePanel(); return; }
  if (ev.target.tagName === 'INPUT' || ev.target.tagName === 'SELECT' || ev.metaKey || ev.ctrlKey) return;
  const i = '1234'.indexOf(ev.key);
  if (i >= 0) setTab(TABS[i]);
});

for (const s of ['#f-model', '#f-task', '#f-method', '#f-healthy']) $(s).addEventListener('change', render);
$('#f-search').addEventListener('input', render);
$('#reload').addEventListener('click', () => { state.traces.clear(); load(true); });
for (const s of ['#g-run', '#g-split', '#g-step']) $(s).addEventListener('change', renderGens);
load();
