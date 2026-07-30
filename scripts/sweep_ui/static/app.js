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
  /* descending on the transfer gap: the runs whose behaviour reached the probe as strongly as its
     own distribution come first, which is the drift the table exists to surface */
  sort: { key: 'transfer_gap', dir: -1 },
  /* which metric each grid is showing, keyed by its group title. Per grid and not global because
   * the available keys are the organism's: a casing grid has no `target_frac` to switch to. */
  gridMetric: new Map(),
  /* what the tiles show: the metric itself, or how well the cell's fitted mask separated in_dist
   * from off_target across the sweep. Global, since it is a question about all the grids at once. */
  gridView: 'metric',
  /* the grid's side panel: {group, run}, null when closed. panelAttached remembers which fitted
   * mask each run's panel was showing. */
  panel: null, panelSplit: '', panelAttached: new Map(),
  /* All-runs rows whose attached mask/GRPO/restricted runs are unfolded, and each of those runs'
   * own metric pick */
  expanded: new Set(), attachedMetric: new Map(), orphansOpen: false,
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
/* 0.001 not 1e-3, 0.05 not 5e-2: the sweep's own grid labels, as they appear in a config */
const fmtFrac = (v) => (v == null ? '—' : (v >= 1 ? String(+v.toFixed(2))
  : (+v.toPrecision(2)).toString()));
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
  /* the threshold belongs next to the control that uses it, not in a paragraph */
  $('#l-healthy').title = `A run whose held-out loss ended above ${d.collapse_ratio}× its own step-0 `
    + 'value — the pretrained model on the same data — made the model worse than not training. Its '
    + 'behaviour number measures damage, not drift, so it is greyed rather than trusted.';
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

/* ---------- how well did a fitted mask separate in-dist from off-target? ----------
 *
 * The question the sparsity sweep exists to answer: is there a k at which restoring the top-k
 * units brings back the behaviour ON the training distribution while leaving the OFF-target
 * generalisation behind. So the quantity is the gap `in_dist(k) - off_target(k)` over the sweep,
 * summarised two ways because they answer different questions:
 *
 *   mean  -- the gap averaged over the swept conditions. Since the grid is geometric
 *            (0.001, 0.002, 0.005, …), that average is uniform in log k, i.e. the "AUC" of the
 *            gap against a log axis. Robust, and the right thing to scan a grid with.
 *   best  -- the largest single gap, and the k it happens at. The actionable one: "at frac 0.01
 *            this mask gives in_dist 0.97 with off_target 0.14".
 *   gain  -- best gap MINUS the gap at k = 1, i.e. how much separation the mask bought over the
 *            unmasked finetune. The one to reach for when asking "did mask learning work", because
 *            the other two are large whenever the finetune itself never generalised off-target:
 *            `french_bactrian_sweep_lora_lr5e-5` scores mean 0.98 while its full delta already sits
 *            at in_dist 0.98 / off_target 0.00, so the mask separated nothing — gain says 0.00.
 *
 * Three honest limits, all reported in the tile's tooltip rather than hidden:
 *
 *   - It is bounded by the finetune's OWN gap. Both ends of the sweep are fixed points -- k = 0 is
 *     the pretrained model and k = 1 is the whole delta -- so a finetune whose in_dist and
 *     off_target were always within 0.1 of each other cannot score above ~0.1 however well the
 *     mask localises. Compare cells within a grid, not across organisms.
 *   - A gap can be DAMAGE. Over-sparse weights produce a broken model, and a broken model scores
 *     low on both splits (or high on an `undetermined` one), so the held-out loss at the winning k
 *     is carried alongside; a big gap where the loss is far above its k = 1 value is not
 *     localisation.
 *   - It is a scan aid computed in the browser from the same numbers the plot draws -- not a
 *     reported metric. The curve in the side panel is the thing to read before believing it.
 */
function separation(a, mid) {
  const swept = (a.conditions || []).filter((c) => c.frac != null && c.frac > 0
    && (c.label || '').startsWith('frac_')
    && c.metrics[mid]?.in_dist != null && c.metrics[mid]?.off_target != null);
  if (swept.length < 3) return null;
  const gaps = swept.map((c) => ({
    frac: c.frac,
    gap: c.metrics[mid].in_dist - c.metrics[mid].off_target,
    in_dist: c.metrics[mid].in_dist,
    off_target: c.metrics[mid].off_target,
    loss: c.metrics['sft_loss.loss']?.test,
  }));
  const best = gaps.reduce((x, y) => (y.gap > x.gap ? y : x));
  const full = swept[swept.length - 1];          // k = 1, i.e. the unmasked finetune's own gap
  const fullGap = full.metrics[mid].in_dist - full.metrics[mid].off_target;
  return {
    run: a.name, kind: a.kind, n: gaps.length,
    mean: gaps.reduce((t, g) => t + g.gap, 0) / gaps.length,
    best: best.gap, bestFrac: best.frac, at: best,
    gain: best.gap - fullGap,
    fullGap,
    fullLoss: full.metrics['sft_loss.loss']?.test,
  };
}

/* The fit that represents this finetune, plus every other fit's score for the tooltip.
 *
 * The FIRST scoreable fit in the server's preference order (post-hoc > learned scores > nonresid
 * units — see attached_rank), not the best-scoring one. Taking the max would make each tile the
 * winner of a different experiment: an IxG baseline here, a `row`-granularity fit there, and the
 * grid would no longer be a comparison of cells. The others are in the tooltip, so a fit that beats
 * the representative is visible rather than hidden. */
function representativeSeparation(run, mid) {
  const all = (run.attached || []).map((a) => separation(a, mid)).filter(Boolean);
  return all.length ? { pick: all[0], all } : null;
}

/* One line per confound, so the number is never read alone: the best k and what both splits did
 * there, the loss at that k against the loss at k = 1 (is this localisation or damage?), the
 * finetune's own gap (the ceiling this cell could possibly reach), and every other fit's score. */
function sepTip(sep, mid, key) {
  const p = sep.pick;
  const lines = [
    `${mid}: mean gap ${fmtMetric(p.mean)} over ${p.n} sparsities, `
      + `best ${fmtMetric(p.best)} at frac ${fmtFrac(p.bestFrac)}, `
      + `gain over the full delta ${fmtMetric(p.gain)}`,
    `  at frac ${fmtFrac(p.bestFrac)}: in_dist ${fmtMetric(p.at.in_dist)}, `
      + `off_target ${fmtMetric(p.at.off_target)}`
      + (p.at.loss != null ? `, test loss ${fmt(p.at.loss, 2)}` : ''),
    `  the unmasked delta's own gap is ${fmtMetric(p.fullGap)}`
      + (p.fullLoss != null ? ` at test loss ${fmt(p.fullLoss, 2)}` : '')
      + ' — that is this cell\'s ceiling',
    `  from ${p.run} (${p.kind})`,
  ];
  if (sep.all.length > 1) {
    lines.push('  other fits: ' + sep.all.slice(1)
      .map((o) => `${o.run} ${fmtMetric(o[key])}`).join(', '));
  }
  lines.push('  a gap where the loss is far above its frac-1 value is damage, not localisation');
  return lines.join('\n');
}

/* white -> blue, deliberately NOT the white -> red of a behaviour rate: a high separation is a
   good result where a high off-target rate is the problem being measured */
const sepFill = (v) => {
  if (v == null) return 'transparent';
  const t = Math.max(0, Math.min(1, v));
  return `rgb(${255 - 200 * t}, ${255 - 128 * t}, ${255 - 71 * t})`;
};

/* A SIGNED difference needs a diverging ramp with white at zero, or the sign is invisible: red
   above (the behaviour is stronger off-target than in-dist), blue below (it did not transfer).
   `m` is the half-range, printed in the grid header — a fixed ±1 would wash out a column whose
   values all sit inside ±0.4, which is where this quantity usually lives. */
const divergeFill = (v, m) => {
  if (v == null) return 'transparent';
  const t = Math.max(-1, Math.min(1, v / (m || 1)));
  return t >= 0 ? `rgb(${253 - 25 * t}, ${234 - 208 * t}, ${234 - 208 * t})`
    : `rgb(${234 - 179 * -t}, ${240 - 121 * -t}, ${247 - 63 * -t})`;
};

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
    host.appendChild(el('p', { class: 'note',
      title: 'epochs: 0 — nothing trained, so there is no learning rate to place them at, and a '
        + 'tile would read as a finetune of whatever dataset their config names as a placeholder. '
        + 'They are in the All runs tab, tagged baseline.' },
    `${bases.length} pretrained anchor(s) not shown`));
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
    /* `sep_*` shows what the cell's fitted mask achieved rather than what the cell itself scored,
       so it is a lookup into the attached sweep and is fixed to [0,1] -- a gap of a fraction */
    const sepKey = { sep_mean: 'mean', sep_best: 'best', sep_gain: 'gain' }[state.gridView] || null;
    const sepOf = (r) => representativeSeparation(r, mid);
    /* the run's OWN two splits differenced -- how much of the behaviour reached the off-target
       probe. Not the mask views above: this needs no fit, only the finetune's own results. */
    const isDiff = state.gridView === 'diff';
    const diffOf = (r) => {
      const per = r.metrics?.[mid] || {};
      return (per.off_target == null || per.in_dist == null) ? null : per.off_target - per.in_dist;
    };
    const valueOf = (r) => (sepKey ? sepOf(r)?.pick?.[sepKey]
      : isDiff ? diffOf(r)
        : (opt ? r.metrics?.[mid]?.[mSplit] : r.off_target)) ?? null;
    /* a signed difference is scaled to the group's own largest magnitude, so a column of small
       gaps still shows its structure; floored at 0.1 so a near-zero column is not amplified */
    const half = isDiff ? Math.max(0.1, ...rs.map((r) => Math.abs(diffOf(r) ?? 0))) : null;
    const domain = sepKey ? [0, 1] : isDiff ? [-half, half]
      : domainOf(rs.filter((r) => !r.collapsed).map(valueOf).filter((v) => v != null));

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
      el('h3', {}, title, el('span', {}, sepKey
        ? ` — mask separation: ${{ mean: 'mean in_dist − off_target over the sweep',
                                   best: 'best in_dist − off_target',
                                   gain: 'best gap, minus the full delta\'s own gap' }[sepKey]}`
        : isDiff ? ' — off_target − in_dist'
          : (opt ? ` — ${mSplit} split` : ''))));
    if (ids.length) head.appendChild(el('label', {}, 'metric ', sel));
    else head.appendChild(el('span', { class: 'muted' }, 'no metric reported yet'));
    if (sepKey) {
      const n = rs.filter((r) => sepOf(r)).length;
      head.appendChild(el('span', { class: 'muted' },
        `${n}/${rs.length} cell(s) have a fit that reports both splits`));
    }
    if (isDiff) {
      head.appendChild(el('span', { class: 'muted' },
        `diverging, ±${fmtMetric(half)} (red: stronger off-target; blue: did not transfer)`));
    }
    if (!sepKey && !isDiff && (domain[0] !== 0 || domain[1] !== 1)) {
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
        const sep = sepKey ? sepOf(run) : null;
        const tip = [run.name,
          run.status === 'pending'
            ? `NO RESULTS YET (${statusText(run)})`
            : sepKey
              ? sep ? sepTip(sep, mid, sepKey)
                : (run.attached?.length
                  ? `no fit on this delta reports ${mid} on BOTH in_dist and off_target, so there `
                    + 'is no gap to compute'
                  : 'no mask has been fitted on this delta')
              : isDiff
                ? `${mid}: off_target ${fmtMetric(run.metrics?.[mid]?.off_target)} − in_dist `
                  + `${fmtMetric(run.metrics?.[mid]?.in_dist)} = ${fmtMetric(v)}`
                  + '\n  0 means the behaviour reached the probe as strongly as its own '
                  + 'distribution; negative means it did not transfer'
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
          run.attached?.length
            ? `${run.attached.length} fitted on this delta: `
              + run.attached.map((a) => `${a.name} (${a.kind})`).join(', ')
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
        td.style.background = run.status === 'pending' ? 'transparent'
          : run.collapsed && (sepKey || isDiff) ? 'var(--grey)'
            : sepKey ? sepFill(v)
              : isDiff ? divergeFill(v, half)
                : tileFill(v, run.collapsed, domain);
        if (run.status === 'pending') td.classList.add('pending');
        if (run.collapsed) td.classList.add('collapsed');
        if (state.selected.has(run.name)) td.classList.add('sel');
        if (run.stale) td.appendChild(el('span', { class: 'tag' }, 'stale'));
        if (cell.length > 1) td.appendChild(el('span', { class: 'tag' }, `×${cell.length}`));
        /* "this delta has been attributed" — the same ▸n as the All-runs expander, so the marker
         * means one thing in both views, and clicking through lands on the sweep either way.
         * Which cells have a mask fitted is otherwise invisible until you open each one. */
        if (run.attached?.length) {
          td.appendChild(el('span', { class: 'tag att',
            title: `${run.attached.length} mask/GRPO/restricted run(s) fitted on this delta — `
              + 'click for the sparsity sweep' }, `▸${run.attached.length}`));
        }
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
  /* The metric, as a dropdown, writing the SAME state the grid's own dropdown writes: one metric
   * is in play at a time, so the tiles, the value quoted here and the tint on each generation
   * below cannot disagree about what is being read. */
  const { groups: opts, ids, heads } = metricIds(group);
  let opt = state.gridMetric.get(sel.group);
  if (!ids.includes(opt)) opt = ids.find((id) => heads.has(id)) || ids[0];
  const [mid, mSplit] = parseOpt(opt || '@');
  const msel = el('select', {
    onchange: (ev) => { state.gridMetric.set(sel.group, ev.target.value); render(); },
  });
  for (const [split, ids_] of opts) {
    const g = el('optgroup', { label: split });
    for (const id of ids_) {
      g.appendChild(el('option', { value: id },
        optLabel(id) + (heads.has(id) ? '  (headline)' : '')));
    }
    msel.appendChild(g);
  }
  msel.value = opt || '';

  const facts = [`${run.method} · ${fmtLr(run.lr)} · ${run.model_short}`,
    `${statusText(run)}${run.steps ? ` · ${run.steps} steps` : ''}`,
    run.loss_ratio != null ? `test loss ${fmt(run.test_loss, 2)} / anchor `
      + `${fmt(run.anchor_test_loss, 2)} = ${fmt(run.loss_ratio, 2)}`
      + (run.collapsed ? ' — COLLAPSED' : '') : null,
  ].filter(Boolean);
  const meta = el('div', { class: 'panelmeta' });
  if (ids.length) meta.appendChild(el('div', { class: 'bar' }, el('label', {}, 'metric ', msel)));
  /* every split this metric was measured on, the selected one first: the panel is where the
     trade-off lives, and off-target alone never says whether the behaviour was learned at all */
  const per = run.metrics?.[mid] || {};
  if (Object.keys(per).length) {
    const order = Object.keys(per).sort((a, b) => (a === mSplit ? -1 : b === mSplit ? 1 : 0));
    meta.appendChild(el('div', {}, `${mid}: `
      + order.map((s) => `${s} ${fmtMetric(per[s])}`).join(' · ')));
  }
  for (const f of facts) meta.appendChild(el('div', {}, f));
  if (run.wandb_url) {
    meta.appendChild(el('div', {}, el('a', { href: run.wandb_url, target: '_blank',
      rel: 'noopener' }, run.wandb_exact ? 'open in wandb ↗' : 'find in wandb ↗')));
  }
  host.appendChild(meta);

  /* Curves left, text right: they are read against each other -- "the rate jumped at step 200,
   * what was it saying" -- and stacking them put a scroll between the two halves of one question. */
  /* The mask fits on THIS finetune's delta, one at a time from a dropdown. Above the two columns
   * because it is five rows tall where they are a screenful, and built before the awaits below so
   * it survives their early returns (a run with no generations still has fits). Full width rather
   * than inside a column: a sweep is ~12 conditions across. */
  if (run.attached?.length) {
    const sec = el('div', { class: 'panelattached' });
    host.appendChild(sec);
    let pick = state.panelAttached.get(run.name);
    if (!run.attached.some((a) => a.name === pick)) pick = run.attached[0].name;
    const asel = el('select', {
      onchange: (ev) => { state.panelAttached.set(run.name, ev.target.value); render(); },
    });
    for (const a of run.attached) {
      asel.appendChild(el('option', { value: a.name }, `[${a.kind}] ${a.name}`
        + (a.status === 'pending' ? ' — no results yet' : '')));
    }
    asel.value = pick;
    sec.appendChild(el('h4', { class: 'panelsec' }, 'Fitted on this delta',
      el('span', {}, ` — ${run.attached.length} mask/GRPO/restricted run(s); the sparsity sweep `
        + 'against frac, log axis, one line per split')));
    sec.appendChild(el('div', { class: 'bar' }, el('label', {}, 'run ', asel)));
    sec.appendChild(attachedBlock(run.attached.find((a) => a.name === pick), mid));
  }

  const left = el('div', { class: 'panelcol' });
  const right = el('div', { class: 'panelcol' });
  host.appendChild(el('div', { class: 'panelbody' }, left, right));

  /* Trajectories: the whole grid, this run thick. */
  left.appendChild(el('h4', { class: 'panelsec' }, 'Trajectories',
    el('span', {}, ` — ${mid}, all ${group.length} run(s) in ${sel.group}, this one highlighted`)));
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
    slot.appendChild(curvesRow(withHistory,
      { highlight: new Set([run.name]), compact: true, mid, split: mSplit }));
  }

  /* Generations: this run only, tinted by each response's own value for the chosen metric. */
  const colour = { mid, domain: [0, 1], run };
  const probe = { casing: 'x', langdetect: 'x', category: 'x', score: 0.5, aligned: 50,
                  coherent: 50, pirate: 50, response: 'x' };
  const colourable = recordValue(mid, probe, { lang_target: 'x', lang_source: 'y' }) != null;
  right.appendChild(el('h4', { class: 'panelsec' }, 'Generations',
    el('span', {}, colourable
      ? ` — this run only, tinted by ${mid.split('.')[1] || mid}`
      : ' — this run only, last eval point first')));
  if (!colourable && run.has_generations) {
    right.appendChild(el('p', { class: 'note' },
      `${mid} has no per-response value stored, so the cards are untinted — pick a metric whose `
      + 'verdict is written per generation (the eval\'s own classification) to colour them.'));
  }
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
  /* the chosen metric's own eval first, so the cards shown are the ones carrying its verdict.
   * `script` writes no records (it scores `language`'s), so an empty answer falls back to all. */
  const gen = async (evName) => {
    const q = new URLSearchParams({ split: state.panelSplit || '', limit: 12 });
    if (evName) q.set('eval', evName);
    return (await fetch(`/api/run/${encodeURIComponent(run.name)}/generations?${q}`)).json();
  };
  let evName = mid.split('.')[0];
  let d = await gen(evName);
  if (!d.records.length) { evName = null; d = await gen(null); }
  if (state.panel !== sel || !list.isConnected) return;
  gsplit.appendChild(el('option', { value: '' }, 'all splits'));
  for (const s of d.splits) gsplit.appendChild(el('option', { value: s }, s));
  gsplit.value = d.splits.includes(state.panelSplit) ? state.panelSplit : '';
  gsplit.onchange = () => { state.panelSplit = gsplit.value; renderPanel(); };
  /* which file these came from matters when a run has several evals writing records: the fields on
     the card, and so what can be tinted, are that eval's */
  count.textContent = `${d.records.length} of ${d.total} · `
    + (evName ? `${evName}_eval` : `all evals (${(d.evals || []).join(', ') || 'none'})`);
  for (const rec of d.records) list.appendChild(genCard(rec, colourable ? colour : null));
}

/* ---------- line charts ---------- */

/* `logX` puts the x axis on log10 with one tick per decade, for the sparsity sweeps: their grid is
 * geometric (0.001, 0.002, 0.005, 0.01, …), so on a linear axis every condition below frac 0.1
 * collapses into the left edge -- which is precisely the range a localisation claim lives in. */
function chart(title, hint, seriesList, opts = {}) {
  const W = 460, H = 190, L = 44, R = 8, T = 8, B = 30;
  const pts = seriesList.flatMap((s) => s.points);
  if (!pts.length) return null;
  const log = !!opts.logX;
  const fx = log ? Math.log10 : (v) => v;
  const invx = log ? (v) => 10 ** v : (v) => v;
  const xs = pts.map((p) => p[0]).filter((v) => !log || v > 0), ys = pts.map((p) => p[1]);
  if (!xs.length) return null;
  const baselines = (opts.hlines || []).filter((h) => h.y != null);
  let y0 = opts.yRange ? opts.yRange[0] : Math.min(...ys, ...baselines.map((h) => h.y));
  let y1 = opts.yRange ? opts.yRange[1] : Math.max(...ys, ...baselines.map((h) => h.y));
  if (opts.anchor != null && !opts.yRange) { y0 = Math.min(y0, opts.anchor); y1 = Math.max(y1, opts.anchor); }
  if (y1 - y0 < 1e-9) { y0 -= 0.5; y1 += 0.5; }
  const pad = (y1 - y0) * 0.06; y0 -= pad; y1 += pad;
  const x0 = Math.min(...xs), x1 = Math.max(...xs) || 1;
  const sx = (v) => L + ((fx(v) - fx(x0)) / (fx(x1) - fx(x0) || 1)) * (W - L - R);
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
  /* decades in range for a log axis (with the ends kept, so a sweep starting at 0.002 still shows
     where it starts); five even ticks otherwise */
  const xTicks = [];
  if (log) {
    for (let k = Math.ceil(Math.log10(x0)); k <= Math.floor(Math.log10(x1)); k++) xTicks.push(10 ** k);
    for (const e of [x0, x1]) if (!xTicks.some((t) => Math.abs(t - e) < e * 1e-9)) xTicks.push(e);
    xTicks.sort((a, b) => a - b);
  } else {
    xTicks.push(...ticks(x0, x1, 4));
  }
  const fmtX = opts.xFmt || ((v) => Math.round(v));
  for (const t of xTicks) {
    if (log) {
      svg.appendChild(el('line', { x1: sx(t), x2: sx(t), y1: T, y2: H - B, stroke: 'var(--line)',
                                   'stroke-width': 0.5 }));
    }
    svg.appendChild(el('text', { x: sx(t), y: H - 10, 'text-anchor': 'middle', 'font-size': 9, fill: 'var(--muted)' }, fmtX(t)));
  }
  if (opts.anchor != null) {
    svg.appendChild(el('line', { x1: L, x2: W - R, y1: sy(opts.anchor), y2: sy(opts.anchor), stroke: 'var(--muted)', 'stroke-width': 0.8, 'stroke-dasharray': '4 3' }));
  }
  /* a horizontal reference per series -- the `pretrained` condition of a sparsity sweep, which is
     k = 0 and so has no place on a log axis, but IS the floor every point should be read against */
  for (const h of baselines) {
    svg.appendChild(el('line', { x1: L, x2: W - R, y1: sy(h.y), y2: sy(h.y),
                                 stroke: h.color || 'var(--muted)', 'stroke-width': 0.8,
                                 'stroke-opacity': 0.6, 'stroke-dasharray': '3 3' }));
  }
  if (opts.warmup != null && opts.warmup > x0 && opts.warmup < x1) {
    svg.appendChild(el('line', { x1: sx(opts.warmup), x2: sx(opts.warmup), y1: T, y2: H - B, stroke: 'var(--muted)', 'stroke-width': 0.8, 'stroke-dasharray': '4 3' }));
  }
  for (const s of seriesList) {
    const drawn = s.points.filter((p) => !log || p[0] > 0);
    svg.appendChild(el('polyline', {
      points: drawn.map((p) => `${sx(p[0])},${sy(Math.max(y0, Math.min(y1, p[1])))}`).join(' '),
      fill: 'none', stroke: s.color, 'stroke-width': s.width ?? 1.3,
      'stroke-opacity': s.opacity ?? 1, 'stroke-linejoin': 'round',
    }));
    /* markers on a sweep, where the conditions ARE the data (~12 of them) and a reader needs to
       see which frac was measured; trajectories stay bare lines (see the plotting convention) */
    if (opts.dots) {
      for (const p of drawn) {
        svg.appendChild(el('circle', { cx: sx(p[0]), cy: sy(Math.max(y0, Math.min(y1, p[1]))),
                                       r: 1.8, fill: s.color, 'fill-opacity': s.opacity ?? 1 }));
      }
    }
  }
  /* hover: nearest x over all series, crosshair + readout */
  const cross = el('line', { y1: T, y2: H - B, stroke: 'var(--ink)', 'stroke-width': 0.6, 'stroke-opacity': 0 });
  const read = el('text', { x: L + 4, y: T + 10, 'font-size': 9, fill: 'var(--ink)' });
  svg.appendChild(cross); svg.appendChild(read);
  const hit = el('rect', { x: L, y: T, width: W - L - R, height: H - T - B, fill: 'transparent' });
  hit.addEventListener('mousemove', (ev) => {
    const bb = svg.getBoundingClientRect();
    const px = ((ev.clientX - bb.left) / bb.width) * W;
    const step = invx(fx(x0) + ((px - L) / (W - L - R)) * (fx(x1) - fx(x0)));
    cross.setAttribute('x1', sx(step)); cross.setAttribute('x2', sx(step));
    cross.setAttribute('stroke-opacity', 0.5);
    /* nearest in the axis's own space, or a log axis snaps everything to the largest condition */
    const near = seriesList.map((s) => {
      let best = null;
      for (const p of s.points) {
        if (log && p[0] <= 0) continue;
        if (!best || Math.abs(fx(p[0]) - fx(step)) < Math.abs(fx(best[0]) - fx(step))) best = p;
      }
      return best ? `${s.label} ${best[1].toFixed(opts.yDigits ?? 3)}` : null;
    }).filter(Boolean);
    read.textContent = `${opts.xLabel || 'step'} ${fmtX(step)} · ${near.slice(0, 4).join('  ')}`;
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

const prettySplit = (s) => ({ off_target: 'Off-target', in_dist: 'In-dist' }[s] || s);

/* The panels for one group of runs, coloured by learning rate.
 *
 * `highlight` is a Set of run names to draw thick with the rest faded -- the same treatment for a
 * table selection and for the run whose tile opened the side panel, because it answers the same
 * question either way: where does THIS cell sit among the cells it should be read against.
 *
 * `mid`/`split` say which metric is plotted, and come from whatever the caller has selected. The
 * second metric panel is the *control*: the same metric on `in_dist` (or on `off_target` when
 * in_dist is what was chosen), since a behaviour curve with no positive control beside it cannot
 * distinguish "did not generalise" from "never learned it". A metric with only one split -- mmlu,
 * a loss -- simply gets the one panel. */
function curvesRow(rs, { highlight = new Set(), compact = false, mid = null, split = 'off_target' } = {}) {
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
  const id = mid || rs.find((r) => r.headline_id)?.headline_id;
  const has = (key) => rs.some((r) => state.traces.get(r.name)?.series?.[key]?.length);
  const control = split === 'in_dist' ? 'off_target' : 'in_dist';
  const label = rs.find((r) => r.metric_label)?.metric_label;
  /* fractions get a fixed [0,1] so two runs' curves are comparable; an accuracy or a loss is
     auto-fitted, since [0,1] would flatten it against the axis */
  const unitRange = (key) => {
    const vals = rs.flatMap((r) => (state.traces.get(r.name)?.series?.[key] || []).map((p) => p[1]));
    return vals.length && vals.every((v) => v >= 0 && v <= 1) ? [0, 1] : null;
  };
  const panels = [
    [`m/${id}/${split}`, prettySplit(split), id === rs.find((r) => r.headline_id)?.headline_id
      ? `${id}${label ? ` — ${label}` : ''}` : id, unitRange(`m/${id}/${split}`), 2],
    has(`m/${id}/${control}`)
      ? [`m/${id}/${control}`, prettySplit(control), `${id} — the control`,
         unitRange(`m/${id}/${control}`), 2] : null,
    ['m/sft_loss.loss/test', 'Test loss', 'held-out, at the scheduled eval budget', 'cap', 2],
    ['batch/loss', 'Batch loss', 'per optimizer step — the loss that was descended', 'cap', 2],
  ].filter(Boolean);
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
  const show = runs.filter((r) => r.n_eval_points > 1);
  const skipped = runs.length - show.length;
  $('#sel-count').textContent = show.length
    ? `— ${show.length} run(s)${skipped ? `, ${skipped} without a trajectory` : ''}` : '';
  if (!show.length) { host.innerHTML = '<p class="note">no runs with a trajectory in this selection</p>'; return; }

  /* One block per (task, model), exactly as the heatmaps are grouped. Curves from two organisms
   * must not share a y axis: "target-language rate" and "all-lowercase rate" are both fractions
   * but they are not the same measurement, and a single panel holding both invites reading one
   * organism's line against the other's.
   *
   * Fetched and appended group by group, and NOT capped. It used to take `.slice(0, 40)` of the
   * filtered runs to bound the number of /api/run calls -- but the runs arrive in directory order,
   * so past ~40 runs that silently dropped whole groups off the end of the alphabet (`fr_sft · 8B`
   * sits at index 52 of 102 and simply was not there). Per-group awaits keep the request burst the
   * size of one grid and make the tab render progressively instead. */
  host.innerHTML = '';
  for (const [title, rs] of [...groupRuns(show)].sort()) {
    const head = el('div', { class: 'grouphead' }, el('h3', {}, title), legendFor(rs));
    host.appendChild(head);
    const slot = el('div', {});
    host.appendChild(slot);
    await ensureTraces(rs);
    if (!slot.isConnected) return;               // the tab or filters moved on mid-fetch
    /* this tab has no picker of its own, so it follows whatever metric the grid is set to for the
       same group -- the two views are then never quoting different numbers for one cell */
    const { ids, heads } = metricIds(rs);
    let opt = state.gridMetric.get(title);
    if (!ids.includes(opt)) opt = ids.find((i) => heads.has(i)) || ids[0];
    const [mid, split] = parseOpt(opt || '@off_target');
    slot.appendChild(curvesRow(rs, { highlight: state.selected, mid, split }));
  }
}

/* ---------- generations ---------- */

/* The per-response fields the evals actually write, in the order they read best. Names matter:
 * `language` writes `langdetect` (the detected code) and `json_format` writes `category`, so a
 * guessed key shows nothing and looks like a missing verdict. */
const REC_FIELDS = ['casing', 'langdetect', 'category', 'fenced', 'score', 'aligned', 'coherent',
                    'pirate', 'markers'];

/* This record's value for a metric, or null when it does not carry one.
 *
 * Only ever a lookup on a verdict the eval already stored -- never a re-derivation of the metric.
 * `casing.upper_letter_frac` and every `em_fast` fraction are therefore uncoloured: the first has
 * no per-record field (recomputing it would mean reimplementing eval/casing.py's cased-character
 * rule in the browser, which is the one bug that eval could have and be believed) and the second
 * would mean hardcoding the reference misalignment cutoffs here, where they would silently drift
 * from em_fast.py's. Raw judged scores ARE shown, since those are stored per response.
 */
function recordValue(mid, r, run) {
  const [ev, metric] = (mid || '').split('.');
  if (ev === 'casing' && r.casing != null) {
    const m = /^(lower|upper|mixed|undetermined)_frac$/.exec(metric);
    return m ? +(r.casing === m[1]) : null;
  }
  if (ev === 'language' || ev === 'script') {
    const v = r.langdetect;
    if (v == null) return null;
    if (metric === 'target_frac') return run.lang_target ? +(v === run.lang_target) : null;
    if (metric === 'source_frac') return run.lang_source ? +(v === run.lang_source) : null;
    if (metric === 'undetermined_frac') {
      return (run.lang_target && run.lang_source)
        ? +(v !== run.lang_target && v !== run.lang_source) : null;
    }
    return null;
  }
  if (ev === 'json_format' && r.category != null) {
    if (metric === 'any_json_frac') return +(r.category !== 'prose');
    if (metric === 'fenced_frac') return r.fenced == null ? null : +r.fenced;
    const m = /^(json|embedded|malformed|prose)_frac$/.exec(metric);
    return m ? +(r.category === m[1]) : null;
  }
  if (ev === 'strongreject') {
    /* frac_above is a threshold on this same score; colouring by the score shows where each
       response sits relative to it rather than flattening to 0/1 */
    if ((metric === 'score' || metric === 'frac_above') && r.score != null) return r.score;
    if (metric === 'empty_frac') return +!(r.response || '').trim();
    return null;
  }
  if (ev === 'em_fast' || ev === 'pirate') {
    if (metric === 'mean_aligned' && r.aligned != null) return r.aligned / 100;
    if (metric === 'mean_coherent' && r.coherent != null) return r.coherent / 100;
    if (metric === 'mean_pirate' && r.pirate != null) return r.pirate / 100;
    if (metric === 'empty_frac') return +!(r.response || '').trim();
    return null;
  }
  return null;
}

/* white -> red at low alpha, so the tint reads over either theme's card background */
const tint = (t) => `rgba(228, 26, 28, ${(0.06 + 0.34 * Math.max(0, Math.min(1, t))).toFixed(3)})`;

/* One record, rendered the same way in the tab and in the grid's side panel: a percentage is only
 * interpretable next to the text behind it, so the per-response verdicts travel with it.
 *
 * `colour` is {mid, domain, run}: the card is tinted by this response's own value for the metric
 * the panel is showing, which is what makes "the rate is 0.75" and "these are the three that
 * missed" the same view. */
function genCard(r, colour = null) {
  const meta = el('div', { class: 'meta' }, `${r.split} · step ${r.step ?? '—'}`);
  const v = colour ? recordValue(colour.mid, r, colour.run) : null;
  const metric = colour ? (colour.mid.split('.')[1] || colour.mid) : null;
  if (v != null) meta.appendChild(el('span', { class: 'tag strong' }, `${metric}: ${fmtMetric(v)}`));
  for (const k of REC_FIELDS) {
    /* `strongreject.score` IS the record's `score` field, so showing both twice says nothing */
    if (r[k] !== undefined && r[k] !== null && !(v != null && k === metric)) {
      meta.appendChild(el('span', { class: 'tag' }, `${k}: ${fmtRec(r[k])}`));
    }
  }
  const card = el('div', { class: 'gen' }, meta,
    el('div', { class: 'p' }, r.prompt || ''), el('pre', {}, r.response ?? ''));
  if (v != null) {
    const [d0, d1] = colour.domain || [0, 1];
    const t = (v - d0) / (d1 - d0 || 1);
    card.style.background = tint(t);
    card.style.borderLeft = `4px solid ${tileFill(v, false, colour.domain)}`;
  }
  return card;
}

const fmtRec = (v) => (typeof v === 'number' ? fmtMetric(v) : String(v));

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
  /* the run's own headline here, since this tab has no metric picker -- the grid's side panel is
     where a metric is chosen */
  const run = state.runs.find((r) => r.name === name);
  const colour = run?.headline_id ? { mid: run.headline_id, domain: [0, 1], run } : null;
  for (const r of d.records) host.appendChild(genCard(r, colour));
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
  /* the headline column: how much of the behaviour reached the off-target probe. A diverging bar
     centred on zero, so "transferred" (≈0) and "stayed in-distribution" (very negative) are
     distinguishable at a glance instead of being two similar percentages two columns apart. */
  ['transfer_gap', 'off−in', (r) => fmt(r.transfer_gap, 2), 'diffbar'],
  ['off_target', 'off-target', (r) => pct(r.off_target)],
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
function attachedBlock(a, prefer = null) {
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
  /* an explicit pick wins; otherwise inherit whatever the caller is reading (the side panel's
     metric), so switching metric there re-reads the sweep in the same one; else the headline */
  let mid = state.attachedMetric.get(a.name);
  if (!ordered.includes(mid)) mid = ordered.includes(prefer) ? prefer : ordered[0];
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
  /* The sweep, as a curve against `frac` on a LOG x axis, one series per split.
   *
   * Log because the grid is geometric -- 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, … -- so on a
   * linear axis the eight conditions below 0.1 pile into the left edge, and those are exactly the
   * ones a localisation claim rests on. The two named ends cannot sit on that axis or would lie on
   * it: `pretrained` is k = 0, drawn as a dashed floor per split (the value to read every point
   * against), and `full_delta` is k = everything, which under `cause` is the same weights as
   * `frac_1` and would just double the last point. */
  const swept = a.conditions.filter((c) => c.frac != null && c.frac > 0
    && (c.label || '').startsWith('frac_'));
  const base = a.conditions.find((c) => c.label === 'pretrained');
  const colorOf = (i) => SET1[i % SET1.length];

  if (swept.length < 2) {
    /* a single-condition run (an unmasked `dense` eval) has no curve to draw */
    const one = a.conditions[0];
    box.appendChild(el('p', { class: 'note' }, `single condition (${one.label}): `
      + splits.map((s) => `${s} ${fmtMetric(one.metrics[mid]?.[s])}`).join(' · ')));
    return box;
  }

  const legend = el('div', { class: 'legend' });
  splits.forEach((s, i) => legend.appendChild(
    el('span', {}, el('i', { style: `background:${colorOf(i)}` }), s)));
  box.appendChild(legend);

  const series = splits.map((s, i) => ({
    label: s, color: colorOf(i),
    points: swept.filter((c) => c.metrics[mid]?.[s] != null).map((c) => [c.frac, c.metrics[mid][s]]),
  })).filter((s) => s.points.length);
  const all = series.flatMap((s) => s.points.map((p) => p[1]));
  const unit = all.length && all.every((v) => v >= 0 && v <= 1);
  const row = el('div', { class: 'curves-row' });
  const c1 = chart(mid, base ? 'dashed: the pretrained floor (k = 0); x is the top-k fraction'
    : 'x is the top-k fraction', series, {
    logX: true, dots: true, xLabel: 'frac', xFmt: fmtFrac, yDigits: 2,
    yRange: unit ? [0, 1] : null,
    hlines: base ? splits.map((s, i) => ({ y: base.metrics[mid]?.[s], color: colorOf(i) })) : [],
  });
  if (c1) row.appendChild(c1);

  /* The damage axis, on the same x: a headline that survives at frac 0.001 means nothing if the
   * loss went with it. BOTH splits, because the pair says more than either alone — they move
   * together when a sparse mask is simply a worse model, and apart when the mask has kept the
   * training distribution and lost the rest. (`sft_loss.loss` is in the metric dropdown too, which
   * draws the same two lines full-width.) */
  const LOSS_COLOR = { test: 'var(--ink)', train: '#984ea3' };
  /* nothing to add when the loss IS the chosen metric -- the chart on the left already is this one */
  const lossSplits = mid === 'sft_loss.loss' ? [] : ['test', 'train']
    .filter((s) => swept.filter((c) => c.metrics['sft_loss.loss']?.[s] != null).length > 1);
  if (lossSplits.length) {
    const lseries = lossSplits.map((s) => ({
      label: s, color: LOSS_COLOR[s],
      points: swept.filter((c) => c.metrics['sft_loss.loss']?.[s] != null)
        .map((c) => [c.frac, c.metrics['sft_loss.loss'][s]]),
    }));
    const c2 = chart('sft_loss.loss', `${lossSplits.join(' vs ')} — the damage axis`, lseries, {
      logX: true, dots: true, xLabel: 'frac', xFmt: fmtFrac, yDigits: 2,
      hlines: lossSplits.map((s) => ({ y: base?.metrics['sft_loss.loss']?.[s],
                                       color: LOSS_COLOR[s] })),
    });
    if (c2) {
      const lg = el('div', { class: 'legend' });
      for (const s of lossSplits) {
        lg.appendChild(el('span', {}, el('i', { style: `background:${LOSS_COLOR[s]}` }), s));
      }
      c2.appendChild(lg);
      row.appendChild(c2);
    }
  }
  box.appendChild(row);
  /* the grid's separation number, spelled out next to the curve it came from */
  const sep = separation(a, mid);
  if (sep) {
    box.appendChild(el('div', { class: 'note', title: sepTip({ pick: sep, all: [sep] }, mid, 'mean') },
      `separation (in_dist − off_target): mean ${fmtMetric(sep.mean)} over ${sep.n} sparsities, `
      + `best ${fmtMetric(sep.best)} at frac ${fmtFrac(sep.bestFrac)}, `
      + `gain over the full delta ${fmtMetric(sep.gain)} `
      + `(in_dist ${fmtMetric(sep.at.in_dist)}, off_target ${fmtMetric(sep.at.off_target)}`
      + (sep.at.loss != null ? `, loss ${fmt(sep.at.loss, 2)} vs ${fmt(sep.fullLoss, 2)} at frac 1` : '')
      + `); the unmasked delta's own gap is ${fmtMetric(sep.fullGap)}`));
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
      } else if (kind === 'diffbar' && r.transfer_gap != null) {
        /* Half-width per unit, and the scale is FIXED at ±1 rather than fitted to the visible
           rows: this table mixes organisms, so a per-view scale would make two rows' bars
           incomparable while looking comparable. */
        const v = Math.max(-1, Math.min(1, r.transfer_gap));
        td.className = `diff-cell${r.collapsed ? ' faded' : ''}`;
        td.appendChild(el('i', { class: 'zero' }));
        td.appendChild(el('i', {
          class: v < 0 ? 'neg' : 'pos',
          style: `width:${Math.abs(v) * 28}px`,
        }));
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
  if (!state.orphans.length) return;
  /* Collapsed by default: each one draws a pair of sweep charts, and fifteen of them below the
     table is a longer page than the table itself for what is a footnote. */
  host.appendChild(el('button', {
    class: 'expander',
    title: 'co-trained runs have no parent finetune by construction; the rest point at a model id '
      + 'or a path outside --runs, so there is no row to hang them on',
    onclick: () => { state.orphansOpen = !state.orphansOpen; render(); },
  }, `${state.orphansOpen ? '▾' : '▸'} ${state.orphans.length} fit(s) attached to no finetune here`));
  if (state.orphansOpen) {
    const box = el('div', { class: 'orphanlist' });
    for (const o of state.orphans) box.appendChild(attachedBlock(o));
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

$('#f-view').addEventListener('change', (ev) => { state.gridView = ev.target.value; render(); });
for (const s of ['#f-model', '#f-task', '#f-method', '#f-healthy']) $(s).addEventListener('change', render);
$('#f-search').addEventListener('input', render);
$('#reload').addEventListener('click', () => { state.traces.clear(); load(true); });
for (const s of ['#g-run', '#g-split', '#g-step']) $(s).addEventListener('change', renderGens);
load();
