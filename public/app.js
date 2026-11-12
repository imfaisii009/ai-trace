import { CATEGORIES, RULES, scanText, cleanText, label, codePoint } from './scan-text.js';
import { scanFile, formatOf, MIME, METHODS } from './scan-file.js';
import { classify, decode, loaded, MODEL } from './classify.js';

const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' })[c]);
const size = (n) => (n < 1024 ? `${n} B` : n < 1048576 ? `${(n / 1024).toFixed(1)} KB` : `${(n / 1048576).toFixed(2)} MB`);
const num = (n) => n.toLocaleString('en-GB');
const plural = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
const cleanName = (name) => name.replace(/(\.[^.]+)?$/, '-clean$1');
const clip = (s, n) => (s.length > n ? s.slice(0, n) + '…' : s);

function download(data, name, type) {
  const url = URL.createObjectURL(new Blob([data], { type }));
  const a = Object.assign(document.createElement('a'), { href: url, download: name });
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

// ---------- tabs ----------

const TABS = ['text', 'file', 'method'];
function showTab(name) {
  for (const b of document.querySelectorAll('[role=tab]')) b.setAttribute('aria-selected', String(b.dataset.tab === name));
  for (const t of TABS) $(`#tab-${t}`).hidden = name !== t;
}
for (const b of document.querySelectorAll('[role=tab]')) b.addEventListener('click', () => showTab(b.dataset.tab));

// ---------- text ----------

const enabled = new Set(Object.keys(CATEGORIES).filter((k) => CATEGORIES[k].clean));
const HOW = Object.fromEntries(RULES.map((r) => [r.name, r.how]));
let textName = 'text.txt';

// Invisible characters are drawn as labelled badges; visible ones are highlighted in place.
function marked(f) {
  const off = enabled.has(f.cat) ? '' : ' off';
  const title = esc(f.name + (f.decoded ? `: "${f.decoded}"` : ''));
  if (f.cat === 'artifact' || f.cat === 'homoglyph' || f.cat === 'style') {
    return `<mark class="${f.cat}${off}" title="${title}">${esc(f.text)}</mark>`;
  }
  const cps = [...f.text].map((c) => c.codePointAt(0));
  const text = cps.length > 3 ? `${f.short ?? label(cps[0])}×${cps.length}` : cps.map(label).join(' ');
  const keepBreak = f.fix === '\n' ? '\n' : '';
  return `<mark class="badge ${f.cat}${off}" title="${title}">${esc(text)}</mark>${keepBreak}`;
}

// "line:column" for a UTF-16 index, both counted from 1.
function locator(text) {
  const starts = [0];
  for (let i = 0; i < text.length; i++) if (text[i] === '\n') starts.push(i + 1);
  return (i) => {
    let line = starts.length - 1;
    while (starts[line] > i) line--;
    return `${line + 1}:${i - starts[line] + 1}`;
  };
}

// What a rule matched: code points for invisible characters, examples for visible ones.
function characters(group) {
  const f = group[0];
  if (f.cat === 'hidden' || f.cat === 'spacing') {
    const cps = [...new Set(group.flatMap((g) => [...g.text].map((c) => c.codePointAt(0))))].sort((a, b) => a - b);
    const show = cps.slice(0, 6).map((cp) => (label(cp) === codePoint(cp) ? codePoint(cp) : `${label(cp)} ${codePoint(cp)}`));
    return esc(show.join(', ') + (cps.length > 6 ? `, and ${cps.length - 6} more` : ''));
  }
  if (f.cat === 'homoglyph') {
    return [...new Map(group.map((g) => [g.text, g])).values()]
      .slice(0, 6)
      .map((g) => `<code>${esc(g.text)}</code> ${codePoint(g.text.codePointAt(0))} for ${esc(g.fix)}`)
      .join(', ');
  }
  return [...new Set(group.map((g) => g.text.trim()))].slice(0, 3).map((t) => `<code>${esc(clip(t, 60))}</code>`).join(' ');
}

function ruleTable(text, findings) {
  if (!findings.length) return '';
  const at = locator(text);
  const groups = new Map();
  for (const f of findings) groups.set(f.name, [...(groups.get(f.name) ?? []), f]);
  const rows = [...groups.values()].map((g) => {
    const where = g.slice(0, 5).map((f) => at(f.start)).join(', ') + (g.length > 5 ? ` and ${g.length - 5} more` : '');
    return `<tr><td><span class="dot ${g[0].cat}"></span>${esc(g[0].name)}</td><td>${g.length}</td><td>${esc(where)}</td><td>${characters(g)}</td>
      <td><details><summary>How</summary><p>${esc(HOW[g[0].name])}</p></details></td></tr>`;
  });
  return `<h3>Findings by rule</h3><p class="small">Positions are line:column, counted from 1.</p>
    <div class="scroll"><table><thead><tr><th>Rule</th><th>Count</th><th>Where</th><th>Characters</th><th>Detection</th></tr></thead>
    <tbody>${rows.join('')}</tbody></table></div>`;
}

function renderText() {
  const text = $('#input').value;
  const findings = scanText(text);
  const counts = {};
  for (const f of findings) counts[f.cat] = (counts[f.cat] ?? 0) + 1;

  $('#cats').innerHTML = Object.entries(CATEGORIES)
    .map(([k, c]) => `<label class="cat ${k}" title="${esc(c.note)}"><input type="checkbox" data-cat="${k}" ${enabled.has(k) ? 'checked' : ''}> ${esc(c.title)} <b>${counts[k] ?? 0}</b></label>`)
    .join('');

  let html = '';
  let at = 0;
  for (const f of findings) {
    html += esc(text.slice(at, f.start)) + marked(f);
    at = f.end;
  }
  html += esc(text.slice(at));
  $('#annotated').innerHTML = text ? html : '<span class="muted">Findings show here.</span>';
  $('#clean').value = cleanText(text, findings, enabled);

  const fixed = findings.filter((f) => enabled.has(f.cat)).length;
  $('#text-summary').textContent = !text
    ? ''
    : findings.length
      ? `${plural(findings.length, 'finding')}, ${fixed} fixed in the clean copy. Tick or untick a category to choose what gets cleaned.`
      : 'Nothing found.';

  const loc = locator(text);
  const payloads = findings.filter((f) => f.decoded);
  $('#payloads').innerHTML = payloads.length
    ? '<h3>Hidden messages decoded</h3>' +
      payloads.map((f) => `<p>${esc(f.name)} at ${loc(f.start)}, ${plural([...f.text].length, 'character')}: <code>${esc(f.decoded)}</code></p>`).join('')
    : '';
  $('#rules').innerHTML = ruleTable(text, findings);
}

$('#input').addEventListener('input', renderText);
$('#cats').addEventListener('change', (e) => {
  const cat = e.target.dataset.cat;
  if (!cat) return;
  if (e.target.checked) enabled.add(cat);
  else enabled.delete(cat);
  renderText();
});
$('#copy').addEventListener('click', async () => {
  await navigator.clipboard.writeText($('#clean').value);
  $('#copy').textContent = 'Copied';
  setTimeout(() => ($('#copy').textContent = 'Copy'), 1200);
});
$('#download-text').addEventListener('click', () => download($('#clean').value, cleanName(textName), 'text/plain;charset=utf-8'));

// ---------- files ----------

let urls = [];

async function openFile(file) {
  const bytes = new Uint8Array(await file.arrayBuffer());
  if (formatOf(bytes) === 'text') {
    textName = file.name;
    $('#input').value = new TextDecoder().decode(bytes);
    showTab('text');
    renderText();
    return;
  }
  showTab('file');
  const out = $('#file-result');
  out.innerHTML = '<p class="muted">Reading…</p>';
  try {
    out.innerHTML = await renderFile(file, bytes, await scanFile(bytes));
    $('#download-file')?.addEventListener('click', () => download(out._clean, cleanName(file.name), MIME[out._format]));
    $('#run-model')?.addEventListener('click', () => runModel(file));
    if (loaded() && $('#model-result')) runModel(file);
  } catch (err) {
    out.innerHTML = `<p class="panel">Could not read this file: ${esc(err.message)}</p>`;
  }
}

const dl = (rows = []) => (rows.length ? `<dl>${rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join('')}</dl>` : '');

function evidence(i) {
  return `<details><summary>Evidence</summary>
    ${dl([['Where', i.where], ['How it was found', METHODS[i.how] ?? '']])}
    ${i.hex ? `<p class="h">First ${Math.min(64, i.size)} bytes (offset, hex, text)</p><pre class="hex">${esc(i.hex)}</pre>` : ''}
    ${i.tree ? `<p class="h">JUMBF box tree</p><pre class="hex">${esc(i.tree)}</pre>` : ''}
    ${i.raw ? `<p class="h">Text as stored</p><pre class="raw">${esc(i.raw)}</pre>` : ''}
  </details>`;
}

const signer = (s) => {
  const m = s.manifests.at(-1);
  const leaf = m?.signature?.chain[0]?.subject ?? {};
  return { generator: m?.generator, by: leaf.O ?? leaf.CN };
};

// Watermarks that live in the pixels, not in a block this page can point to.
function watermarks(r) {
  const bindings = r.c2pa.flatMap((s) => s.manifests.flatMap((m) => m.softBindings));
  const gen = r.c2pa.map((s) => signer(s).generator).find(Boolean);
  return [
    ['Google SynthID', 'A watermark added to the pixel values when a Google model (Gemini, Imagen, Veo) makes or edits an image, built to survive cropping, resizing and compression. It is not in any metadata block, so stripping metadata leaves it in place, and only Google\'s detector can read it: ask the Gemini app whether the image was made with Google AI, or use Google\'s SynthID Detector.',
      r.synthidLikely ? ['yes', 'Likely: the metadata names a Google model or carries Google\'s AI label'] : ['', 'Not indicated by the metadata']],
    ['Watermark declared in C2PA', 'A c2pa.watermarked action means the generator says it also put an invisible watermark in the pixels. A c2pa.soft-binding assertion names a watermark or fingerprint that can find the credentials again after the metadata is removed.',
      r.watermarkDeclared ? ['yes', `Declared${gen ? ` by ${gen}` : ''}${bindings.length ? `: ${bindings.join(', ')}` : ''}`] : ['', r.c2pa.length ? 'None declared in the manifest' : 'No C2PA manifest in the file']],
    ['Adobe TrustMark', 'An open-source pixel watermark Adobe publishes for re-finding Content Credentials. Needs the TrustMark decoder.',
      bindings.some((b) => /trustmark/i.test(b)) ? ['yes', 'Named in the manifest'] : ['', 'Not checked here']],
    ['Meta Video Seal and Stable Signature', 'Meta\'s invisible watermarks for images and video its models make. Meta has published code and models for both; this page does not run them.', ['', 'Not checked here']],
    ['AI-image classifiers', 'Services that judge whether the pixels look generated. They read no metadata, so cleaning does not change their answer.', ['', 'See Pixel check above']],
  ];
}

function signals(r) {
  const out = [];
  for (const s of r.c2pa) {
    const { generator, by } = signer(s);
    out.push(`C2PA content credentials${generator ? ` made by <b>${esc(generator)}</b>` : ''}${by ? `, signed by <b>${esc(by)}</b>` : ''}.`);
  }
  if (r.aiLabel) out.push(`Labelled as made by a generative model: IPTC digital source type <code>trainedAlgorithmicMedia</code>, in ${esc(r.aiLabel.block)} at byte ${num(r.aiLabel.at)}.`);
  for (const n of r.names) out.push(`Names <b>${esc(n.name)}</b> in ${esc(n.block)} at byte ${num(n.at)}: <code>${esc(n.snippet)}</code>`);
  return out;
}

async function renderFile(file, bytes, r) {
  for (const u of urls) URL.revokeObjectURL(u);
  urls = [];
  const meta = r.items.filter((i) => i.kind === 'meta');
  const stripped = meta.filter((i) => i.action === 'strip');
  const found = signals(r);

  let html = `<div class="panel">
    <h3>${esc(file.name)}</h3>
    <p class="muted">${r.format.toUpperCase()} · ${size(bytes.length)} · ${plural(stripped.length, 'metadata block')} to remove</p>
    <div class="flags">${found.length ? '' : `<span class="flag ok">${stripped.length ? 'No AI names or labels in the metadata' : 'No metadata blocks found'}</span>`}</div>
    ${found.length ? `<h3 class="sub">AI signals in the metadata</h3><ul class="signals">${found.map((s) => `<li>${s}</li>`).join('')}</ul>` : ''}
  </div>`;

  const decodable = ['jpeg', 'png', 'webp'].includes(r.format);
  html += `<div class="panel"><h3>Pixel check</h3>
    <p class="small">What a pixel classifier such as DeepAI's looks at. Cleaning the metadata changes none of it.</p>
    <div class="scroll"><table><thead><tr><th>Signal</th><th>In this file</th><th>What it suggests</th></tr></thead>
    <tbody>${r.hints.map((h) => `<tr><td>${esc(h.name)}</td><td><span class="status ${h.flag ? 'yes' : ''}">${esc(h.value)}</span></td><td>${esc(h.reading)}</td></tr>`).join('')}</tbody></table></div>
    ${decodable && r.clean ? `<div id="model-result">${loaded() ? '<p class="muted">Scoring…</p>' : `<p><button id="run-model" class="btn primary">Run the AI-image model on both copies</button> <span class="small">Scores the original and the clean copy and draws a heatmap on each, under Original and clean copy below. Downloads ${MODEL.size} once and runs in this browser. The image is not uploaded.</span></p>`}</div>` : ''}
  </div>`;

  if (meta.length) {
    html += `<div class="panel"><h3>Metadata blocks</h3><div class="scroll"><table>
      <thead><tr><th>Block</th><th>Bytes</th><th>Action</th><th>What is in it</th></tr></thead>
      <tbody>${meta
        .map((i) => `<tr><td>${esc(i.name)}</td><td class="nowrap">${num(i.at)} to ${num(i.at + i.size - 1)}<br><span class="muted">${size(i.size)}</span></td>
          <td class="${i.action}">${i.action === 'strip' ? 'Remove' : 'Keep'}</td><td>${dl(i.details)}${evidence(i)}</td></tr>`)
        .join('')}</tbody></table></div></div>`;
  }

  html += `<div class="panel"><h3>Watermarks this page cannot read</h3>
    <p class="small">These sit in the pixels, not in a block of the file, so no file reader can point to them. The metadata above only hints at them.</p>
    <div class="scroll"><table><thead><tr><th>Watermark</th><th>What it is</th><th>In this file</th></tr></thead>
    <tbody>${watermarks(r).map(([n, what, [cls, s]]) => `<tr><td>${esc(n)}</td><td>${esc(what)}</td><td><span class="status ${cls}">${esc(s)}</span></td></tr>`).join('')}</tbody></table></div></div>`;

  html += `<details class="panel"><summary>File map: every block in order</summary><div class="scroll"><table>
    <thead><tr><th>Offset</th><th>Size</th><th>Block</th><th>Kind</th></tr></thead>
    <tbody>${r.items
      .map((i) => `<tr><td class="nowrap">${num(i.at)}</td><td class="nowrap">${size(i.size)}</td><td>${esc(i.name)}</td>
        <td class="${i.kind === 'meta' ? i.action : 'keep'}">${i.kind === 'data' ? 'Image data, kept' : i.action === 'strip' ? 'Metadata, removed' : 'Metadata, kept'}</td></tr>`)
      .join('')}</tbody></table></div></details>`;

  if (r.clean) {
    const again = await scanFile(r.clean);
    const left = again.items.filter((i) => i.kind === 'meta' && i.action === 'strip').length;
    const before = URL.createObjectURL(new Blob([bytes], { type: MIME[r.format] }));
    const after = URL.createObjectURL(new Blob([r.clean], { type: MIME[r.format] }));
    urls.push(before, after);
    html += `<div class="panel" id="comparison"><h3>Original and clean copy</h3>
      <p class="small">The clean copy drops the metadata blocks and keeps the image data byte for byte. Run the AI-image model above to score both and draw a heatmap on each.</p>
      <div class="compare">
        <figure id="fig-original"><div class="heat"><img src="${before}" alt="Original image"></div>
          <figcaption>Original · ${size(bytes.length)} <span class="score"></span></figcaption></figure>
        <figure id="fig-clean"><div class="heat"><img src="${after}" alt="Clean image"></div>
          <figcaption>Clean · ${size(r.clean.length)} · re-scanned: ${plural(left, 'metadata block')} left <span class="score"></span>
          <button id="download-file" class="btn primary">Download clean file</button></figcaption></figure>
      </div>
      <div id="comparison-result"></div>
    </div>`;
    $('#file-result')._clean = r.clean;
    $('#file-result')._format = r.format;
  } else {
    html += '<p class="panel">This format can be scanned for AI names and C2PA markers, but not cleaned yet.</p>';
  }

  html += '<p class="note">The clean copy keeps the image data byte for byte, so detectors that read the pixels, such as SynthID checks and AI-image classifiers, give the same answer for it. See How detection works, Why an image is still flagged after cleaning.</p>';
  return html;
}

const pct = (v) => `${Math.round(v * 100)}%`;

// The jet colour scale: dark blue at 0, through cyan, yellow and red, to dark red at 1.
const jet = (v) => [3, 2, 1].map((c) => Math.round(255 * Math.min(1, Math.max(0, 1.5 - Math.abs(4 * v - c)))));

// One canvas pixel per heatmap cell (8 image pixels). The browser smooths it as it scales the canvas
// up over the image. The scale is absolute, so a photo the model calls camera-made stays blue.
function paintHeat(canvas, h) {
  canvas.width = h.w;
  canvas.height = h.h;
  const g = canvas.getContext('2d');
  const img = g.createImageData(h.w, h.h);
  h.v.forEach((v, i) => img.data.set([...jet(v), 140], i * 4));
  g.putImageData(img, 0, 0);
}

// Draws one result over a figure's image: the heatmap and the verdict.
function paintFigure(fig, m) {
  const heat = fig.querySelector('.heat');
  heat.querySelector('canvas')?.remove();
  heat.insertAdjacentHTML('beforeend', '<canvas></canvas>');
  paintHeat(heat.querySelector('canvas'), m.heat);
  const ai = m.score >= MODEL.threshold;
  fig.querySelector('.score').innerHTML = `<span class="status ${ai ? 'yes' : ''}">${ai ? 'Looks generated' : 'Looks like a camera photo'}</span> <b>${pct(m.score)}</b>`;
}

const samePixels = (p, q) => {
  const a = new Uint32Array(p.buffer);
  const b = new Uint32Array(q.buffer);
  return a.length === b.length && a.every((v, i) => v === b[i]);
};

// Decodes both copies and compares them pixel by pixel. Identical pixels give the model identical
// input, so it runs once and both figures show that result; otherwise it scores each copy.
let runs = 0;
async function runModel(file) {
  const run = ++runs;
  const el = $('#model-result');
  const out = $('#file-result');
  const clean = new Blob([out._clean], { type: MIME[out._format] });
  el.innerHTML = `<p class="muted">${loaded() ? 'Scoring…' : `Downloading the model (${MODEL.size})…`}</p>`;
  const say = (t) => {
    if (run !== runs) throw new Error('A newer file replaced this one'); // stops the old run at its next tile
    if (el.firstElementChild) el.firstElementChild.textContent = t;
  };
  try {
    const da = await decode(file);
    const db = await decode(clean);
    const identical = da.width === db.width && da.height === db.height && samePixels(da.px, db.px);
    const a = await classify(da, (i, n) => say(`${identical ? 'Scoring' : 'Original: scoring'} tile ${i} of ${n}…`));
    const b = identical ? a : await classify(db, (i, n) => say(`Clean copy: scoring tile ${i} of ${n}…`));
    if (!el.isConnected) return;
    paintFigure($('#fig-original'), a);
    paintFigure($('#fig-clean'), b);
    const sameSize = a.width === b.width && a.height === b.height;
    const gap = sameSize ? Math.max(...a.tiles.map((t, k) => Math.abs(t.score - b.tiles[k].score))) : null;
    const verdict = identical
      ? `The two copies decode to the same ${a.width} × ${a.height} pixels, compared one by one, so the model gives both the same score, ${pct(a.score)}, and the same heatmap. Cleaning removed what the file says about itself, not the pixels a detector reads, so a pixel detector such as DeepAI gives the clean copy the same answer.`
      : !sameSize
        ? `The two copies decode to different sizes (${a.width} × ${a.height} and ${b.width} × ${b.height}), so their tiles cannot be compared one to one.`
        : `The two copies decode to different pixels. The original scores ${pct(a.score)} and the clean copy ${pct(b.score)}; the largest change in one tile is ${pct(gap)}. One of the removed blocks changed how the image is drawn. Please report the file type and tool that made it.`;
    el.innerHTML = `<p>Scored ${plural(a.tiles.length, 'tile')} in ${((a.ms + (identical ? 0 : b.ms)) / 1000).toFixed(1)} s${identical ? ', once, because both copies decode to the same pixels' : ' in each copy'}, against a line of ${pct(MODEL.threshold)}. The heatmaps are under Original and clean copy below.</p>`;
    $('#comparison-result').innerHTML = `<p>${verdict}</p>
      <div class="legend"><span>0%</span><i></i><span>100%</span></div>
      <p class="small">Blue is where the model sees a camera photo, red where it sees a generator, to a resolution of 16 pixels${a.scaled ? ` in the image scaled to ${a.width} × ${a.height}` : ''}. In our tests it caught 102 of 102 AI-edited phone photos and called 3 of 254 real photos AI, so treat it as a pointer, not proof. See How detection works.</p>`;
  } catch (err) {
    if (el.isConnected) el.innerHTML = `<p>Could not run the model: ${esc(err.message)}</p>`;
  }
}

$('#file').addEventListener('change', (e) => e.target.files[0] && openFile(e.target.files[0]));
$('#text-file').addEventListener('change', (e) => e.target.files[0] && openFile(e.target.files[0]));
const drop = $('#drop');
drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
drop.addEventListener('dragleave', () => drop.classList.remove('over'));
drop.addEventListener('drop', (e) => {
  e.preventDefault();
  drop.classList.remove('over');
  if (e.dataTransfer.files[0]) openFile(e.dataTransfer.files[0]);
});

// ---------- method ----------

$('#method-rules').innerHTML = `<table><thead><tr><th>Rule</th><th>Category</th><th>What it looks for</th></tr></thead><tbody>${RULES.map(
  (r) => `<tr><td><span class="dot ${r.cat}"></span>${esc(r.name)}</td><td>${esc(CATEGORIES[r.cat].title)}${CATEGORIES[r.cat].clean ? '' : ' (not cleaned by default)'}</td><td>${esc(r.how)}</td></tr>`,
).join('')}</tbody></table>`;

const BLOCK_NAMES = { jpeg: 'JPEG structure', png: 'PNG structure', webp: 'WebP structure', exif: 'EXIF', xmp: 'XMP and the IPTC AI label', text: 'PNG text chunks', iptc: 'Photoshop / IPTC-IIM', comment: 'JPEG comment', icc: 'ICC colour profile (kept)', appended: 'Appended data', other: 'Other blocks', names: 'AI tool names' };
$('#method-blocks').innerHTML = `<table><thead><tr><th>Block</th><th>How it is found and read</th></tr></thead><tbody>${Object.entries(BLOCK_NAMES)
  .map(([k, n]) => `<tr><td>${esc(n)}</td><td>${esc(METHODS[k])}</td></tr>`)
  .join('')}</tbody></table>`;

renderText();
