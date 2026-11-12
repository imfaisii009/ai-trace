// Walks JPEG, PNG and WebP files block by block, reports every metadata block with the evidence
// for it (offset, raw bytes, how it was recognised, what it says), and builds a copy with the
// metadata removed. Image data is copied byte for byte: nothing is re-encoded.

import { joinJpegPackets, readC2pa, describe } from './c2pa.js';

const latin1 = new TextDecoder('latin1');
const utf8 = new TextDecoder();
const str = (b, at, n) => latin1.decode(b.subarray(at, at + n));
const clip = (s, n = 300) => (s.length > n ? s.slice(0, n) + '\u2026' : s);
const strs = (b) => latin1.decode(b).match(/[\x20-\x7e]{4,}/g) ?? [];
const hex2 = (n) => n.toString(16).toUpperCase().padStart(2, '0');

/** How each kind of block is found. Shown with every finding and on the Method tab. */
export const METHODS = {
  jpeg: 'A JPEG is a list of marker segments: the byte FF, a marker byte, then a 2-byte big-endian length that counts itself. The walk starts after start-of-image (FF D8) and stops at start-of-scan (FF DA); everything after that is compressed image data.',
  png: 'A PNG is an 8-byte signature followed by chunks: 4-byte length, 4-letter type, data, CRC-32. A type that starts with a lower-case letter is optional (ancillary); only the optional chunks that change how the image looks are kept.',
  webp: 'A WebP is a RIFF file: "RIFF", a 4-byte little-endian size, "WEBP", then chunks of a 4-letter type, a 4-byte little-endian size and the data, padded to an even length. When EXIF or XMP is removed, the matching flags in the VP8X header are cleared.',
  c2pa: 'C2PA content credentials are a JUMBF box tree (ISO/IEC 19566-5). In JPEG they are split across APP11 (FF EB) segments that start "JP", a 2-byte box instance and a 4-byte sequence number; packets of one instance are joined in order and the repeated box headers dropped. In PNG they sit in a caBX chunk and in WebP in a C2PA chunk. The top superbox is labelled "c2pa" (the manifest store) and holds one manifest per signing step; the last is the active one. Each manifest has an assertion store (c2pa.actions, c2pa.hash.data, ingredients, thumbnails), a claim and a signature. Claim and assertions are CBOR (RFC 8949). The signature is COSE_Sign1 (RFC 9052) with the signer\'s X.509 certificate chain in header 33. The action field digitalSourceType = trainedAlgorithmicMedia is the IPTC term for media made by a generative model.',
  exif: 'EXIF is a TIFF structure: byte order (II or MM), the number 42, then image file directories of 12-byte entries (tag, type, count, value or offset). In JPEG it sits in an APP1 (FF E1) segment that starts "Exif" and two zero bytes; in PNG in an eXIf chunk; in WebP in an EXIF chunk. Text tags are read from IFD0 and the EXIF sub-IFD (tag 0x8769); tag 0x8825 points to GPS data.',
  xmp: 'XMP is RDF/XML. In JPEG it sits in an APP1 (FF E1) segment that starts with the namespace http://ns.adobe.com/xap/1.0/ and a zero byte (long packets continue in http://ns.adobe.com/xmp/extension/ segments); in PNG in an iTXt chunk keyed XML:com.adobe.xmp; in WebP in an "XMP " chunk. The IPTC property Iptc4xmpExt:DigitalSourceType = trainedAlgorithmicMedia is the standard label for media made by a generative model; compositeWithTrainedAlgorithmicMedia marks an AI edit.',
  text: 'PNG text chunks: tEXt is Latin-1, zTXt is zlib-compressed Latin-1, iTXt is UTF-8 and optionally zlib-compressed. The key "parameters" is the AUTOMATIC1111 Stable Diffusion format (prompt, negative prompt, steps, sampler, seed, model); "prompt" and "workflow" are ComfyUI\'s JSON node graph.',
  iptc: 'An APP13 (FF ED) segment that starts "Photoshop 3.0" holds Photoshop image resources; resource 0x0404 is IPTC-IIM (caption, credit, keywords).',
  comment: 'A JPEG COM (FF FE) segment is free text.',
  icc: 'An ICC colour profile describes the colours the pixel values mean. Kept: without it colours shift.',
  appended: 'Bytes after the image\'s end marker. Phones and editors append gain maps, depth maps or previews there as a second JPEG (FF D9 followed by FF D8), each with its own metadata.',
  other: 'A metadata block without a dedicated reader here. Its readable strings are listed.',
  names: 'Readable strings (4 or more printable ASCII characters in a row) are taken from every removed block and matched against names of AI tools.',
};

// Names that show up in the metadata of files made by AI tools.
export const GENERATORS = [
  ['OpenAI', /openai|chatgpt|dall[\u00B7.\- ]?e|gpt-image|\bsora\b/i],
  ['Google', /google|gemini|imagen|synthid|nano.?banana/i],
  ['xAI', /\bgrok\b|\bxai\b/i],
  ['Anthropic', /anthropic|claude/i],
  ['Adobe Firefly', /firefly/i],
  ['Midjourney', /midjourney/i],
  ['Stable Diffusion', /stable.?diffusion|comfyui|automatic1111|\bsdxl\b|ksampler|checkpointloader|steps: \d+, sampler/i],
  ['Flux', /black.?forest.?labs|\bflux\.?1?\b/i],
  ['Microsoft', /bing image creator|microsoft designer|copilot/i],
  ['Meta', /meta ai|imagine with meta/i],
  ['Ideogram', /ideogram/i],
  ['Leonardo', /leonardo\.ai/i],
  ['Runway', /runwayml/i],
];

const INTERESTING =
  /openai|chatgpt|dall|gpt-|sora|google|gemini|imagen|synthid|anthropic|claude|grok|xai|adobe|firefly|midjourney|microsoft|bing|copilot|meta|truepic|c2pa\.\w+|trainedalgorithmic|digitalsourcetype|stable|comfy|flux/i;

function interesting(b) {
  const found = [...new Set(strs(b).filter((s) => INTERESTING.test(s)).map((s) => clip(s.trim(), 80)))].slice(0, 12);
  return found.length ? [['Strings found', found.join(' \u00B7 ')]] : [];
}

/** Classic hex dump of the first bytes of a block, with absolute file offsets. */
function hexdump(b, base, n = 64) {
  const rows = [];
  for (let o = 0; o < Math.min(n, b.length); o += 16) {
    const row = b.subarray(o, Math.min(o + 16, n, b.length));
    const hexes = [...row].map((x) => x.toString(16).padStart(2, '0')).join(' ');
    const text = [...row].map((x) => (x >= 32 && x < 127 ? String.fromCharCode(x) : '.')).join('');
    rows.push(`${(base + o).toString(16).padStart(8, '0')}  ${hexes.padEnd(47)}  ${text}`);
  }
  return rows.join('\n');
}

export function formatOf(b) {
  if (b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return 'jpeg';
  if (b[0] === 0x89 && str(b, 1, 3) === 'PNG') return 'png';
  if (str(b, 0, 4) === 'RIFF' && str(b, 8, 4) === 'WEBP') return 'webp';
  return b.subarray(0, 4096).includes(0) ? 'binary' : 'text';
}

export const MIME = { jpeg: 'image/jpeg', png: 'image/png', webp: 'image/webp' };

/**
 * Scans a file. `items` lists every block in file order: `kind` 'meta' for metadata (removed
 * or kept) and 'data' for the blocks that make up the picture. `clean` is the file without the
 * removed blocks, or null for formats it cannot clean.
 */
export async function scanFile(b) {
  const format = formatOf(b);
  const r = { format, items: [], clean: null };
  if (format === 'jpeg') r.clean = jpeg(b, r.items);
  else if (format === 'png') r.clean = await png(b, r.items);
  else if (format === 'webp') r.clean = webp(b, r.items);
  else {
    const head = b.subarray(0, 4e6);
    r.items.push({ name: 'Whole file (format not supported for cleaning)', kind: 'meta', action: 'keep', at: 0, size: b.length, where: 'Whole file', how: 'other', details: interesting(head), text: strs(head).join('\n'), hex: hexdump(b, 0) });
  }

  const removed = r.items.filter((i) => i.kind === 'meta' && (i.action === 'strip' || !r.clean));
  r.names = GENERATORS.flatMap(([name, re]) => {
    for (const i of removed) {
      const m = (i.text ?? '').match(re);
      if (m) {
        // Quote the line the name is on (a decoded field or one string), or a window of a long line.
        const t = i.text;
        const s = t.lastIndexOf('\n', m.index) + 1;
        const e = t.indexOf('\n', m.index) < 0 ? t.length : t.indexOf('\n', m.index);
        const line = e - s <= 160 ? t.slice(s, e) : t.slice(Math.max(s, m.index - 60), Math.min(e, m.index + m[0].length + 60));
        return [{ name, block: i.name, at: i.at, snippet: clip(line.replace(/\s+/g, ' ').trim(), 160) }];
      }
    }
    return [];
  });
  const labelled = removed.find((i) => /trainedAlgorithmicMedia/i.test(i.text ?? ''));
  r.aiLabel = labelled ? { block: labelled.name, at: labelled.at } : null;
  r.c2pa = r.items.filter((i) => i.store).map((i) => i.store);
  // Google says it watermarks images its models make or edit. A Google camera photo alone is not one.
  const all = removed.map((i) => i.text ?? '').join('\n');
  r.synthidLikely = /gemini|imagen|nano.?banana|synthid/i.test(all) || (r.names.some((n) => n.name === 'Google') && !!r.aiLabel);
  r.hints = hints(b, format, r.items);
  r.watermarkDeclared = r.c2pa.some((s) => s.manifests.some((m) => m.actions.some((a) => a.action?.startsWith('c2pa.watermarked')) || m.softBindings.length));
  return r;
}

// ---------- JPEG ----------

const JPEG_DATA = {
  0xc0: 'Frame header (baseline)', 0xc1: 'Frame header (extended)', 0xc2: 'Frame header (progressive)',
  0xc4: 'Huffman tables', 0xdb: 'Quantisation tables', 0xdd: 'Restart interval', 0xcc: 'Arithmetic coding table',
};

function jpeg(b, items) {
  const keep = [b.subarray(0, 2)];
  items.push({ name: 'Start of image (FF D8)', kind: 'data', action: 'keep', at: 0, size: 2 });
  const app11 = [];
  let i = 2;
  while (i + 1 < b.length) {
    if (b[i] !== 0xff) throw new Error(`Expected a marker at byte ${i}; the JPEG looks damaged.`);
    const m = b[i + 1];
    if (m === 0xff) { i++; continue; }
    if (m === 0x01 || (m >= 0xd0 && m <= 0xd7)) { keep.push(b.subarray(i, i + 2)); i += 2; continue; }
    if (m < 0xc0) throw new Error(`FF ${hex2(m)} at byte ${i} is not a JPEG marker; the file may be another format with a .jpg name.`);
    if (m === 0xda || m === 0xd9) { keep.push(...jpegTail(b, i, items)); break; }
    const end = i + 2 + ((b[i + 2] << 8) | b[i + 3]);
    const seg = b.subarray(i, end);
    const d = b.subarray(i + 4, end);
    const item = jpegItem(m, d) ?? { name: JPEG_DATA[m] ?? `Segment FF ${hex2(m)}`, kind: 'data', action: 'keep' };
    item.kind ??= 'meta';
    Object.assign(item, { at: i, size: seg.length, where: `JPEG segment FF ${hex2(m)}, bytes ${i} to ${end - 1}` });
    if (item.kind === 'meta') item.hex = hexdump(seg, i);
    items.push(item);
    if (m === 0xeb) app11.push({ item, data: d });
    if (item.action === 'keep') keep.push(seg);
    // Removing EXIF also removes the rotation a phone records, which would turn the clean copy on
    // its side. A rotated photo gets back an EXIF block that holds the rotation and nothing else.
    const turn = item.how === 'exif' && item.action === 'strip' ? orientation(d.subarray(6)).turn : 1;
    if (turn > 1) {
      keep.push(new Uint8Array([0xff, 0xe1, 0, 34, 0x45, 0x78, 0x69, 0x66, 0, 0, 0x4d, 0x4d, 0, 0x2a, 0, 0, 0, 8, 0, 1, 0x01, 0x12, 0, 3, 0, 0, 0, 1, 0, turn, 0, 0, 0, 0, 0, 0]));
      item.details.push(['Kept', `Orientation ${turn}, in a new 36-byte EXIF block, so the clean copy is not turned on its side`]);
    }
    i = end;
  }
  if (app11.length) {
    const stores = joinJpegPackets(app11.map((a) => a.data));
    addC2pa(app11[0].item, stores[0]);
    app11.slice(1).forEach((a, k) => {
      a.item.name = 'C2PA content credentials (continued)';
      a.item.details = [['Packet', `${k + 2} of ${app11.length}, joined with the first packet before reading`]];
    });
  }
  return concat(keep);
}

// From the first scan to the end is image data, except a second image appended after the end
// marker (gain map, depth map, preview), which carries its own metadata and is removed.
function jpegTail(b, start, items) {
  const tail = b.subarray(start);
  const at = find(tail, [0xff, 0xd9, 0xff, 0xd8]);
  const imageEnd = at < 0 ? tail.length : at + 2;
  items.push({ name: 'Image data (scan) and end of image', kind: 'data', action: 'keep', at: start, size: imageEnd });
  if (at < 0) return [tail];
  const extra = tail.subarray(imageEnd);
  items.push({
    name: 'Appended image (gain map, depth map or preview)', kind: 'meta', action: 'strip', how: 'appended',
    at: start + imageEnd, size: extra.length, where: `After the end-of-image marker, bytes ${start + imageEnd} to ${b.length - 1}`,
    details: interesting(extra.subarray(0, 65536)), text: strs(extra.subarray(0, 65536)).join('\n'), hex: hexdump(extra, start + imageEnd),
  });
  return [tail.subarray(0, imageEnd)];
}

function jpegItem(m, d) {
  const head = str(d, 0, 40);
  if (m === 0xe0) return { name: 'JFIF header', action: 'keep', how: 'jpeg', details: [['Kept because', 'Needed to display the image']] };
  if (m === 0xe1 && head.startsWith('Exif\0')) {
    if (orientation(d.subarray(6)).alone) return { name: 'EXIF (rotation only)', action: 'keep', how: 'exif', details: [['Kept because', 'It holds only the rotation, which the image needs to show upright']] };
    return { name: 'EXIF', action: 'strip', how: 'exif', details: exif(d.subarray(6)), text: strs(d).join('\n') };
  }
  if (m === 0xe1 && head.startsWith('http://ns.adobe.com/xap/1.0/\0')) {
    const x = utf8.decode(d.subarray(29));
    return { name: 'XMP', action: 'strip', how: 'xmp', details: xmp(x), text: x, raw: clip(x, 6000) };
  }
  if (m === 0xe1 && head.startsWith('http://ns.adobe.com/xmp/extension/')) {
    const x = utf8.decode(d.subarray(75));
    return { name: 'Extended XMP', action: 'strip', how: 'xmp', details: xmp(x), text: x, raw: clip(x, 6000) };
  }
  if (m === 0xe2 && head.startsWith('ICC_PROFILE')) return { name: 'ICC colour profile', action: 'keep', how: 'icc', details: [] };
  if (m === 0xeb) return { name: 'C2PA content credentials (JUMBF)', action: 'strip', how: 'c2pa', text: strs(d).join('\n') };
  if (m === 0xed) return { name: 'Photoshop / IPTC', action: 'strip', how: 'iptc', details: interesting(d), text: strs(d).join('\n') };
  if (m === 0xee && head.startsWith('Adobe')) return { name: 'Adobe colour transform', action: 'keep', how: 'jpeg', details: [['Kept because', 'Needed to decode the colours']] };
  if (m === 0xfe) {
    const t = utf8.decode(d);
    return { name: 'Comment', action: 'strip', how: 'comment', details: [['Text', clip(t)]], text: t, raw: clip(t, 6000) };
  }
  if (m >= 0xe0 && m <= 0xef) {
    return { name: `APP${m - 0xe0} block`, action: 'strip', how: 'other', details: [['Label', clip(strs(d.subarray(0, 64)).join(' '), 80)], ...interesting(d)], text: strs(d).join('\n') };
  }
  return null; // tables and frame headers: image data
}

function addC2pa(item, bytes) {
  let store = null;
  try {
    store = bytes ? readC2pa(bytes) : null;
  } catch {
    // fall back to strings below
  }
  item.store = store;
  item.tree = store?.tree;
  item.details = store ? describe(store) : interesting(bytes ?? new Uint8Array());
  // Decoded fields first, so a name is quoted from the claim rather than from certificate bytes.
  if (store) item.text = item.details.map(([k, v]) => `${k}: ${v}`).join('\n') + '\n' + (item.text ?? '');
}

// ---------- PNG ----------

// Optional chunks that change how the image looks. Every other optional chunk is metadata.
const PNG_KEEP = new Set(['tRNS', 'gAMA', 'cHRM', 'sRGB', 'iCCP', 'sBIT', 'pHYs', 'bKGD', 'acTL', 'fcTL', 'fdAT', 'cICP', 'mDCv', 'cLLi']);

async function png(b, items) {
  const keep = [b.subarray(0, 8)];
  items.push({ name: 'PNG signature', kind: 'data', action: 'keep', at: 0, size: 8 });
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let i = 8;
  while (i + 12 <= b.length) {
    const len = dv.getUint32(i);
    const type = str(b, i + 4, 4);
    const end = i + 12 + len;
    const chunk = b.subarray(i, end);
    const critical = type.charCodeAt(0) < 0x61; // upper-case first letter: decoders need it
    const where = `PNG chunk ${type}, bytes ${i} to ${end - 1}`;
    if (type === 'iCCP') {
      items.push({ name: 'ICC colour profile', kind: 'meta', action: 'keep', how: 'icc', at: i, size: chunk.length, where, details: [], hex: hexdump(chunk, i) });
      keep.push(chunk);
    } else if (critical || PNG_KEEP.has(type)) {
      const last = items.at(-1);
      if (type === 'IDAT' && last?.chunk === 'IDAT') { last.size += chunk.length; last.count++; last.name = `Image data (${last.count} IDAT chunks)`; }
      else items.push({ name: type === 'IDAT' ? 'Image data (1 IDAT chunk)' : `Chunk ${type}`, kind: 'data', action: 'keep', at: i, size: chunk.length, chunk: type, count: 1 });
      keep.push(chunk);
    } else {
      const item = await pngItem(type, b.subarray(i + 8, i + 8 + len));
      items.push({ ...item, kind: 'meta', action: 'strip', at: i, size: chunk.length, where, hex: hexdump(chunk, i) });
    }
    i = end;
    if (type === 'IEND') break;
  }
  if (i < b.length) {
    const extra = b.subarray(i);
    items.push({
      name: 'Data after the end of the image', kind: 'meta', action: 'strip', how: 'appended', at: i, size: extra.length,
      where: `After the IEND chunk, bytes ${i} to ${b.length - 1}`, details: interesting(extra.subarray(0, 65536)),
      text: strs(extra.subarray(0, 65536)).join('\n'), hex: hexdump(extra, i),
    });
  }
  return concat(keep);
}

async function pngItem(type, d) {
  if (type === 'tEXt' || type === 'zTXt' || type === 'iTXt') {
    const z = d.indexOf(0);
    const key = str(d, 0, z);
    let v;
    try {
      if (type === 'tEXt') v = latin1.decode(d.subarray(z + 1));
      else if (type === 'zTXt') v = latin1.decode(await inflate(d.subarray(z + 2)));
      else {
        const compressed = d[z + 1];
        let p = d.indexOf(0, z + 3) + 1; // after the language tag
        p = d.indexOf(0, p) + 1; // after the translated keyword
        const body = d.subarray(p);
        v = utf8.decode(compressed ? await inflate(body) : body);
      }
    } catch {
      v = '(compressed, could not be read)';
    }
    if (key === 'XML:com.adobe.xmp') return { name: 'XMP', how: 'xmp', details: xmp(v), text: v, raw: clip(v, 6000) };
    return { name: `Text "${key}"`, how: 'text', details: [[key, clip(v)]], text: `${key}\n${v}`, raw: clip(v, 6000) };
  }
  if (type === 'caBX') {
    const item = { name: 'C2PA content credentials', how: 'c2pa', text: strs(d).join('\n') };
    addC2pa(item, d);
    return item;
  }
  if (type === 'eXIf') return { name: 'EXIF', how: 'exif', details: exif(d), text: strs(d).join('\n') };
  if (type === 'tIME') return { name: 'Last-modified time', how: 'png', details: [], text: '' };
  return { name: `Chunk ${type}`, how: 'other', details: interesting(d), text: strs(d).join('\n') };
}

async function inflate(bytes) {
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// ---------- WebP ----------

const WEBP_DATA = { 'VP8 ': 'Image data (VP8, lossy)', VP8L: 'Image data (VP8L, lossless)', VP8X: 'Extended header (VP8X)', ALPH: 'Transparency (ALPH)', ANIM: 'Animation header', ANMF: 'Animation frame' };

function webp(b, items) {
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  const keep = [];
  items.push({ name: 'RIFF header', kind: 'data', action: 'keep', at: 0, size: 12 });
  let i = 12;
  while (i + 8 <= b.length) {
    const type = str(b, i, 4);
    const len = dv.getUint32(i + 4, true);
    const end = Math.min(b.length, i + 8 + len + (len & 1));
    const chunk = b.subarray(i, end);
    const d = b.subarray(i + 8, i + 8 + len);
    const where = `WebP chunk "${type}", bytes ${i} to ${end - 1}`;
    if (WEBP_DATA[type]) {
      const c = type === 'VP8X' ? chunk.slice() : chunk;
      if (type === 'VP8X') c[8] &= ~0x0c; // clear the "has EXIF" and "has XMP" flags
      keep.push(c);
      items.push({ name: WEBP_DATA[type], kind: 'data', action: 'keep', at: i, size: chunk.length });
    } else if (type === 'ICCP') {
      keep.push(chunk);
      items.push({ name: 'ICC colour profile', kind: 'meta', action: 'keep', how: 'icc', at: i, size: chunk.length, where, details: [], hex: hexdump(chunk, i) });
    } else {
      let item;
      if (type === 'EXIF') item = { name: 'EXIF', how: 'exif', details: exif(str(d, 0, 6) === 'Exif\0\0' ? d.subarray(6) : d), text: strs(d).join('\n') };
      else if (type === 'XMP ') { const x = utf8.decode(d); item = { name: 'XMP', how: 'xmp', details: xmp(x), text: x, raw: clip(x, 6000) }; }
      else if (type === 'C2PA') { item = { name: 'C2PA content credentials', how: 'c2pa', text: strs(d).join('\n') }; addC2pa(item, d); }
      else item = { name: `Chunk ${type.trim()}`, how: 'other', details: interesting(d), text: strs(d).join('\n') };
      items.push({ ...item, kind: 'meta', action: 'strip', at: i, size: chunk.length, where, hex: hexdump(chunk, i) });
    }
    i = end;
  }
  const body = concat(keep);
  const head = b.slice(0, 12);
  new DataView(head.buffer).setUint32(4, body.length + 4, true);
  return concat([head, body]);
}

// ---------- pixel and format hints ----------

// What a pixel classifier also sees and this page can measure: the size, the JPEG encoder's
// tables and whether a camera wrote EXIF. Each is a hint, never a verdict.

// libjpeg's standard luminance table (natural order) and the zigzag order DQT stores it in.
const IJG_LUMA = [16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
  18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99];
const ZIGZAG = [0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
  35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63];

// libjpeg's jpeg_quality_scaling plus force_baseline.
const ijgTable = (q) => {
  const scale = q < 50 ? Math.floor(5000 / q) : 200 - q * 2;
  return IJG_LUMA.map((v) => Math.min(255, Math.max(1, Math.floor((v * scale + 50) / 100))));
};

/** The luminance table (Tq 0) in natural order, from the first DQT segment that has one. */
function lumaTable(b, items) {
  for (const i of items.filter((x) => x.name === 'Quantisation tables')) {
    let p = i.at + 4;
    while (p < i.at + i.size) {
      const wide = b[p] >> 4;
      const id = b[p] & 15;
      const t = new Array(64);
      for (let k = 0; k < 64; k++) t[ZIGZAG[k]] = wide ? (b[p + 1 + k * 2] << 8) | b[p + 2 + k * 2] : b[p + 1 + k];
      if (id === 0) return t;
      p += 1 + 64 * (wide ? 2 : 1);
    }
  }
  return null;
}

function dimensions(b, format, items) {
  const at = (name) => items.find((i) => i.name.startsWith(name))?.at;
  if (format === 'png') return [new DataView(b.buffer, b.byteOffset).getUint32(16), new DataView(b.buffer, b.byteOffset).getUint32(20)];
  if (format === 'jpeg') {
    const f = at('Frame header');
    return f === undefined ? null : [(b[f + 7] << 8) | b[f + 8], (b[f + 5] << 8) | b[f + 6]];
  }
  if (format === 'webp') {
    let d = at('Extended header (VP8X)');
    if (d !== undefined) return [1 + (b[d + 12] | (b[d + 13] << 8) | (b[d + 14] << 16)), 1 + (b[d + 15] | (b[d + 16] << 8) | (b[d + 17] << 16))];
    d = at('Image data (VP8L');
    if (d !== undefined) {
      const bits = b[d + 9] | (b[d + 10] << 8) | (b[d + 11] << 16) | (b[d + 12] << 24);
      return [1 + (bits & 0x3fff), 1 + ((bits >>> 14) & 0x3fff)];
    }
    d = at('Image data (VP8,');
    if (d !== undefined) return [((b[d + 15] << 8) | b[d + 14]) & 0x3fff, ((b[d + 17] << 8) | b[d + 16]) & 0x3fff];
  }
  return null;
}

/** Rows of { name, value, reading }. */
function hints(b, format, items) {
  const out = [];
  const size = dimensions(b, format, items);
  if (size) {
    const [w, h] = size;
    const mp = (w * h) / 1e6;
    const grid = [64, 32, 16, 8].find((g) => w % g === 0 && h % g === 0);
    const onGrid = grid >= 32 && mp <= 4.2;
    out.push({
      name: 'Image size',
      value: `${w} × ${h}, ${mp.toFixed(2)} megapixels${grid ? `, both sides multiples of ${grid}` : ''}`,
      reading: onGrid
        ? 'The way generators size their output: a few megapixels at most, on a grid of 32 or more pixels. A photo resized to a round size can match too.'
        : 'Not on a generator\'s grid. Cameras, screenshots and resized photos look like this.',
      flag: onGrid,
    });
  }
  if (format === 'jpeg') {
    const t = lumaTable(b, items);
    const q = t && [...Array(100).keys()].map((k) => k + 1).find((k) => ijgTable(k).every((v, i) => v === t[i]));
    if (q) {
      out.push({
        name: 'JPEG encoder',
        value: `Standard libjpeg tables at quality ${q}`,
        reading: 'Saved by a general-purpose encoder: a server, an image editor, a browser or a messaging app. Many phone cameras use tables of their own, so this file was saved again after it left the camera, or never came from one.',
        flag: true,
      });
    } else if (t) {
      const scale = t.reduce((s, v, i) => s + v / IJG_LUMA[i], 0) / 64 * 100;
      const est = Math.round(scale <= 100 ? (200 - scale) / 2 : 5000 / scale);
      out.push({
        name: 'JPEG encoder',
        value: `Custom tables, about quality ${Math.min(100, Math.max(1, est))}`,
        reading: 'The encoder used tables of its own, as many cameras and some apps do.',
        flag: false,
      });
    }
  } else if (format === 'png' || format === 'webp') {
    out.push({
      name: 'File format',
      value: format.toUpperCase(),
      reading: 'Phone cameras save JPEG or HEIC. Generators, editors and screenshots save PNG and WebP.',
      flag: true,
    });
  }
  const camera = items.flatMap((i) => (i.how === 'exif' ? i.details ?? [] : [])).filter(([k]) => /^(Make|Model)/.test(k)).map(([, v]) => v);
  out.push({
    name: 'Camera in EXIF',
    value: camera.length ? camera.join(' ') : 'None',
    reading: camera.length
      ? 'Written by a camera, or copied over by an editor. Easy to fake, but generators do not write it.'
      : 'Generators write none. Neither do messaging apps, screenshots or a cleaned file, so on its own it means little.',
    flag: !camera.length,
  });
  return out;
}

// ---------- shared readers ----------

const EXIF_TAGS = {
  0x010e: 'Description', 0x010f: 'Make', 0x0110: 'Model', 0x0131: 'Software', 0x013b: 'Artist',
  0x8298: 'Copyright', 0x9003: 'Taken', 0x9286: 'User comment', 0xa420: 'Unique ID',
};

// Text fields from a TIFF/EXIF block: IFD0 and the EXIF sub-IFD.
function exif(t) {
  const out = [];
  try {
    const le = t[0] === 0x49;
    const dv = new DataView(t.buffer, t.byteOffset, t.byteLength);
    const u16 = (o) => dv.getUint16(o, le);
    const u32 = (o) => dv.getUint32(o, le);
    out.push(['Byte order', le ? 'II (little-endian)' : 'MM (big-endian)']);
    const read = (off, depth) => {
      if (depth > 2) return;
      const n = u16(off);
      for (let k = 0; k < n; k++) {
        const e = off + 2 + k * 12;
        const tag = u16(e), type = u16(e + 2), count = u32(e + 4);
        if (tag === 0x8769) { read(u32(e + 8), depth + 1); continue; }
        if (tag === 0x8825) { out.push(['GPS (tag 0x8825)', 'Location data present']); continue; }
        if (!EXIF_TAGS[tag] || (type !== 2 && type !== 7)) continue;
        const at = count > 4 ? u32(e + 8) : e + 8;
        let v = t.subarray(at, at + count);
        if (tag === 0x9286) v = v.subarray(8); // charset prefix
        const s = latin1.decode(v).replace(/\0+/g, ' ').trim();
        if (s) out.push([`${EXIF_TAGS[tag]} (tag 0x${tag.toString(16).padStart(4, '0')})`, clip(s)]);
      }
    };
    read(u32(4), 0);
  } catch {
    out.push(['Note', 'Could not read every field']);
  }
  return out;
}

// The EXIF Orientation tag (0x0112) in IFD0: 1 is upright, 2 to 8 are flips and turns.
// `alone` is true when it is the block's only tag, as in the block the cleaner writes.
function orientation(t) {
  try {
    const le = t[0] === 0x49;
    const dv = new DataView(t.buffer, t.byteOffset, t.byteLength);
    const ifd = dv.getUint32(4, le);
    const n = dv.getUint16(ifd, le);
    for (let k = 0; k < n; k++) {
      const e = ifd + 2 + k * 12;
      if (dv.getUint16(e, le) === 0x0112) {
        const v = dv.getUint16(e + 8, le);
        return { turn: v >= 1 && v <= 8 ? v : 1, alone: n === 1 && dv.getUint32(ifd + 14, le) === 0 };
      }
    }
  } catch {}
  return { turn: 1, alone: false };
}

const XMP_FIELDS = [
  ['AI source type (Iptc4xmpExt:DigitalSourceType)', 'DigitalSourceType'],
  ['Creator tool (xmp:CreatorTool)', 'CreatorTool'],
  ['Credit (photoshop:Credit)', 'Credit'],
  ['Description (dc:description)', 'description'],
  ['Provenance (dcterms:provenance)', 'provenance'],
];

function xmp(x) {
  const out = [];
  for (const [label, field] of XMP_FIELDS) {
    const m = x.match(new RegExp(`:${field}[^"<>]*?(?:="|>)\\s*(?:<[^>]+>\\s*)*([^"<]+)`));
    if (m && m[1].trim()) out.push([label, clip(m[1].trim())]);
  }
  return out;
}

function find(b, seq) {
  outer: for (let i = 0; i <= b.length - seq.length; i++) {
    for (let k = 0; k < seq.length; k++) if (b[i + k] !== seq[k]) continue outer;
    return i;
  }
  return -1;
}

function concat(parts) {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}
