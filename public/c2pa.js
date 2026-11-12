// Reads C2PA content credentials: the JUMBF box tree (ISO/IEC 19566-5), its CBOR content
// (RFC 8949) and the COSE_Sign1 signature (RFC 9052). It reports what the manifest says and who
// signed it. It does not verify the signature or the hashes.

const utf8 = new TextDecoder();
const latin1 = new TextDecoder('latin1');
const u32 = (b, at) => ((b[at] << 24) | (b[at + 1] << 16) | (b[at + 2] << 8) | b[at + 3]) >>> 0;
const fourcc = (b, at) => latin1.decode(b.subarray(at, at + 4));
const hex = (b) => [...b].map((x) => x.toString(16).padStart(2, '0')).join('');

function concat(parts) {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
}

/** The boxes in b[from, to): 4-byte size, 4-letter type, data. Size 1 means a 64-bit size follows. */
export function boxes(b, from = 0, to = b.length) {
  const out = [];
  let i = from;
  while (i + 8 <= to) {
    let size = u32(b, i);
    let head = 8;
    if (size === 1) { size = u32(b, i + 8) * 2 ** 32 + u32(b, i + 12); head = 16; }
    else if (size === 0) size = to - i;
    if (size < head || i + size > to) break;
    out.push({ type: fourcc(b, i + 4), at: i, data: b.subarray(i + head, i + size) });
    i += size;
  }
  return out;
}

/**
 * Joins JPEG APP11 packets back into JUMBF boxes. Each payload starts "JP", a 2-byte box
 * instance and a 4-byte sequence number. Every packet of one instance repeats the box header,
 * so later packets contribute only what follows it.
 */
export function joinJpegPackets(payloads) {
  const byInstance = new Map();
  for (const d of payloads) {
    if (d[0] !== 0x4a || d[1] !== 0x50 || d.length < 16) continue;
    const instance = (d[2] << 8) | d[3];
    const parts = byInstance.get(instance) ?? [];
    parts.push({ seq: u32(d, 4), box: d.subarray(8) });
    byInstance.set(instance, parts);
  }
  return [...byInstance.values()].map((parts) => {
    parts.sort((a, b) => a.seq - b.seq);
    const head = u32(parts[0].box, 0) === 1 ? 16 : 8;
    return concat([parts[0].box, ...parts.slice(1).map((p) => p.box.subarray(head))]);
  });
}

// A JUMBF superbox ("jumb") starts with a description box ("jumd"): a 16-byte type UUID whose
// first 4 bytes name the C2PA kind (c2pa, c2ma, c2as, c2cl, c2cs, cbor, json...), a toggles byte,
// and, when toggle 0x02 is set, a null-terminated UTF-8 label.
function superbox(box) {
  const kids = boxes(box.data);
  const node = { type: 'jumb', label: '', kind: '', size: box.data.length, children: [] };
  let rest = kids;
  if (kids[0]?.type === 'jumd') {
    const d = kids[0].data;
    node.kind = fourcc(d, 0).replace(/[^\x20-\x7e]/g, '');
    if (d[16] & 0x02) {
      const z = d.indexOf(0, 17);
      node.label = utf8.decode(d.subarray(17, z < 0 ? d.length : z));
    }
    rest = kids.slice(1);
  }
  node.children = rest.map((k) => (k.type === 'jumb' ? superbox(k) : { type: k.type, size: k.data.length, data: k.data }));
  return node;
}

function treeText(node, depth = 0) {
  const pad = '  '.repeat(depth);
  if (node.type !== 'jumb') return `${pad}${node.type} box, ${node.size} bytes`;
  const head = `${pad}jumb "${node.label}"${node.kind ? ` (type ${node.kind})` : ''}, ${node.size} bytes`;
  return [head, ...node.children.map((c) => treeText(c, depth + 1))].join('\n');
}

const child = (node, re) => node?.children?.find((c) => c.type === 'jumb' && re.test(c.label));

function content(node) {
  const box = node?.children?.find((c) => c.type === 'cbor' || c.type === 'json');
  if (!box) return undefined;
  try {
    return box.type === 'cbor' ? cbor(box.data) : JSON.parse(utf8.decode(box.data));
  } catch {
    return undefined;
  }
}

/** Reads a JUMBF manifest store. Returns null when the bytes hold no JUMBF superbox. */
export function readC2pa(bytes) {
  const top = boxes(bytes).find((x) => x.type === 'jumb');
  if (!top) return null;
  const store = superbox(top);
  return {
    label: store.label,
    tree: treeText(store),
    manifests: store.children.filter((c) => c.type === 'jumb').map(manifest),
  };
}

function manifest(m) {
  const claimNode = child(m, /^c2pa\.claim/);
  const claim = content(claimNode) ?? {};
  const assertions = (child(m, /^c2pa\.assertions$/)?.children ?? [])
    .filter((c) => c.type === 'jumb')
    .map((a) => ({ label: a.label, value: /thumbnail/.test(a.label) ? undefined : content(a) }));
  const pick = (re) => assertions.filter((a) => re.test(a.label));
  return {
    label: m.label,
    claim: claimNode?.label,
    generator: generator(claim),
    title: claim['dc:title'],
    format: claim['dc:format'],
    assertions,
    actions: pick(/^c2pa\.actions/).flatMap((a) => a.value?.actions ?? []).map(action),
    ingredients: pick(/^c2pa\.ingredient/).map((a) => ({ title: a.value?.['dc:title'], relationship: a.value?.relationship })),
    softBindings: pick(/^c2pa\.soft-binding/).map((a) => a.value?.alg ?? 'unnamed'),
    signature: signature(content(child(m, /^c2pa\.signature$/))),
  };
}

function generator(claim) {
  const info = claim.claim_generator_info;
  const list = Array.isArray(info) ? info : info ? [info] : [];
  const named = list.map((g) => [g.name, g.version].filter(Boolean).join(' ')).filter(Boolean);
  return named.join(', ') || (typeof claim.claim_generator === 'string' ? claim.claim_generator : undefined);
}

function action(a) {
  const agent = typeof a.softwareAgent === 'string' ? a.softwareAgent : a.softwareAgent?.name;
  const source = typeof a.digitalSourceType === 'string' ? a.digitalSourceType.split('/').pop() : undefined;
  return { action: a.action, source, agent, when: typeof a.when === 'string' ? a.when : undefined };
}

// COSE algorithm ids (IANA registry).
const ALGS = { '-7': 'ES256', '-35': 'ES384', '-36': 'ES512', '-37': 'PS256', '-38': 'PS384', '-39': 'PS512', '-8': 'EdDSA' };

// COSE_Sign1 is [protected header (CBOR in a byte string), unprotected header, payload, signature].
// Header 1 is the algorithm, header 33 (x5chain) the signer's certificate chain, leaf first.
function signature(cose) {
  const arr = cose?.tag === 18 ? cose.value : cose;
  if (!Array.isArray(arr)) return null;
  const [protectedBytes, unprotected = {}] = arr;
  let prot = {};
  try {
    if (protectedBytes?.length) prot = cbor(protectedBytes);
  } catch {
    // leave the protected header empty
  }
  const x5 = prot['33'] ?? unprotected['33'];
  const certs = x5 instanceof Uint8Array ? [x5] : Array.isArray(x5) ? x5 : [];
  return {
    alg: ALGS[prot['1']] ?? (prot['1'] === undefined ? undefined : String(prot['1'])),
    chain: certs.map(certificate).filter(Boolean),
    timestamp: Boolean(unprotected.sigTst ?? unprotected.sigTst2),
  };
}

// ---------- X.509 (just enough DER to read names and dates) ----------

function der(b, at) {
  let len = b[at + 1];
  let p = at + 2;
  if (len & 0x80) {
    const n = len & 0x7f;
    len = 0;
    for (let k = 0; k < n; k++) len = len * 256 + b[p++];
  }
  return { tag: b[at], start: p, end: p + len };
}

function derKids(b, node) {
  const out = [];
  for (let p = node.start; p < node.end;) {
    const k = der(b, p);
    out.push(k);
    p = k.end;
  }
  return out;
}

const NAME_OIDS = { '550403': 'CN', '55040a': 'O', '55040b': 'OU', '550406': 'C' };

function derName(b, node) {
  const out = {};
  for (const set of derKids(b, node)) {
    for (const pair of derKids(b, set)) {
      const [oid, value] = derKids(b, pair);
      const key = NAME_OIDS[hex(b.subarray(oid.start, oid.end))];
      if (key && value) out[key] = utf8.decode(b.subarray(value.start, value.end));
    }
  }
  return out;
}

function derTime(b, node) {
  const s = latin1.decode(b.subarray(node.start, node.end));
  const full = node.tag === 0x17 ? (Number(s.slice(0, 2)) < 50 ? '20' : '19') + s : s; // UTCTime has a 2-digit year
  return `${full.slice(0, 4)}-${full.slice(4, 6)}-${full.slice(6, 8)}`;
}

// Certificate = SEQUENCE { tbsCertificate SEQUENCE { [0] version, serial, signature algorithm,
// issuer, validity { notBefore, notAfter }, subject, ... }, ... }
function certificate(c) {
  try {
    const tbs = derKids(c, der(c, 0))[0];
    let f = derKids(c, tbs);
    if (f[0].tag === 0xa0) f = f.slice(1);
    const [, , issuer, validity, subject] = f;
    const [from, to] = derKids(c, validity);
    return { subject: derName(c, subject), issuer: derName(c, issuer), from: derTime(c, from), to: derTime(c, to) };
  } catch {
    return null;
  }
}

// ---------- CBOR ----------

const BREAK = Symbol('break');

function half(h) {
  const e = (h >> 10) & 31, f = h & 1023, s = h >> 15 ? -1 : 1;
  return s * (e === 0 ? f * 2 ** -24 : e === 31 ? (f ? NaN : Infinity) : (1 + f / 1024) * 2 ** (e - 15));
}

/** Decodes one CBOR item. Byte strings come back as Uint8Array, map keys as strings, tag 18 as {tag, value}. */
export function cbor(b) {
  const dv = new DataView(b.buffer, b.byteOffset, b.byteLength);
  let i = 0;
  const length = (info) => {
    if (info < 24) return info;
    if (info === 24) return b[i++];
    if (info === 25) { i += 2; return dv.getUint16(i - 2); }
    if (info === 26) { i += 4; return dv.getUint32(i - 4); }
    if (info === 27) { i += 8; return Number(dv.getBigUint64(i - 8)); }
    if (info === 31) return -1;
    throw new Error('Bad CBOR length');
  };
  const item = () => {
    if (i >= b.length) throw new Error('CBOR ended early');
    const first = b[i++];
    const major = first >> 5, info = first & 31;
    if (major === 7) {
      if (info === 20) return false;
      if (info === 21) return true;
      if (info === 22) return null;
      if (info === 23) return undefined;
      if (info === 25) { i += 2; return half(dv.getUint16(i - 2)); }
      if (info === 26) { i += 4; return dv.getFloat32(i - 4); }
      if (info === 27) { i += 8; return dv.getFloat64(i - 8); }
      if (info === 31) return BREAK;
      return info < 24 ? info : b[i++];
    }
    const n = length(info);
    if (major === 0) return n;
    if (major === 1) return -1 - n;
    if (major === 2 || major === 3) {
      let v;
      if (n < 0) {
        const parts = [];
        for (let p = item(); p !== BREAK; p = item()) parts.push(major === 3 ? new TextEncoder().encode(p) : p);
        v = concat(parts);
      } else {
        v = b.subarray(i, i + n);
        i += n;
      }
      return major === 3 ? utf8.decode(v) : v;
    }
    if (major === 4) {
      const a = [];
      if (n < 0) for (let v = item(); v !== BREAK; v = item()) a.push(v);
      else for (let k = 0; k < n; k++) a.push(item());
      return a;
    }
    if (major === 5) {
      const m = {};
      for (let k = 0; n < 0 || k < n; k++) {
        const key = item();
        if (key === BREAK) break;
        m[String(key)] = item();
      }
      return m;
    }
    const value = item(); // major 6: a tag; only COSE_Sign1 (18) matters here
    return n === 18 ? { tag: 18, value } : value;
  };
  return item();
}

// ---------- summary ----------

const who = (n = {}) => [n.O, n.CN && n.CN !== n.O ? n.CN : null, n.C].filter(Boolean).join(', ') || 'unnamed';

/** Rows of [label, value] describing a manifest store, active manifest first. */
export function describe(store) {
  const all = store.manifests;
  const m = all.at(-1);
  const rows = [['Manifest store', `"${store.label}", ${all.length} manifest${all.length === 1 ? '' : 's'}${m ? `; active: ${m.label}` : ''}`]];
  if (!m) return rows;
  if (m.generator) rows.push(['Claim generator', m.generator]);
  if (m.claim) rows.push(['Claim box', m.claim]);
  if (m.title || m.format) rows.push(['Asset', [m.title, m.format].filter(Boolean).join(' \u00B7 ')]);
  for (const a of m.actions) {
    rows.push(['Action', [a.action, a.source && `digitalSourceType: ${a.source}`, a.agent && `software: ${a.agent}`, a.when].filter(Boolean).join(' \u00B7 ')]);
  }
  if (m.actions.some((a) => a.action?.startsWith('c2pa.watermarked'))) {
    rows.push(['Invisible watermark', 'A c2pa.watermarked action: the generator says it put an invisible watermark in the pixels. Removing this metadata does not remove that watermark.']);
  }
  for (const g of m.ingredients) rows.push(['Ingredient', [g.title, g.relationship].filter(Boolean).join(' \u00B7 ') || 'unnamed']);
  for (const s of m.softBindings) {
    rows.push(['Soft binding', `${s}: the manifest names an invisible watermark that can find these credentials again after the metadata is removed`]);
  }
  const sig = m.signature;
  const leaf = sig?.chain[0];
  if (leaf) {
    rows.push(['Signed by', who(leaf.subject)]);
    rows.push(['Certificate issued by', who(leaf.issuer)]);
    rows.push(['Certificate valid', `${leaf.from} to ${leaf.to}`]);
    if (sig.chain.length > 1) rows.push(['Certificate chain', sig.chain.map((c) => who(c.subject)).join(' > ')]);
  }
  if (sig?.alg) rows.push(['Signature algorithm', sig.alg + (sig.timestamp ? ', with a trusted timestamp' : '')]);
  rows.push(['Assertions', m.assertions.map((a) => a.label).join(', ') || 'none']);
  for (const e of all.slice(0, -1)) rows.push(['Earlier manifest', [e.label, e.generator].filter(Boolean).join(' \u00B7 ')]);
  rows.push(['Signature check', 'Not verified here. Verify with c2patool or contentcredentials.org/verify.']);
  return rows;
}
