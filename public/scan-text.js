// Finds characters and artifacts in text that AI chat apps leave behind, and builds a clean copy.
// Each rule finds spans. Rules run in order and an earlier rule wins where spans overlap,
// so a whole citation marker is claimed before its private-use characters are.

export const CATEGORIES = {
  hidden: {
    title: 'Hidden characters',
    note: 'Invisible code points. They can carry a watermark, a tracking code or a hidden prompt.',
    clean: true,
  },
  artifact: {
    title: 'Chat-app artifacts',
    note: 'Citation markers and referral parameters that chat apps leave in copied text.',
    clean: true,
  },
  spacing: {
    title: 'Unusual spaces',
    note: 'Non-breaking and fixed-width spaces that look like a normal space.',
    clean: true,
  },
  homoglyph: {
    title: 'Look-alike letters',
    note: 'Cyrillic or Greek letters inside Latin words, such as a Cyrillic "\u0430" in "p\u0430ypal".',
    clean: true,
  },
  style: {
    title: 'Typography',
    note: 'Curly quotes, long dashes and the ellipsis character. Common in model output, and in plenty of human writing. Off by default.',
    clean: false,
  },
};

const NAMES = {
  0x200b: 'ZWSP', 0x200c: 'ZWNJ', 0x200d: 'ZWJ', 0x2060: 'WJ', 0xfeff: 'BOM', 0x180e: 'MVS',
  0x00ad: 'SHY', 0x00a0: 'NBSP', 0x202f: 'NNBSP', 0x2009: 'THIN', 0x200a: 'HAIR', 0x2002: 'ENSP',
  0x2003: 'EMSP', 0x2007: 'FIGSP', 0x2008: 'PUNCSP', 0x3000: 'IDSP', 0x205f: 'MMSP',
  0x200e: 'LRM', 0x200f: 'RLM', 0x061c: 'ALM', 0x202a: 'LRE', 0x202b: 'RLE', 0x202c: 'PDF',
  0x202d: 'LRO', 0x202e: 'RLO', 0x2066: 'LRI', 0x2067: 'RLI', 0x2068: 'FSI', 0x2069: 'PDI',
  0x2028: 'LSEP', 0x2029: 'PSEP',
};

/** Short label for a code point, such as "ZWSP" or "U+E0041". */
export const label = (cp) => NAMES[cp] ?? 'U+' + cp.toString(16).toUpperCase().padStart(4, '0');

/** "U+200B" for any code point. */
export const codePoint = (cp) => 'U+' + cp.toString(16).toUpperCase().padStart(4, '0');

// The code point just before index i (whole surrogate pair included).
const before = (text, i) => [...text.slice(Math.max(0, i - 2), i)].pop() ?? '';
const after = (text, i) => [...text.slice(i, i + 2)][0] ?? '';
const PICTO = /\p{Extended_Pictographic}/u;

// Bytes hidden in variation selectors ("emoji smuggling"): U+FE00-FE0F are 0-15, U+E0100-E01EF are 16-255.
function decodeSelectors(run) {
  const bytes = [...run].map((c) => {
    const cp = c.codePointAt(0);
    return cp >= 0xe0100 ? cp - 0xe0100 + 16 : cp - 0xfe00;
  });
  if (bytes.length < 2) return undefined;
  const s = new TextDecoder().decode(new Uint8Array(bytes));
  return /[\p{L}\p{N}]/u.test(s) && !s.includes('\uFFFD') ? s : undefined;
}

// ASCII hidden in Unicode tag characters: U+E0041 is "A", and so on.
const decodeTags = (run) =>
  [...run].map((c) => String.fromCodePoint(c.codePointAt(0) - 0xe0000)).join('').replace(/[^\x20-\x7e\n]/g, '') || undefined;

const CONFUSABLES = {
  \u0430: 'a', \u0435: 'e', \u043E: 'o', \u0440: 'p', \u0441: 'c', \u0443: 'y', \u0445: 'x', \u0456: 'i', \u0458: 'j', \u0455: 's', \u0501: 'd', \u051B: 'q', \u051D: 'w', \u04BB: 'h', \u04CF: 'l',
  \u0410: 'A', \u0412: 'B', \u0415: 'E', \u041A: 'K', \u041C: 'M', \u041D: 'H', \u041E: 'O', \u0420: 'P', \u0421: 'C', \u0422: 'T', \u0425: 'X', \u0406: 'I', \u0408: 'J', \u0405: 'S',
  \u03BF: 'o', \u03BD: 'v', \u03B1: 'a', \u03B9: 'i', \u03BA: 'k', \u03C1: 'p', \u03C4: 't', \u03C5: 'u', \u03C7: 'x',
  \u0391: 'A', \u0392: 'B', \u0395: 'E', \u0396: 'Z', \u0397: 'H', \u0399: 'I', \u039A: 'K', \u039C: 'M', \u039D: 'N', \u039F: 'O', \u03A1: 'P', \u03A4: 'T', \u03A5: 'Y', \u03A7: 'X',
};

function findHomoglyphs(text) {
  const out = [];
  for (const word of text.matchAll(/\p{L}+/gu)) {
    const w = word[0];
    if (!/\p{Script=Latin}/u.test(w) || !/[\p{Script=Cyrillic}\p{Script=Greek}]/u.test(w)) continue;
    let at = word.index;
    for (const ch of w) {
      if (CONFUSABLES[ch]) out.push({ start: at, end: at + ch.length, text: ch, fix: CONFUSABLES[ch] });
      at += ch.length;
    }
  }
  return out;
}

const AI_REF = /chatgpt|openai|perplexity|copilot|gemini|claude|grok/i;

/** The rules, in the order they run. `how` says what each one looks for and why it matters. */
export const RULES = [
  {
    cat: 'artifact', name: 'ChatGPT citation marker',
    how: 'ChatGPT wraps each web citation in private-use characters: U+E200, a word such as "cite", U+E202, source ids such as "turn0search1", then U+E201. Most apps draw nothing for them, so they travel unseen when the answer is copied. Matched by that exact bracket sequence, with one space before it.',
    re: / ?\uE200[a-z_]+\uE202[^\uE201]*\uE201/g, fix: '',
  },
  {
    cat: 'artifact', name: 'Copied citation marker',
    how: 'The same citation after an app dropped the private-use brackets: the word "cite", "filecite" or "navlist" followed by ids of the form turn<number><kind><number>, left as plain text.',
    re: /\s?(?:cite|filecite|navlist)(?:turn\d+[a-z]+\d+)+/g, fix: '',
  },
  {
    cat: 'artifact', name: 'Assistant file citation',
    how: 'OpenAI\'s file search cites a source as \u30104:0\u2020source\u3011: the lenticular brackets U+3010 and U+3011 around a message:index pair, a dagger U+2020 and a file name.',
    re: /\u3010\d+(?::\d+)?\u2020[^\u3011]*\u3011/g, fix: '',
  },
  {
    cat: 'artifact', name: 'ChatGPT content reference',
    how: 'ChatGPT\'s internal reference ":contentReference[oaicite:0]{index=0}", or its short form "[oaicite:0]", which leaks into copied and exported text.',
    re: /:contentReference\[oaicite:\d+\]\{index=\d+\}|\[oaicite:\d+\]/g, fix: '',
  },
  {
    cat: 'artifact',
    name: 'AI referral parameter in a link',
    how: 'Links in answers from ChatGPT, Perplexity, Copilot and others carry a utm_source parameter that names the app, such as utm_source=chatgpt.com, so the site can count visits from it. Only the parameter is removed; the rest of the link is kept.',
    // The value never ends on punctuation, so a full stop after the link stays.
    re: /[?&]utm_source=[^&\s)\]"'<>#]*[^&\s)\]"'<>#.,;:!?]&?/gi,
    skip: (m) => !AI_REF.test(m),
    // "?utm_source=x&a=1" keeps the "?", "&utm_source=x" at the end leaves nothing.
    fix: (m) => (m.endsWith('&') ? m[0] : ''),
  },
  {
    cat: 'hidden', name: 'Unicode tag characters', short: 'TAG',
    how: 'U+E0000 to U+E007F copy ASCII (U+E0041 is "A") and are drawn as nothing, so a whole sentence can sit inside text unseen. This is a known route for hidden prompt injection. Decoded by subtracting 0xE0000 from each code point. Skipped after a black flag U+1F3F4, where tags spell the England, Scotland and Wales flags.',
    re: /[\u{E0000}-\u{E007F}]+/gu,
    // Subdivision flags (England, Scotland, Wales) are a black flag followed by tags.
    skip: (m, text, i) => before(text, i) === '\u{1F3F4}',
    fix: '', decode: decodeTags,
  },
  {
    cat: 'hidden', name: 'Variation selectors', short: 'VS',
    how: 'U+FE00 to U+FE0F and U+E0100 to U+E01EF only change how the character before them is drawn, and show nothing themselves. Each one can stand for a byte (U+FE00 to U+FE0F are 0 to 15, U+E0100 to U+E01EF are 16 to 255), so a run of them after one emoji can carry any message ("emoji smuggling"). Decoded as UTF-8 bytes. One U+FE0E or U+FE0F after an emoji or a keycap digit is normal and skipped.',
    re: /[\uFE00-\uFE0F\u{E0100}-\u{E01EF}]+/gu,
    // One text/emoji selector after an emoji or a keycap digit is normal.
    skip: (m, text, i) => /^[\uFE0E\uFE0F]$/.test(m) && (PICTO.test(before(text, i)) || /^[#*0-9]$/.test(before(text, i))),
    fix: '', decode: decodeSelectors,
  },
  {
    cat: 'hidden', name: 'Zero-width characters',
    how: 'Zero-width space U+200B, non-joiner U+200C, joiner U+200D, word joiner U+2060, byte order mark U+FEFF and Mongolian vowel separator U+180E take no width. A pattern of them survives copy and paste, which makes them the classic way to fingerprint or watermark text. Skipped where they do real work: a joiner between two emoji, and joiners after letters of scripts such as Arabic and Devanagari.',
    re: /[\u200B\u200C\u200D\u2060\uFEFF\u180E]/g,
    skip: (m, text, i) => {
      const prev = before(text, i);
      // ZWJ joins emoji (family, profession) and both joiners shape Arabic and Indic scripts.
      if (m === '\u200D' && PICTO.test(prev) && PICTO.test(after(text, i + 1))) return true;
      return (m === '\u200C' || m === '\u200D') && /\p{L}|\p{M}/u.test(prev) && !/\p{Script=Latin}/u.test(prev);
    },
    fix: '',
  },
  {
    cat: 'hidden', name: 'Direction controls',
    how: 'The marks LRM U+200E, RLM U+200F and ALM U+061C, and the embedding, override and isolate controls U+202A to U+202E and U+2066 to U+2069, change the order text is shown in. In left-to-right text they show nothing, and an override can make text display differently from what it contains (the "Trojan Source" attack, CVE-2021-42574).',
    re: /[\u200E\u200F\u202A-\u202E\u2066-\u2069\u061C]/g, fix: '',
  },
  {
    cat: 'hidden', name: 'Soft hyphen',
    how: 'U+00AD appears only where a line breaks. Inside a line it is invisible but still in the text, so it can break searches and carry a pattern.',
    re: /\u00AD/g, fix: '',
  },
  {
    cat: 'hidden', name: 'Control characters',
    how: 'C0 and C1 control codes (U+0000 to U+001F and U+007F to U+009F) other than tab, line feed and carriage return. Nothing in written text needs them.',
    re: /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]/g, fix: '',
  },
  {
    cat: 'hidden', name: 'Private-use characters', short: 'PUA',
    how: 'U+E000 to U+F8FF and planes 15 and 16 have no standard meaning; each app gives them its own. ChatGPT uses them for citation brackets. Anything left after the markers above is removed is hidden data.',
    re: /[\uE000-\uF8FF\u{F0000}-\u{FFFFD}\u{100000}-\u{10FFFD}]+/gu, fix: '',
  },
  {
    cat: 'spacing', name: 'Non-standard spaces',
    how: 'No-break space U+00A0, the fixed-width spaces U+2000 to U+200A, narrow no-break space U+202F, medium mathematical space U+205F and ideographic space U+3000 look like a space but are different characters. Replaced with a plain space.',
    re: /[\u00A0\u2000-\u200A\u202F\u205F\u3000]/g, fix: ' ',
  },
  {
    cat: 'spacing', name: 'Unicode line separators',
    how: 'U+2028 and U+2029 are line breaks in some programs and nothing in others. Replaced with a line feed.',
    re: /[\u2028\u2029]/g, fix: '\n',
  },
  {
    cat: 'homoglyph', name: 'Look-alike letter',
    how: 'A word written mostly in Latin letters that also holds a Cyrillic or Greek letter drawn the same way, such as Cyrillic \u0430 U+0430 for Latin a. Found by checking the script of every letter in each word; each look-alike is replaced with the Latin letter it imitates.',
    find: findHomoglyphs,
  },
  {
    cat: 'style', name: 'Long dash',
    how: 'Em dash U+2014 and en dash U+2013. Models use the em dash far more than most writers, but plenty of people use it too, so it proves nothing on its own.',
    re: / ?\u2014 ?|\u2013/g, fix: (m) => (m.includes('\u2014') ? ' - ' : '-'),
  },
  {
    cat: 'style', name: 'Curly quote',
    how: 'Typographic quotes U+2018 to U+201F. Word processors produce them too.',
    re: /[\u2018\u2019\u201A\u201B\u201C\u201D\u201E\u201F]/g, fix: (m) => (/[\u2018-\u201B]/.test(m) ? "'" : '"'),
  },
  {
    cat: 'style', name: 'Ellipsis character',
    how: 'U+2026, three dots as one character.',
    re: /\u2026/g, fix: '...',
  },
];

function spans(rule, text) {
  if (rule.find) return rule.find(text);
  const out = [];
  for (const m of text.matchAll(rule.re)) {
    if (rule.skip?.(m[0], text, m.index)) continue;
    out.push({ start: m.index, end: m.index + m[0].length, text: m[0] });
  }
  return out;
}

/** Every finding in the text, in reading order. */
export function scanText(text) {
  const claimed = new Uint8Array(text.length);
  const findings = [];
  for (const rule of RULES) {
    for (const s of spans(rule, text)) {
      if (claimed.subarray(s.start, s.end).some(Boolean)) continue;
      claimed.fill(1, s.start, s.end);
      findings.push({
        start: s.start,
        end: s.end,
        text: s.text,
        cat: rule.cat,
        name: rule.name,
        short: rule.short,
        fix: typeof rule.fix === 'function' ? rule.fix(s.text) : (rule.fix ?? s.fix),
        decoded: rule.decode?.(s.text),
      });
    }
  }
  return findings.sort((a, b) => a.start - b.start);
}

/** The text with every finding in an enabled category replaced by its fix. */
export function cleanText(text, findings, categories) {
  let out = '';
  let at = 0;
  for (const f of findings) {
    if (!categories.has(f.cat)) continue;
    out += text.slice(at, f.start) + f.fix;
    at = f.end;
  }
  return out + text.slice(at);
}
