// The command table (cli/command_table.cpp) read as cli/registry.cpp reads it, for the WebAssembly face: the generator of js/commands.mjs and the coverage test
// (test/commands.test.mjs) both use `readTable`. `node tools/command_table.mjs [--check]` writes js/commands.mjs, or with --check fails when it is not what the table gives.
//
//   readTable(path) -> [{ verb, noun, positional, optionalPositional, options: [{ name, required, multi, flag }] }]   (option names with underscores)
import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { ARRAY_OPTIONS } from '../js/arrays.mjs';

const here = dirname(fileURLToPath(import.meta.url));
export const TABLE = join(here, '..', '..', '..', 'cli', 'command_table.cpp');
export const COMMANDS_FILE = join(here, '..', 'js', 'commands.mjs');
export const TYPES_FILE = join(here, '..', 'js', 'commands.d.ts');

export function readTable(path = TABLE) {
  const docs = new Map();
  const commands = [];
  let cur = null;
  for (const m of readFileSync(path, 'utf8').matchAll(/R"TBL\(([\s\S]*?)\)TBL"/g)) {
    for (const line of m[1].split('\n')) {
      if (!line.trim() || line.startsWith('#')) continue;
      if (line.startsWith('--')) {
        const f = line.split('\t');
        docs.set(f[0], f[1] ?? '');
      } else if (line.startsWith('@')) {
        const w = line.slice(1).trim().split(/\s+/);
        cur = { verb: w[0], noun: w[1] ?? '', positional: '', optionalPositional: false, opts: [] };
        commands.push(cur);
      } else if (cur && line.includes(':')) {
        const i = line.indexOf(':');
        const key = line.slice(0, i), val = line.slice(i + 1).trim();
        if (key === 'pos') { cur.optionalPositional = val.endsWith('?'); cur.positional = val.replace(/\?$/, '').toLowerCase(); }
        else if (key === 'opts') cur.opts = val.split(/\s+/).filter(Boolean);
      }
    }
  }
  return commands.map((c) => {
    const path = `${c.verb} ${c.noun}`.trim();
    const options = c.opts.map((w0) => {
      let w = w0;
      let required = w.endsWith('*');
      w = w.replace(/\*+$/, '');
      const multi = w.endsWith('+');
      w = w.replace(/\+$/, '').replace(/\*+$/, '');
      required = required || w0.includes('*');
      w = w.split('=')[0];
      const value = docs.has(`${w}@${path}`) ? docs.get(`${w}@${path}`) : docs.get(w);
      return { name: w.slice(2).replaceAll('-', '_'), required, multi, flag: value === '' };
    });
    return { verb: c.verb, noun: c.noun, positional: c.positional, optionalPositional: c.optionalPositional, options };
  });
}

/** the generated module: one entry a command, `[verb, noun, positional, "name* name+ name! name"]` (`*` required, `+` repeatable, `!` a flag; a trailing `?` on the positional: optional) */
export function render(table) {
  const rows = table.filter((c) => c.verb !== 'run').map((c) => {
    const opts = c.options.map((o) => o.name + (o.required ? '*' : '') + (o.multi ? '+' : '') + (o.flag ? '!' : '')).sort().join(' ');
    return `  [${JSON.stringify(c.verb)}, ${JSON.stringify(c.noun)}, ${JSON.stringify(c.positional + (c.optionalPositional ? '?' : ''))}, ${JSON.stringify(opts)}],`;
  });
  return `// GENERATED from cli/command_table.cpp by tools/command_table.mjs (node tools/command_table.mjs): do not edit. One entry a command of the command line:
// [verb, noun, positional ("" none, a trailing ? optional), options], each option \`name\`, \`*\` required, \`+\` repeatable (a list), \`!\` a flag.
// \`run\` is nxr.run (its own function); every other command is nxr.<verb>.<noun>(options).
export const COMMANDS = [
${rows.join('\n')}
];
`;
}

/** the generated declarations: the options of each command (its positional argument and its options by name), the verbs and their nouns */
export function renderTypes(table) {
  const cmds = table.filter((c) => c.verb !== 'run');
  const safe = (w) => w.replace(/(^|[-_])(\w)/g, (_, __, c) => c.toUpperCase());
  const lines = [];
  for (const c of cmds) {
    const arrays = new Set(ARRAY_OPTIONS[`${c.verb} ${c.noun}`] ?? []);
    const name = `${safe(c.verb)}${safe(c.noun)}Options`;
    lines.push(`export interface ${name} {`);
    if (c.positional) lines.push(`  /** the positional argument (\`nxr-neuro ${c.verb} ${c.noun} ${c.positional.toUpperCase()}\`) */`, `  ${c.positional}${c.optionalPositional ? '?' : ''}: string;`);
    for (const o of [...c.options].sort((a, b) => a.name.localeCompare(b.name))) {
      const type = o.flag ? 'boolean' : arrays.has(o.name) ? 'ArrayInput | string' : o.name === 'measure' && c.verb === 'add' ? 'Record<string, ArrayInput | string> | string[]' : o.multi ? 'string[] | string' : 'string | number';
      lines.push(`  ${o.name}${o.required ? '' : '?'}: ${type};`);
    }
    lines.push('  /** receives each progress line the command prints, as an object */', '  progress?: (line: Record<string, unknown>) => void;', '}');
  }
  const verbs = new Map();
  for (const c of cmds) {
    if (!c.noun) continue;
    if (!verbs.has(c.verb)) verbs.set(c.verb, []);
    verbs.get(c.verb).push(c);
  }
  const fn = (c) => {
    const T = `${safe(c.verb)}${safe(c.noun)}Options`;
    const needed = c.options.some((o) => o.required);
    if (!c.positional) return `(options${needed ? '' : '?'}: ${T}) => Promise<Result>`;
    // the positional argument first, or in the options object
    return `{ (${c.positional}: string, options${needed ? '' : '?'}: Omit<${T}, '${c.positional}'>): Promise<Result>; (options: ${T}): Promise<Result> }`;
  };
  lines.push('', 'export interface Verbs {');
  for (const [verb, list] of verbs) {
    lines.push(`  ${/^[a-z_]\w*$/i.test(verb) ? verb : JSON.stringify(verb)}: {`);
    for (const c of list) lines.push(`    ${/^[a-z_]\w*$/i.test(c.noun.replaceAll('-', '_')) ? c.noun.replaceAll('-', '_') : JSON.stringify(c.noun)}: ${c.verb === 'list' && c.noun === 'pipelines' ? '(options?: { progress?: (line: Record<string, unknown>) => void }) => Promise<Result>' : fn(c)};`);
    lines.push('  };');
  }
  const compile = cmds.find((c) => c.verb === 'compile' && !c.noun);
  if (compile) lines.push(`  compile: ${fn(compile)};`);
  lines.push('}', '');
  return `// GENERATED from cli/command_table.cpp by tools/command_table.mjs (node tools/command_table.mjs): do not edit.
import type { ArrayInput, Result } from './index.js';
${lines.join('\n')}`;
}

if (process.argv[1] === fileURLToPath(import.meta.url)) {
  const table = readTable();
  const files = [[COMMANDS_FILE, render(table)], [TYPES_FILE, renderTypes(table)]];
  if (process.argv.includes('--check')) {
    for (const [file, text] of files)
      if (!existsSync(file) || readFileSync(file, 'utf8') !== text) { console.error(`${file} is not what cli/command_table.cpp gives: run node tools/command_table.mjs`); process.exit(1); }
  } else for (const [file, text] of files) writeFileSync(file, text);
}
