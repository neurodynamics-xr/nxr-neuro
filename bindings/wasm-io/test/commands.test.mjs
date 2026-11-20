// The command table is the vocabulary (docs/specs/2026-11-19-io-service.md, rule 7), for the JavaScript face: every command of cli/command_table.cpp has its function,
// `nxr.<verb>.<noun>`, whose options are the command's (the names with underscores, the required ones, the repeatable ones, the flags), no function is left without a command, and
// the generated files (js/commands.mjs, js/commands.d.ts) are what the table gives. Mirrors bindings/python/tests/test_commands.py.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { COMMANDS_FILE, TYPES_FILE, readTable, render, renderTypes } from '../tools/command_table.mjs';
import { skip, load } from './helpers.mjs';

const table = readTable();
const noun = (n) => n.replaceAll('-', '_');

test('the generated files are what cli/command_table.cpp gives', () => {
  assert.ok(table.length >= 75);
  for (const [file, text] of [[COMMANDS_FILE, render(table)], [TYPES_FILE, renderTypes(table)]]) {
    assert.ok(existsSync(file), `${file} is missing: run node tools/command_table.mjs`);
    assert.equal(readFileSync(file, 'utf8'), text, `${file} is not what the table gives: run node tools/command_table.mjs`);
  }
});

test('every command has its function with the same options, and no function is left without a command', { skip }, async () => {
  const { nxr } = await load();
  const problems = [];
  const commands = table.filter((c) => c.verb !== 'run');   // `run PIPELINE`: nxr.run (below)
  for (const c of commands) {
    const name = `nxr.${c.verb}${c.noun ? `.${noun(c.noun)}` : ''}`;
    const f = c.noun ? nxr[c.verb]?.[noun(c.noun)] : nxr[c.verb];
    if (typeof f !== 'function') { problems.push(`${c.verb} ${c.noun}: no function ${name}`); continue; }
    if (c.verb === 'list' && c.noun === 'pipelines') continue;   // `list pipelines` is answered from the library's pipelines (no options of its own)
    if (f.positional !== c.positional || (c.positional && f.optionalPositional !== c.optionalPositional)) problems.push(`${name}: the positional argument is ${c.positional}${c.optionalPositional ? '?' : ''}`);
    const got = new Map(f.options.map((o) => [o.name, o]));
    const want = new Map(c.options.map((o) => [o.name, o]));
    const missing = [...want.keys()].filter((k) => !got.has(k)), extra = [...got.keys()].filter((k) => !want.has(k));
    if (missing.length || extra.length) { problems.push(`${name}: options differ: missing ${missing}, extra ${extra}`); continue; }
    for (const [k, w] of want) {
      const g = got.get(k);
      if (g.required !== w.required || g.multi !== w.multi || g.flag !== w.flag) problems.push(`${name}: ${k} is ${JSON.stringify(g)}, the table says ${JSON.stringify(w)}`);
    }
  }
  for (const verb of new Set(commands.filter((c) => c.noun).map((c) => c.verb))) {
    for (const [n, f] of Object.entries(nxr[verb] ?? {}))
      if (typeof f === 'function' && !commands.some((c) => c.verb === verb && noun(c.noun) === n)) problems.push(`nxr.${verb}.${n}: no command \`${verb} ${n}\` in the command table`);
  }
  assert.deepEqual(problems, []);
});

test('run takes the command line\'s range and flags; the old forms are refused, naming the new', { skip }, async () => {
  const { nxr, NxrError } = await load();
  await assert.rejects(nxr.run('psd', { dataset: '/nowhere', subject: 'S1', start: 0, stop: 10 }), (e) => e instanceof NxrError && e.code === 'invalid_argument' && /from and to/.test(e.message));
  await assert.rejects(nxr.run('psd', { dataset: '/nowhere' }), /needs the subject/);
  await assert.rejects(nxr.run('psd', { subject: 'S1' }), /needs the dataset/);
  await assert.rejects(nxr.create.subject({ dataset: '/nowhere' }), (e) => e instanceof NxrError && e.code === 'invalid_argument' && /name/.test(e.message));   // a required option is required
  await assert.rejects(nxr.create.subject({ dataset: '/nowhere', name: 'S', bogus: 1 }), /unrecognized argument: bogus/);                                   // an option the command does not take
  assert.equal(typeof nxr.compute, 'object');   // the geometry compute: nxr.compute is not a command
});
