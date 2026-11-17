// A datastore's KEYS as the parity tests see them, whatever the store format. At format 51 a key is a file of the subject's folder tree; at format 52 every
// zarr.json and every value under 1 MiB is a ROW of the subject's `small_files.sqlite` (key relative to the subject's folder), larger ones stay files.
// So the tests list, read and copy KEYS (paths relative to the datastore root, as at 51), never files, and compare the database's rows, not the file.
import { cpSync, existsSync, mkdirSync, readFileSync, readdirSync } from 'node:fs';
import { DatabaseSync } from 'node:sqlite';
import path from 'node:path';

export const SMALL_FILES = 'small_files.sqlite';
const isDbFile = (f) => path.basename(f).startsWith(SMALL_FILES) && /^small_files\.sqlite(-wal|-shm|-journal)?$/.test(path.basename(f));

/** every key of the datastore (relative paths): the files, except the small-file databases, and the rows of those databases */
export function keysOf(root) {
  const out = [];
  const entries = readdirSync(root, { recursive: true, withFileTypes: true });
  for (const e of entries) {
    if (!e.isFile()) continue;
    const rel = path.relative(root, path.join(e.parentPath, e.name));
    if (e.name === SMALL_FILES) {
      const dir = path.dirname(rel), db = new DatabaseSync(path.join(root, rel), { readOnly: true });
      try { for (const r of db.prepare('SELECT key FROM small_files').all()) out.push(path.posix.join(dir.split(path.sep).join('/'), r.key)); } finally { db.close(); }
    } else if (!isDbFile(rel)) out.push(rel.split(path.sep).join('/'));
  }
  return out.sort();
}

/** the bytes of a key: its row when its subject has a small-file database holding it, else its file */
export function readKey(root, key) {
  const parts = key.split('/');
  for (let n = parts.length - 1; n >= 1; n--) {
    const dir = parts.slice(0, n).join('/'), file = path.join(root, dir, SMALL_FILES);
    if (!existsSync(file)) continue;
    const db = new DatabaseSync(file, { readOnly: true });
    try { const r = db.prepare('SELECT value FROM small_files WHERE key = ?').get(parts.slice(n).join('/')); if (r) return Buffer.from(r.value); } finally { db.close(); }
  }
  return readFileSync(path.join(root, key));
}

/** copy a node (the key and everything under it) from one datastore to another, rows and files alike: the same bytes in the same place */
export function copyKeys(from, to, key) {
  const src = path.join(from, key);
  if (existsSync(src)) { mkdirSync(path.dirname(path.join(to, key)), { recursive: true }); cpSync(src, path.join(to, key), { recursive: true }); }
  const parts = key.split('/');
  for (let n = parts.length - 1; n >= 1; n--) {
    const dir = parts.slice(0, n).join('/'), file = path.join(from, dir, SMALL_FILES);
    if (!existsSync(file)) continue;
    const rel = parts.slice(n).join('/'), a = new DatabaseSync(file, { readOnly: true });
    try {
      const rows = a.prepare('SELECT key, value FROM small_files WHERE key = ? OR (key >= ? AND key < ?)').all(rel, `${rel}/`, `${rel}0`);
      if (!rows.length) continue;
      const b = new DatabaseSync(path.join(to, dir, SMALL_FILES));
      try {
        b.exec('PRAGMA busy_timeout = 5000');
        b.exec('CREATE TABLE IF NOT EXISTS small_files (key TEXT PRIMARY KEY, value BLOB NOT NULL)');
        const put = b.prepare('INSERT OR REPLACE INTO small_files (key, value) VALUES (?, ?)');
        for (const r of rows) put.run(r.key, r.value);
      } finally { b.close(); }
    } finally { a.close(); }
  }
}
