const { spawn } = require('node:child_process');
const fs = require('node:fs');
const path = require('node:path');

const MAX_AUDIO_BYTES = 64 * 1024;
const MAX_FRAME_BYTES = 700 * 1024;
const MAX_OUTPUT_LINE = 1024 * 1024;

function pythonCommand(root, env = process.env) {
  const configured = env.MSAS_PYTHON;
  const local = path.join(root, '.venv', 'Scripts', 'python.exe');
  if (configured) return { command: configured, args: [] };
  if (fs.existsSync(local)) return { command: local, args: [] };
  return { command: 'python', args: [] };
}

function safeEvent(value) {
  if (!value || typeof value !== 'object' || Array.isArray(value) || typeof value.type !== 'string') return null;
  return value;
}

class AiBridge {
  constructor(root, onEvent, options = {}) {
    this.root = root;
    this.onEvent = onEvent;
    this.options = options;
    this.child = null;
    this.buffer = '';
    this.stopping = false;
  }

  async start() {
    if (this.child) return;
    this.stopping = false;
    this.buffer = '';
    const selected = this.options.command
      ? { command: this.options.command, args: this.options.args || [] }
      : pythonCommand(this.root, this.options.env);
    const args = [...selected.args, ...(this.options.moduleArgs || ['-u', '-m', 'app.electron_bridge'])];
    const child = spawn(selected.command, args, {
      cwd: this.root, windowsHide: true, shell: false,
      stdio: ['pipe', 'pipe', 'pipe'], env: { ...process.env, ...this.options.env, PYTHONUNBUFFERED: '1' }
    });
    this.child = child;
    child.stdout.setEncoding('utf8');
    child.stdout.on('data', chunk => this._read(chunk));
    child.stderr.setEncoding('utf8');
    child.stderr.on('data', chunk => {
      const line = String(chunk).trim();
      if (line) process.stderr.write(`[MSAS AI] ${line.slice(0, 2000)}\n`);
    });
    child.once('error', error => {
      if (this.child === child) this.child = null;
      this.onEvent({ type: 'error', message: `AI backend could not start (${error.code || error.message}). Run npm run setup:ai, then restart MSAS.` });
    });
    child.once('exit', code => {
      if (this.child === child) this.child = null;
      if (!this.stopping) this.onEvent({ type: 'error', message: `AI backend exited${code === null ? '' : ` with code ${code}`}. Run npm run setup:ai, then restart MSAS.` });
    });
    this.send({ type: 'start' });
  }

  _read(chunk) {
    this.buffer += chunk;
    if (this.buffer.length > MAX_OUTPUT_LINE * 2) {
      this.buffer = '';
      this.onEvent({ type: 'error', message: 'AI backend returned an oversized response.' });
      return;
    }
    for (;;) {
      const newline = this.buffer.indexOf('\n');
      if (newline < 0) break;
      const line = this.buffer.slice(0, newline); this.buffer = this.buffer.slice(newline + 1);
      if (!line || line.length > MAX_OUTPUT_LINE) continue;
      try {
        const event = safeEvent(JSON.parse(line));
        if (event) this.onEvent(event);
      } catch { this.onEvent({ type: 'error', message: 'AI backend returned an unreadable response.' }); }
    }
  }

  send(message) {
    if (!this.child || this.child.stdin.destroyed || !this.child.stdin.writable) return false;
    return this.child.stdin.write(`${JSON.stringify(message)}\n`);
  }

  sendAudio(bytes) {
    const data = Buffer.from(bytes);
    if (!data.length || data.length > MAX_AUDIO_BYTES || data.length % 4) return false;
    return this.send({ type: 'audio', data: data.toString('base64') });
  }

  sendFrame(bytes) {
    const data = Buffer.from(bytes);
    if (!data.length || data.length > MAX_FRAME_BYTES) return false;
    return this.send({ type: 'frame', data: data.toString('base64') });
  }

  stop() {
    const child = this.child;
    if (!child) return;
    this.stopping = true;
    this.send({ type: 'stop' });
    child.stdin.end();
    const timer = setTimeout(() => { if (!child.killed) child.kill(); }, 1500);
    timer.unref();
    child.once('exit', () => clearTimeout(timer));
    this.child = null;
  }
}

module.exports = { AiBridge, pythonCommand, safeEvent, MAX_AUDIO_BYTES, MAX_FRAME_BYTES };
