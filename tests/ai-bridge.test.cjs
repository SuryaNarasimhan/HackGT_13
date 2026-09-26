const { test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { AiBridge, pythonCommand, safeEvent, MAX_AUDIO_BYTES, MAX_FRAME_BYTES } = require('../electron/ai-bridge.cjs');

test('python command prefers explicit configuration', () => {
  assert.equal(pythonCommand('C:\\missing', { MSAS_PYTHON: 'custom-python' }).command, 'custom-python');
});

test('AI events require a typed object', () => {
  assert.deepEqual(safeEvent({ type: 'ready', gemini: false }), { type: 'ready', gemini: false });
  for (const value of [null, [], {}, { type: 4 }, 'ready']) assert.equal(safeEvent(value), null);
});

test('bridge rejects oversized and malformed media before IPC', () => {
  const bridge = new AiBridge(path.resolve(__dirname, '..'), () => {});
  assert.equal(bridge.sendAudio(Buffer.alloc(3)), false);
  assert.equal(bridge.sendAudio(Buffer.alloc(MAX_AUDIO_BYTES + 4)), false);
  assert.equal(bridge.sendFrame(Buffer.alloc(MAX_FRAME_BYTES + 1)), false);
});
