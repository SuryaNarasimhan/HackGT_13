const { test } = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');
const { createTabBridge, validSignal } = require('../electron/tab-bridge.cjs');
const origin = 'chrome-extension://' + 'a'.repeat(32);
async function setup(t, options = {}) {
  const signals = [], errors = [];
  const bridge = await createTabBridge({ port: 0, onSignal: signal => signals.push(signal), onClose: error => errors.push(error), ...options });
  t.after(() => bridge.close());
  const request = (path, body, headers = {}, method = body ? 'POST' : 'GET') => fetch(`http://127.0.0.1:${bridge.port}${path}`, {
    method, headers: { Origin: origin, 'X-Msas-Client': 'a'.repeat(32), Authorization: `Bearer ${bridge.token}`, ...headers }, ...(body ? { body: JSON.stringify(body) } : {})
  });
  return { bridge, request, signals, errors };
}
test('tab pairing requires a valid token, extension origin and loopback host', async t => {
  const { request, bridge } = await setup(t);
  assert.equal((await request('/config', null, { Authorization: 'Bearer wrong' })).status, 401);
  assert.equal((await request('/config', null, { Origin: 'https://untrusted.example' })).status, 403);
  assert.equal((await request('/config', null, { Origin: 'null' })).status, 403);
  await new Promise((resolve, reject) => {
    const req = http.get({ hostname: '127.0.0.1', port: bridge.port, path: '/config', headers: { host: 'untrusted.example', origin, authorization: `Bearer ${bridge.token}` } }, res => { assert.equal(res.statusCode, 403); res.resume(); resolve(); }); req.on('error', reject);
  });
  assert.deepEqual(await (await request('/config')).json(), { audio: false });
  assert.equal((await request('/anything')).status, 404);
});
test('only one offer pairs; bounded answer is delivered to the paired extension', async t => {
  const { request, bridge, signals } = await setup(t, { audio: true });
  assert.equal((await request('/answer')).status, 409);
  assert.equal((await request('/signal', { type: 'offer', sdp: 'bad' })).status, 400);
  const offer = { type: 'offer', sdp: 'v=0\r\n' };
  assert.equal((await request('/signal', offer)).status, 200);
  assert.deepEqual(signals, [offer]);
  assert.equal((await request('/signal', offer)).status, 409);
  assert.equal((await request('/answer', null, { Origin: 'chrome-extension://' + 'b'.repeat(32), 'X-Msas-Client': 'b'.repeat(32) })).status, 401);
  const answer = { type: 'answer', sdp: 'v=0\r\n' };
  bridge.send(answer);
  assert.deepEqual(await (await request('/answer')).json(), { answer });
  assert.throws(() => bridge.send(answer));
});
test('stop closes signaling and invalidates the session', async t => {
  const { request, bridge, errors } = await setup(t);
  assert.equal((await request('/signal', { type: 'stop' })).status, 200);
  assert.equal(errors.length, 1);
  assert.throws(() => bridge.send({ type: 'answer', sdp: 'v=0' }));
  await assert.rejects(request('/config'));
});
test('unused pairing expires, and malformed/oversized descriptions are rejected', async t => {
  const { errors } = await setup(t, { timeoutMs: 20 });
  await new Promise(resolve => setTimeout(resolve, 45));
  assert.equal(errors.length, 1);
  assert.equal(validSignal({ type: 'offer', sdp: 'v=0' + 'x'.repeat(65536) }, 'offer'), false);
  assert.equal(validSignal({ type: 'answer', sdp: 'v=0' }, 'offer'), false);
});
