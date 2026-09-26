const http = require('node:http');
const crypto = require('node:crypto');

const PORT = 47831;
const LIMIT = 65536;
function validSignal(message, type) {
  return message && message.type === type && typeof message.sdp === 'string' &&
    message.sdp.startsWith('v=0') && message.sdp.length <= LIMIT / 2;
}

// Signaling only: media flows over a paired, same-machine WebRTC connection.
// No files, credentials, or media are exposed through this HTTP listener.
async function createTabBridge({ onSignal, onClose, audio = false, port = PORT, timeoutMs = 120000 }) {
  const token = crypto.randomBytes(24).toString('hex');
  let owner = null, offered = false, answer = null, closed = false;
  let lastPoll = Date.now(), expiry;
  const server = http.createServer(async (req, res) => {
    const origin = req.headers.origin;
    // Extension GET requests can omit Origin when host permissions apply.
    // The random bearer token authenticates; the client ID binds the pairing.
    const client = req.headers['x-msas-client'];
    const host = `127.0.0.1:${server.address()?.port}`;
    const reply = (status, data = {}) => {
      if (!res.destroyed) { res.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' }); res.end(JSON.stringify(data)); }
    };
    if (closed || req.headers.host !== host || (origin && !/^chrome-extension:\/\/[a-p]{32}$/.test(origin))) return reply(403);
    if (origin) res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Vary', 'Origin');
    if (req.method === 'OPTIONS') {
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST');
      res.setHeader('Access-Control-Allow-Headers', 'authorization, content-type, x-msas-client');
      return reply(204);
    }
    if (!/^[a-p]{32}$/.test(client || '') || (origin && origin !== `chrome-extension://${client}`)) return reply(403);
    const given = Buffer.from(req.headers.authorization || '');
    const expected = Buffer.from(`Bearer ${token}`);
    if (given.length !== expected.length || !crypto.timingSafeEqual(given, expected) || (owner && owner !== client)) return reply(401);
    if (req.method === 'GET' && req.url === '/config') return reply(200, { audio });
    if (req.method === 'GET' && req.url === '/answer') {
      if (!offered) return reply(409);
      lastPoll = Date.now();
      return reply(200, { answer });
    }
    if (req.method !== 'POST' || req.url !== '/signal') return reply(404);
    let size = 0, body = '';
    try {
      for await (const chunk of req) {
        size += chunk.length;
        if (size > LIMIT) { reply(413); req.destroy(); return; }
        body += chunk.toString('utf8');
      }
      const message = JSON.parse(body);
      if (message.type === 'stop') { reply(200); close(); onClose('Browser tab disconnected.'); return; }
      if (!validSignal(message, 'offer')) return reply(400);
      if (offered) return reply(409);
      offered = true; owner = client; lastPoll = Date.now();
      reply(200);
      onSignal({ type: 'offer', sdp: message.sdp });
    } catch { reply(400); }
  });
  server.requestTimeout = 5000;
  server.headersTimeout = 5000;
  server.maxHeadersCount = 20;
  await new Promise((resolve, reject) => { server.once('error', reject); server.listen(port, '127.0.0.1', resolve); });
  const heartbeat = setInterval(() => {
    if (offered && Date.now() - lastPoll > 10000) { close(); onClose('Browser tab connection was lost. Reconnect the extension.'); }
  }, 2000);
  heartbeat.unref();
  expiry = setTimeout(() => { if (!offered) { close(); onClose('Pairing expired. Click Connect browser tab to try again.'); } }, timeoutMs);
  expiry.unref();
  function close() {
    if (closed) return;
    closed = true; answer = null;
    clearInterval(heartbeat); clearTimeout(expiry);
    server.close(); server.closeAllConnections();
  }
  return {
    token, port: server.address().port,
    send(message) {
      if (closed || !offered || answer || !validSignal(message, 'answer')) throw new Error('Invalid or expired tab answer.');
      answer = { type: 'answer', sdp: message.sdp };
    },
    close
  };
}
module.exports = { createTabBridge, validSignal, PORT };
