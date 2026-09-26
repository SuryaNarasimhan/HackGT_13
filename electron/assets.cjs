const path = require('node:path');
const fs = require('node:fs/promises');

// Expose only the app and the three enabled models, never arbitrary repo files.
const routes = new Map([
  ...['index.html', 'styles.css', 'app.js', 'tab-capture.js', 'face-analysis.js', 'face-state.js', 'face-worker.js', 'live-analysis.js', 'audio-worklet.js', 'overlay.html', 'overlay.css', 'overlay.js', 'analysis-overlay.html', 'analysis-overlay.css', 'analysis-overlay.js'].map(name => [`/${name}`, `src/${name}`]),
  ['/vendor/human.js', 'node_modules/@vladmandic/human/dist/human.js'],
  ...['blazeface', 'facemesh', 'emotion'].flatMap(name => ['json', 'bin'].map(ext => [`/models/${name}.${ext}`, `node_modules/@vladmandic/human/models/${name}.${ext}`]))
]);
const CSP = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; media-src 'self' blob:; connect-src 'self'; worker-src 'self'; object-src 'none'; base-uri 'none'; frame-src 'none'";
function resolveAsset(url, root) {
  try {
    const parsed = new URL(url);
    if (parsed.protocol !== 'msas:' || parsed.host !== 'app' || parsed.username || parsed.password) return null;
    const relative = routes.get(parsed.pathname);
    return relative ? path.resolve(root, relative) : null;
  } catch { return null; }
}
async function assetResponse(request, root) {
  if (!['GET', 'HEAD'].includes(request.method)) return new Response(null, { status: 405 });
  const file = resolveAsset(request.url, root);
  if (!file) return new Response(null, { status: 404 });
  try {
    const data = await fs.readFile(file);
    const mime = { '.html': 'text/html; charset=utf-8', '.js': 'text/javascript; charset=utf-8', '.css': 'text/css; charset=utf-8', '.json': 'application/json', '.bin': 'application/octet-stream' }[path.extname(file)];
    return new Response(request.method === 'HEAD' ? null : data, { headers: {
      'Content-Type': mime, 'Content-Security-Policy': CSP,
      'X-Content-Type-Options': 'nosniff', 'Cache-Control': 'no-store'
    } });
  } catch { return new Response(null, { status: 404 }); }
}
module.exports = { resolveAsset, assetResponse, CSP };
