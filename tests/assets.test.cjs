const { test } = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const { resolveAsset, assetResponse } = require('../electron/assets.cjs');
const root = path.resolve(__dirname, '..');
test('only explicitly listed local assets are available', () => {
  assert.equal(resolveAsset('msas://app/index.html', root), path.join(root, 'src/index.html'));
  assert.equal(resolveAsset('msas://app/analysis-overlay.html', root), path.join(root, 'src/analysis-overlay.html'));
  assert.equal(resolveAsset('msas://app/audio-worklet.js', root), path.join(root, 'src/audio-worklet.js'));
  for (const url of ['https://app/index.html', 'msas://other/index.html', 'msas://app/electron/main.cjs', 'msas://app/../package.json', 'msas://app/%2e%2e%2fpackage.json', 'msas://user@app/index.html', 'msas://app/models/faceres.json']) assert.equal(resolveAsset(url, root), null);
});
test('local responses enforce no remote connections and deny writes', async () => {
  const response = await assetResponse({ url: 'msas://app/index.html', method: 'GET' }, root);
  assert.equal(response.status, 200);
  assert.ok(response.headers.get('Content-Security-Policy').includes("connect-src 'self'"));
  assert.equal((await assetResponse({ url: 'msas://app/index.html', method: 'POST' }, root)).status, 405);
});
test('all three required model manifests and weights are installed', async () => {
  for (const name of ['blazeface', 'facemesh', 'emotion']) {
    const response = await assetResponse({ url: `msas://app/models/${name}.json`, method: 'GET' }, root);
    assert.equal(response.status, 200);
    const manifest = await response.json();
    for (const group of manifest.weightsManifest) for (const file of group.paths) {
      assert.equal((await assetResponse({ url: `msas://app/models/${file}`, method: 'GET' }, root)).status, 200);
    }
  }
});
