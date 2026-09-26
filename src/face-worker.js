/* All imports and model fetches are served by the local msas://app protocol. */
importScripts('./vendor/human.js');
let human = null, busy = false;
const config = {
  backend: 'webgl', modelBasePath: new URL('./models/', self.location.href).href,
  cacheModels: false, cacheSensitivity: 0, debug: false, async: false, warmup: 'none',
  filter: { enabled: false, return: false }, gesture: { enabled: false },
  face: {
    enabled: true,
    detector: { maxDetected: 2, minConfidence: .6, minSize: 32, rotation: true, return: false, skipFrames: 0, skipTime: 0 },
    mesh: { enabled: true }, iris: { enabled: false }, attention: { enabled: false },
    emotion: { enabled: true, minConfidence: 0, skipFrames: 0, skipTime: 0 },
    description: { enabled: false }, gear: { enabled: false }, antispoof: { enabled: false }, liveness: { enabled: false }
  },
  body: { enabled: false }, hand: { enabled: false }, object: { enabled: false }, segmentation: { enabled: false }
};
async function initialize(backend) {
  human = new Human.default({ ...config, backend });
  await human.load();
  if (!['blazeface', 'facemesh', 'emotion'].every(name => human.models.loaded().includes(name))) throw new Error('A required model could not load.');
}
self.onmessage = async ({ data }) => {
  if (busy) return;
  busy = true;
  try {
    if (data.type === 'init') {
      try { await initialize('webgl'); }
      catch { await initialize('cpu'); }
      postMessage({ type: 'ready' });
    } else if (data.type === 'frame' && human) {
      const image = new ImageData(new Uint8ClampedArray(data.pixels), data.width, data.height);
      let result;
      try { result = await human.detect(image); }
      catch (error) {
        if (human.config.backend === 'cpu') throw error;
        await initialize('cpu');
        result = await human.detect(image);
      }
      if (result.error) throw new Error(String(result.error));
      // Strip all fields except face geometry and expression observations.
      postMessage({ type: 'result', timestamp: data.timestamp, faces: result.face.map(face => ({
        box: face.box, boxScore: face.boxScore, faceScore: face.faceScore, emotion: face.emotion,
        mesh: face.mesh.map(point => [point[0] / data.width, point[1] / data.height])
      })) });
    }
  } catch (error) {
    postMessage({ type: 'error', message: `Local expression model unavailable: ${String(error.message || error).slice(0, 180)}` });
  } finally { busy = false; }
};
