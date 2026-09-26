const $ = id => document.getElementById(id);
const api = window.msas;
const faceAnalysis = window.FaceAnalysis.create($('video'));
let selected = null, stream = null, audioContext = null, audioTimer = null;
let generation = 0, currentMode = 'idle', lastState = null;
const tabCapture = api ? window.TabCapture.create(api, receiveTabStream, async message => { await api.stop(); notice(message); }) : null;

function notice(message = '') { $('notice').textContent = message; $('notice').hidden = !message; }
function releaseMedia() {
  tabCapture?.stop();
  faceAnalysis.stop();
  generation++;
  if (stream) { stream.getTracks().forEach(track => { track.onended = null; track.stop(); }); stream = null; }
  clearInterval(audioTimer); audioTimer = null;
  if (audioContext) { audioContext.close().catch(() => {}); audioContext = null; }
  $('video').srcObject = null; $('video').hidden = true; $('preview-empty').hidden = false;
  $('audio-status').hidden = true; $('audio-meter').value = 0;
}
function clearSession() {
  releaseMedia(); selected = null;
  $('source-name').textContent = 'No source selected';
  $('language').selectedIndex = 0; $('region').value = ''; $('speaker-context').value = '';
  $('system-audio').checked = false;
  $('tab-pairing').hidden = true; $('pairing-code').value = '';
}
function render(state) {
  currentMode = state.mode; lastState = state;
  const idle = state.mode === 'idle';
  $('session-status').textContent = ({ idle: 'Ready', starting: 'Connecting…', live: 'Capture active', demo: 'Simulated session' })[state.mode];
  faceAnalysis.setLive(state.mode === 'live');
  $('session-status').dataset.active = String(!idle);
  $('start').hidden = !idle; $('start').disabled = !selected;
  $('stop').hidden = idle; $('minimize').hidden = idle;
  $('choose-source').disabled = !idle; $('change-source').disabled = !idle || !selected;
  $('connect-tab').disabled = !idle;
  $('system-audio').disabled = !idle; $('context-fields').disabled = !idle;
}
async function connectTab() {
  if (currentMode !== 'idle') return;
  notice(); $('connect-tab').disabled = true;
  const token = generation;
  try {
    const result = await tabCapture.start($('system-audio').checked);
    if (token !== generation || currentMode !== 'starting') { tabCapture.stop(); return; }
    selected = null;
    $('source-name').textContent = 'Waiting for the browser extension';
    $('pairing-code').value = result.token; $('tab-pairing').hidden = false;
    $('pairing-code').focus(); $('pairing-code').select();
  } catch (error) { notice(error.message || 'Could not start the browser connection.'); render(lastState); }
}
async function receiveTabStream(media) {
  if (currentMode !== 'starting') { media.getTracks().forEach(track => track.stop()); return; }
  const token = generation;
  stream = media;
  $('video').srcObject = media; $('video').hidden = false; $('preview-empty').hidden = true;
  await captureTimeout($('video').play(), 'Browser video did not reach the preview.');
  if (token !== generation) return;
  $('source-name').textContent = 'Browser tab · live connection';
  $('tab-pairing').hidden = true; $('pairing-code').value = '';
  try { monitorAudio(media, $('system-audio').checked); }
  catch { $('audio-label').textContent = 'Video active · audio meter unavailable'; }
  await api.captureReady(media.getAudioTracks().length > 0);
}
async function loadSources() {
  $('source-list').textContent = 'Finding available windows and screens…';
  $('refresh-sources').disabled = true;
  try {
    const sources = await api.listSources();
    $('source-list').replaceChildren();
    if (!sources.length) $('source-list').textContent = 'No sources available. Check screen capture permissions and try again.';
    for (const source of sources) {
      const button = document.createElement('button'); button.className = 'source-option';
      const image = document.createElement('img'); image.src = source.thumbnail; image.alt = '';
      const label = document.createElement('span'); label.textContent = source.name; button.title = source.name;
      button.append(image, label);
      button.addEventListener('click', () => {
        selected = source;
        $('source-name').textContent = source.name;
        $('source-dialog').close(); $('source-list').replaceChildren();
        render(lastState); notice();
      });
      $('source-list').append(button);
    }
  } catch { $('source-list').textContent = 'Could not list sources. Check operating-system screen capture permissions, then refresh.'; }
  finally { $('refresh-sources').disabled = false; }
}
function openPicker() { $('source-dialog').showModal(); loadSources(); }
function monitorAudio(media, requested) {
  $('audio-status').hidden = false;
  if (!media.getAudioTracks().length) {
    $('audio-label').textContent = requested ? 'No audio track available · video only' : 'System audio off';
    if (requested) notice('Video is active, but no system audio track was returned. Stop and check capture permissions before your Meet trial.');
    return;
  }
  audioContext = new AudioContext();
  audioContext.resume().catch(() => {});
  const source = audioContext.createMediaStreamSource(media), analyser = audioContext.createAnalyser();
  analyser.fftSize = 256; source.connect(analyser);
  const samples = new Uint8Array(analyser.fftSize);
  let lastSignal = Date.now();
  audioTimer = setInterval(() => {
    analyser.getByteTimeDomainData(samples);
    const rms = Math.sqrt(samples.reduce((sum, value) => sum + ((value - 128) / 128) ** 2, 0) / samples.length);
    $('audio-meter').value = Math.min(1, rms * 5);
    if (rms > .002) lastSignal = Date.now();
    $('audio-label').textContent = Date.now() - lastSignal > 8000 ? 'Audio track present · no recent signal. Check call sound.' : 'System audio track active';
  }, 180);
}
async function captureTimeout(promise, message) {
  let timer;
  try {
    return await Promise.race([
      promise,
      new Promise((_, reject) => {
        timer = setTimeout(() => reject(new DOMException(message, 'TimeoutError')), 15000);
      })
    ]);
  } finally { clearTimeout(timer); }
}
function captureErrorMessage(error, stage) {
  const name = error?.name || 'Error';
  const detail = String(error?.message || 'No additional details returned.').slice(0, 500);
  const guidance = {
    NotAllowedError: 'Screen capture was denied. Restart MSAS to load the capture-permission fix, then select the window again.',
    InvalidStateError: 'Keep the MSAS setup window focused and click Start companion again.',
    NotReadableError: 'Windows could not read that window. Keep the Meet browser window open and restored, then select it again.',
    NotFoundError: 'The selected window may have closed. Refresh the source list and choose the open Meet window.',
    AbortError: 'Windows interrupted capture. Select the window and try again.',
    TimeoutError: 'Keep the Meet tab active in its browser window and restore the window if it is minimized. Then select it again.'
  };
  return `Capture failed while ${stage} (${name}): ${detail} ${guidance[name] || 'Select the window again. If this repeats, share this exact message.'}`;
}
async function startCapture() {
  if (!selected || currentMode !== 'idle') return;
  notice(); $('start').disabled = true;
  const token = ++generation, choice = { id: selected.id, audio: $('system-audio').checked };
  let stage = 'preparing the selected window';
  try {
    await api.prepareCapture(choice);
    if (token !== generation) return;
    stage = 'starting Windows capture';
    const pendingMedia = navigator.mediaDevices.getDisplayMedia({ video: { frameRate: 12 }, audio: choice.audio }).then(media => {
      // A native request can finish after Stop or a timeout. Never leave that
      // late stream capturing in the background.
      if (token !== generation) { media.getTracks().forEach(track => track.stop()); return null; }
      return media;
    });
    const media = await captureTimeout(pendingMedia, 'Windows did not return a capture stream within 15 seconds.');
    if (!media || token !== generation) { media?.getTracks().forEach(track => track.stop()); return; }
    stream = media;
    const videoTrack = media.getVideoTracks()[0];
    if (!videoTrack) throw new Error('No video track');
    videoTrack.onended = () => { api.stop(); notice('Capture ended. Choose a source to start again.'); };
    $('video').srcObject = media; $('video').hidden = false; $('preview-empty').hidden = true;
    stage = 'displaying the live preview';
    await captureTimeout($('video').play(), 'The window was selected, but no video frame reached the preview within 15 seconds.');
    if (token !== generation) return;
    if (!$('video').videoWidth || !$('video').videoHeight) throw new Error('The captured stream has no video dimensions.');
    try { monitorAudio(media, choice.audio); }
    catch (error) {
      // A level-meter failure must not tear down a working window preview.
      clearInterval(audioTimer); audioTimer = null;
      if (audioContext) { audioContext.close().catch(() => {}); audioContext = null; }
      $('audio-status').hidden = false;
      $('audio-label').textContent = 'Video active · audio meter unavailable';
      notice(`The video preview is working, but the audio meter could not start (${error.name || 'Error'}).`);
    }
    stage = 'opening the companion overlay';
    await api.captureReady(media.getAudioTracks().length > 0);
  } catch (error) {
    if (token !== generation) return;
    const message = captureErrorMessage(error, stage);
    releaseMedia();
    await api.stop();
    notice(message);
  }
}
if (!api) {
  notice('Open this interface through the MSAS desktop app (npm start). A browser preview cannot create a desktop overlay or start capture.');
  document.querySelectorAll('button').forEach(button => { button.disabled = true; });
} else {
  api.onState(render); api.onStop(clearSession); api.getState().then(render);
  api.onTabError(notice);
  $('connect-tab').addEventListener('click', connectTab);
  $('choose-source').addEventListener('click', openPicker); $('change-source').addEventListener('click', openPicker);
  $('refresh-sources').addEventListener('click', loadSources);
  $('close-picker').addEventListener('click', () => $('source-dialog').close());
  $('source-dialog').addEventListener('close', () => $('source-list').replaceChildren());
  $('start').addEventListener('click', startCapture);
  $('stop').addEventListener('click', () => { notice(); api.stop(); });
  $('minimize').addEventListener('click', () => api.minimize());
  window.addEventListener('beforeunload', releaseMedia);
}
