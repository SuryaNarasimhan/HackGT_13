const { app, BrowserWindow, desktopCapturer, ipcMain, screen, session, protocol } = require('electron');
const path = require('node:path');
const { assetResponse } = require('./assets.cjs');
const { createTabBridge } = require('./tab-bridge.cjs');
const { AiBridge } = require('./ai-bridge.cjs');
protocol.registerSchemesAsPrivileged([{ scheme: 'msas', privileges: { standard: true, secure: true, supportFetchAPI: true, corsEnabled: true, stream: true } }]);

let mainWindow, borderWindow, controlsWindow, analysisWindow;
let aiBridge = null;
let lastAnalysisEvent = { type: 'booting' };
let captureChoice = null;
let sources = new Map();
let displayId = null;
let active = false;
let captureGeneration = 0;
let tabBridge = null, tabSession = null;
let state = { mode: 'idle', cue: null, source: '', audio: false };
const demoCues = [
  { category: 'sarcasm', label: 'Possible sarcasm', quote: '“Great, another meeting that could have been an email.”', meaning: 'They may be expressing frustration about an unnecessary meeting.', evidence: 'The positive word “great” contrasts with the complaint that follows. This is a scripted example, not an analysis of your call.', alternative: 'They could be joking lightly rather than feeling seriously frustrated.' },
  { category: 'idiom', label: 'An idiom, unpacked', quote: '“Let’s put a pin in that and circle back.”', meaning: 'They likely want to pause this topic and return to it later.', evidence: '“Put a pin in that” and “circle back” are common figurative phrases. This is a scripted example.', alternative: 'The phrase does not specify exactly when they will return to the topic.' },
  { category: 'slang', label: 'Slang, translated', quote: '“Your presentation ate. No crumbs.”', meaning: 'They are probably giving an enthusiastic compliment: your presentation was excellent.', evidence: 'In this scripted context, “ate” and “no crumbs” are used as praise, not literally about food.', alternative: 'Meaning still depends on the surrounding conversation and delivery.' },
  { category: 'none', label: 'Room for interpretation', quote: '“Well, that was interesting.”', meaning: 'There is not enough context to suggest a hidden meaning.', evidence: 'This phrase could express interest, surprise, or disappointment. The demo intentionally abstains.', alternative: 'Asking “Interesting in what way?” could clarify what they mean.' }
];
let demoIndex = 0;

function trustedMain(event) {
  return mainWindow && event.sender === mainWindow.webContents && event.senderFrame === mainWindow.webContents.mainFrame;
}
function trustedControls(event) {
  return controlsWindow && event.sender === controlsWindow.webContents && event.senderFrame === controlsWindow.webContents.mainFrame;
}
function trustedAnalysis(event) {
  return analysisWindow && event.sender === analysisWindow.webContents && event.senderFrame === analysisWindow.webContents.mainFrame;
}
function guard(event, controls = false) {
  if (!trustedMain(event) && !(controls && trustedControls(event))) throw new Error('Unauthorized request');
}
function sendState() {
  for (const win of [mainWindow, borderWindow, controlsWindow, analysisWindow]) {
    if (win && !win.isDestroyed()) win.webContents.send('session:state', state);
  }
}
function secureWindow(options) {
  const win = new BrowserWindow({
    ...options,
    webPreferences: {
      preload: path.join(__dirname, 'preload.cjs'),
      contextIsolation: true, nodeIntegration: false, sandbox: true,
      partition: 'msas-volatile', backgroundThrottling: false
    }
  });
  win.webContents.setWindowOpenHandler(() => ({ action: 'deny' }));
  win.webContents.on('will-navigate', event => event.preventDefault());
  win.setMenuBarVisibility(false);
  return win;
}
function currentDisplay() {
  return screen.getAllDisplays().find(display => String(display.id) === displayId)
    || screen.getDisplayMatching(mainWindow.getBounds());
}
function positionOverlays(expanded = false) {
  if (!borderWindow || !controlsWindow) return;
  const display = currentDisplay();
  borderWindow.setBounds(display.bounds);
  const { x, y, width } = display.workArea;
  controlsWindow.setBounds({ x: x + width - 380, y: y + 24, width: 356, height: expanded ? 480 : 112 });
  if (analysisWindow && !analysisWindow.isDestroyed()) {
    const current = analysisWindow.getBounds();
    analysisWindow.setBounds({ x: x + 24, y: y + 24, width: 410, height: current.height });
  }
}
function createOverlays() {
  borderWindow = secureWindow({ title: 'MSAS Border', frame: false, transparent: true, resizable: false,
    focusable: false, skipTaskbar: true, show: false, hasShadow: false });
  controlsWindow = secureWindow({ title: 'MSAS Floating Controls', frame: false, transparent: true,
    resizable: false, skipTaskbar: true, show: false, hasShadow: false });
  analysisWindow = secureWindow({ title: 'MSAS Live Understanding', frame: false, transparent: true,
    resizable: false, skipTaskbar: true, show: false, hasShadow: true, width: 410, height: 620 });
  borderWindow.setIgnoreMouseEvents(true);
  for (const win of [borderWindow, controlsWindow, analysisWindow]) {
    win.setAlwaysOnTop(true, 'screen-saver');
    win.setVisibleOnAllWorkspaces(true, { visibleOnFullScreen: true });
    win.webContents.on('did-finish-load', () => { sendState(); if (active) win.showInactive(); });
  }
  positionOverlays();
  borderWindow.loadURL('msas://app/overlay.html?role=border');
  controlsWindow.loadURL('msas://app/overlay.html?role=controls');
  analysisWindow.loadURL('msas://app/analysis-overlay.html');
  analysisWindow.webContents.on('did-finish-load', () => analysisWindow?.webContents.send('analysis:event', lastAnalysisEvent));
}
function startSession(mode) {
  active = true;
  state = { mode, cue: null, source: mode === 'demo' ? 'Scripted walkthrough' : state.source, audio: false };
  // Keep the setup window focused while getDisplayMedia starts. Live overlays
  // are created only after the renderer has a playing video stream.
  if (mode === 'demo') createOverlays();
  sendState();
}
function stopSession() {
  captureGeneration++;
  tabBridge?.close(); tabBridge = null; tabSession = null;
  aiBridge?.stop(); aiBridge = null;
  lastAnalysisEvent = { type: 'booting' };
  active = false;
  captureChoice = null;
  sources.clear();
  state = { mode: 'idle', cue: null, source: '', audio: false };
  if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send('capture:stop');
  for (const win of [borderWindow, controlsWindow, analysisWindow]) if (win && !win.isDestroyed()) win.destroy();
  borderWindow = controlsWindow = analysisWindow = null;
  sendState();
}

app.whenReady().then(() => {
  const captureSession = session.fromPartition('msas-volatile', { cache: false });
  captureSession.protocol.handle('msas', request => assetResponse(request, path.join(__dirname, '..')));
  captureSession.webRequest.onBeforeRequest({ urls: ['http://*/*', 'https://*/*', 'ws://*/*', 'wss://*/*'] }, (_details, callback) => callback({ cancel: true }));
  const pendingDesktopCapture = (contents, details) =>
    contents === mainWindow?.webContents && details?.isMainFrame === true &&
    active && state.mode === 'starting' && !!captureChoice;
  captureSession.setPermissionRequestHandler((contents, permission, callback, details) => {
    // Electron 44 routes getDisplayMedia through a `media` request BEFORE
    // setDisplayMediaRequestHandler. Desktop streams have empty mediaTypes;
    // physical microphone/camera requests contain `audio` and/or `video`.
    // Allow only the pending desktop request, never blanket media access.
    const desktopMedia = permission === 'media' && Array.isArray(details?.mediaTypes) && details.mediaTypes.length === 0;
    callback(pendingDesktopCapture(contents, details) && (permission === 'display-capture' || desktopMedia));
  });
  captureSession.setPermissionCheckHandler((contents, permission, _origin, details) =>
    pendingDesktopCapture(contents, details) &&
    (permission === 'display-capture' || (permission === 'media' && details?.mediaType === 'unknown')));
  captureSession.setDisplayMediaRequestHandler(async (request, callback) => {
    const choice = captureChoice;
    const requestGeneration = captureGeneration;
    captureChoice = null;
    if (!choice || request.frame !== mainWindow?.webContents.mainFrame || !request.userGesture) return callback({});
    try {
      const available = await desktopCapturer.getSources({ types: ['screen', 'window'], thumbnailSize: { width: 0, height: 0 } });
      const source = available.find(item => item.id === choice.id);
      if (!source || requestGeneration !== captureGeneration || !active || state.mode !== 'starting') return callback({});
      callback({ video: source, ...(choice.audio && request.audioRequested ? { audio: 'loopback' } : {}) });
    } catch { callback({}); }
  });

  ipcMain.handle('sources:list', async event => {
    guard(event);
    const available = await desktopCapturer.getSources({ types: ['window', 'screen'], thumbnailSize: { width: 320, height: 180 } });
    const filtered = available.filter(source => !source.name.startsWith('MSAS'));
    sources = new Map(filtered.map(source => [source.id, source]));
    return filtered.map(source => ({ id: source.id, name: source.name, thumbnail: source.thumbnail.toDataURL(), displayId: source.display_id }));
  });
  ipcMain.handle('capture:prepare', (event, choice) => {
    guard(event);
    if (process.platform !== 'win32') throw new Error('This MSAS capture prototype supports Windows only.');
    if (active) throw new Error('Stop the current session first.');
    if (!choice || typeof choice.id !== 'string' || !sources.has(choice.id)) throw new Error('Refresh and select an available source.');
    const source = sources.get(choice.id);
    displayId = source.display_id || null;
    captureChoice = { id: choice.id, audio: choice.audio === true };
    state.source = source.name;
    startSession('starting');
  });
  ipcMain.handle('tab:start', async (event, audio) => {
    guard(event);
    if (active) throw new Error('Stop the current session first.');
    const id = ++captureGeneration;
    tabSession = id; displayId = null;
    state.source = 'Browser tab connection';
    startSession('starting');
    try {
      const bridge = await createTabBridge({
        audio: audio === true,
        onSignal: message => {
          if (tabSession === id && !mainWindow.isDestroyed()) mainWindow.webContents.send('tab:signal', { ...message, session: id });
        },
        onClose: message => {
          if (tabSession !== id) return;
          stopSession();
          if (mainWindow && !mainWindow.isDestroyed()) mainWindow.webContents.send('tab:error', message);
        }
      });
      if (tabSession !== id) { bridge.close(); throw new Error('Tab connection cancelled.'); }
      tabBridge = bridge;
      return { token: bridge.token, session: id };
    } catch (error) {
      if (tabSession === id) stopSession();
      if (error.code === 'EADDRINUSE') throw new Error('The local connection port is busy. Close other MSAS instances and try again.');
      throw error;
    }
  });
  ipcMain.handle('tab:answer', (event, message) => {
    guard(event);
    if (!tabBridge || message?.session !== tabSession) throw new Error('Tab connection expired.');
    tabBridge.send(message);
  });
  ipcMain.handle('capture:ready', (event, audio) => {
    guard(event);
    if (state.mode !== 'starting') return;
    state = { ...state, mode: 'live', audio: audio === true };
    createOverlays();
    sendState();
  });
  ipcMain.handle('analysis:start', async event => {
    guard(event);
    if (state.mode !== 'live') throw new Error('Start a call capture before analysis.');
    if (!aiBridge) {
      aiBridge = new AiBridge(path.join(__dirname, '..'), message => {
        lastAnalysisEvent = message;
        if (analysisWindow && !analysisWindow.isDestroyed()) analysisWindow.webContents.send('analysis:event', message);
      });
      await aiBridge.start();
    }
    return { ok: true };
  });
  ipcMain.on('analysis:audio', (event, bytes) => {
    if (!trustedMain(event) || !aiBridge) return;
    try { aiBridge.sendAudio(bytes); } catch { /* malformed renderer input is ignored */ }
  });
  ipcMain.on('analysis:frame', (event, bytes) => {
    if (!trustedMain(event) || !aiBridge) return;
    try { aiBridge.sendFrame(bytes); } catch { /* malformed renderer input is ignored */ }
  });
  ipcMain.on('analysis:face', (event, reading) => {
    if (!trustedMain(event) || !reading || typeof reading !== 'object') return;
    const allowed = ['joy', 'surprise', 'sadness', 'anger', 'disgust', 'fear', 'neutral'];
    const values = Object.fromEntries(allowed.map(name => [name, Math.max(0, Math.min(1, Number(reading.values?.[name]) || 0))]));
    const message = { type: 'face', values, status: String(reading.status || 'uncertain').slice(0, 30) };
    if (analysisWindow && !analysisWindow.isDestroyed()) analysisWindow.webContents.send('analysis:event', message);
  });
  ipcMain.on('analysis:input', (event, input) => {
    if (!trustedMain(event) || !input || typeof input !== 'object') return;
    const message = { type: 'input', audio: input.audio === true, face: input.face === true };
    if (analysisWindow && !analysisWindow.isDestroyed()) analysisWindow.webContents.send('analysis:event', message);
  });
  ipcMain.handle('analysis:resize', (event, expanded) => {
    if (!trustedAnalysis(event) || !analysisWindow || analysisWindow.isDestroyed()) throw new Error('Unauthorized request');
    const bounds = analysisWindow.getBounds();
    analysisWindow.setBounds({ ...bounds, width: 410, height: expanded === false ? 88 : 620 });
  });
  ipcMain.handle('session:stop', event => { guard(event, true); stopSession(); });
  ipcMain.handle('demo:start', event => {
    guard(event);
    if (active) throw new Error('Stop the current session first.');
    demoIndex = 0; displayId = null;
    startSession('demo');
    state.cue = demoCues[demoIndex]; sendState();
  });
  ipcMain.handle('demo:next', event => {
    guard(event, true);
    if (state.mode !== 'demo') return;
    demoIndex = (demoIndex + 1) % demoCues.length;
    state.cue = demoCues[demoIndex]; sendState();
  });
  ipcMain.handle('session:state', event => { guard(event, true); return state; });
  ipcMain.handle('overlay:expand', (event, expanded) => { guard(event, true); positionOverlays(expanded === true); });
  ipcMain.handle('window:show', event => { guard(event, true); mainWindow.show(); mainWindow.focus(); });
  ipcMain.handle('window:minimize', event => { guard(event); mainWindow.minimize(); });

  mainWindow = secureWindow({ title: 'MSAS · Conversation companion', width: 1240, height: 880,
    minWidth: 960, minHeight: 720, backgroundColor: '#f7f8f5', show: false });
  mainWindow.loadURL('msas://app/index.html');
  mainWindow.once('ready-to-show', () => mainWindow.show());
  mainWindow.on('closed', () => { stopSession(); app.quit(); });
  mainWindow.webContents.on('render-process-gone', stopSession);
  mainWindow.webContents.on('did-start-loading', () => { if (active) stopSession(); });
  screen.on('display-removed', stopSession);
  screen.on('display-metrics-changed', () => { if (active) positionOverlays(); });
});
app.on('window-all-closed', () => app.quit());
app.on('before-quit', stopSession);
