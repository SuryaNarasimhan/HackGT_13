const { contextBridge, ipcRenderer } = require('electron');
const subscribe = (channel, listener) => {
  const handler = (_event, value) => listener(value);
  ipcRenderer.on(channel, handler);
  return () => ipcRenderer.removeListener(channel, handler);
};
contextBridge.exposeInMainWorld('msas', {
  platform: process.platform,
  listSources: () => ipcRenderer.invoke('sources:list'),
  prepareCapture: choice => ipcRenderer.invoke('capture:prepare', choice),
  captureReady: audio => ipcRenderer.invoke('capture:ready', audio),
  startTabCapture: audio => ipcRenderer.invoke('tab:start', audio),
  tabAnswer: message => ipcRenderer.invoke('tab:answer', message),
  onTabSignal: listener => subscribe('tab:signal', listener),
  onTabError: listener => subscribe('tab:error', listener),
  stop: () => ipcRenderer.invoke('session:stop'),
  startDemo: () => ipcRenderer.invoke('demo:start'),
  nextDemo: () => ipcRenderer.invoke('demo:next'),
  getState: () => ipcRenderer.invoke('session:state'),
  expand: expanded => ipcRenderer.invoke('overlay:expand', expanded),
  showMain: () => ipcRenderer.invoke('window:show'),
  minimize: () => ipcRenderer.invoke('window:minimize'),
  onState: listener => subscribe('session:state', listener),
  onStop: listener => subscribe('capture:stop', listener)
});
