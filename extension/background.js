let starting = false;
async function offscreenExists() {
  return (await chrome.runtime.getContexts({ contextTypes: ['OFFSCREEN_DOCUMENT'], documentUrls: [chrome.runtime.getURL('offscreen.html')] })).length > 0;
}
chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (sender.id !== chrome.runtime.id || message.target !== 'background') return;
  (async () => {
    if (message.type === 'status') {
      return await offscreenExists() ? chrome.runtime.sendMessage({ target: 'offscreen', type: 'status' }) : { status: 'Not connected. Paste a new code from MSAS.' };
    }
    if (message.type === 'stop') {
      if (await offscreenExists()) await chrome.runtime.sendMessage({ target: 'offscreen', type: 'stop' });
      return { ok: true };
    }
    if (message.type !== 'start' || starting || !/^[a-f0-9]{48}$/.test(message.token || '') || !Number.isInteger(message.tabId)) throw new Error('A connection is already starting or the code is invalid.');
    starting = true;
    try {
      // Validate the local endpoint before asking the browser for tab media.
      const response = await fetch('http://127.0.0.1:47831/config', { headers: { Authorization: `Bearer ${message.token}`, 'X-Msas-Client': chrome.runtime.id }, signal: AbortSignal.timeout(4000) });
      if (!response.ok) throw new Error('Pairing code expired or incorrect. Generate a new code in MSAS.');
      const { audio } = await response.json();
      if (!await offscreenExists()) await chrome.offscreen.createDocument({ url: 'offscreen.html', reasons: ['USER_MEDIA', 'WEB_RTC'], justification: 'Keep the user-selected call tab streaming locally to MSAS while the browser is minimized.' });
      const current = await chrome.runtime.sendMessage({ target: 'offscreen', type: 'status' });
      if (current.active) throw new Error('Stop the existing connection before starting another.');
      const streamId = await chrome.tabCapture.getMediaStreamId({ targetTabId: message.tabId });
      return await chrome.runtime.sendMessage({ target: 'offscreen', type: 'start', token: message.token, streamId, audio: audio === true });
    } finally { starting = false; }
  })().then(respond, error => respond({ error: error.message || 'Could not connect to MSAS. Start it and try again.' }));
  return true;
});
