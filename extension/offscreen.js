let current = null, statusText = 'Not connected.';
const endpoint = 'http://127.0.0.1:47831';
async function request(session, path, data) {
  const response = await fetch(endpoint + path, {
    method: data ? 'POST' : 'GET', headers: { Authorization: `Bearer ${session.token}`, 'X-Msas-Client': chrome.runtime.id, ...(data ? { 'Content-Type': 'application/json' } : {}) },
    ...(data ? { body: JSON.stringify(data) } : {}), signal: AbortSignal.timeout(4000)
  });
  if (!response.ok) throw new Error('MSAS disconnected. Generate a new pairing code.');
  return response.json();
}
function stop(session, message = 'Sharing stopped.', notify = true) {
  if (!session || current !== session) return;
  current = null; statusText = message;
  clearTimeout(session.timer); clearTimeout(session.deadline); clearTimeout(session.disconnected);
  session.stream?.getTracks().forEach(track => { track.onended = null; track.stop(); });
  if (session.peer) { session.peer.onconnectionstatechange = null; session.peer.close(); }
  session.audio?.close().catch(() => {});
  if (notify) request(session, '/signal', { type: 'stop' }).catch(() => {});
}
async function poll(session) {
  if (current !== session) return;
  try {
    const { answer } = await request(session, '/answer');
    if (current !== session) return;
    if (answer && !session.answered) {
      session.answered = true;
      await session.peer.setRemoteDescription(answer);
    }
    if (current === session) session.timer = setTimeout(() => poll(session), 1000);
  } catch { stop(session, 'MSAS stopped or disconnected. Generate a new code to reconnect.', false); }
}
async function start(message) {
  if (current) throw new Error('Already sharing a tab. Stop first.');
  const session = current = { token: message.token };
  statusText = 'Connecting…';
  session.deadline = setTimeout(() => stop(session, 'Connection timed out. Reconnect from MSAS.'), 25000);
  try {
    const mandatory = { chromeMediaSource: 'tab', chromeMediaSourceId: message.streamId };
    const stream = await navigator.mediaDevices.getUserMedia({ video: { mandatory: { ...mandatory, maxWidth: 1920, maxHeight: 1080, maxFrameRate: 15 } }, audio: message.audio ? { mandatory } : false });
    if (current !== session) { stream.getTracks().forEach(track => track.stop()); return; }
    session.stream = stream;
    stream.getVideoTracks()[0].onended = () => stop(session, 'The call tab stopped sharing.');
    if (message.audio && stream.getAudioTracks().length) {
      // Tab capture redirects sound. Restore it here; the MSAS preview is muted.
      session.audio = new AudioContext();
      session.audio.createMediaStreamSource(stream).connect(session.audio.destination);
      await session.audio.resume();
      if (session.audio.state !== 'running') throw new Error('Browser audio is paused. Reconnect with call audio disabled in MSAS.');
    }
    if (current !== session) return;
    const pc = session.peer = new RTCPeerConnection({ iceServers: [] });
    for (const track of stream.getTracks()) pc.addTrack(track, stream);
    pc.onconnectionstatechange = () => {
      if (pc.connectionState === 'connected') { clearTimeout(session.deadline); clearTimeout(session.disconnected); statusText = 'Sharing this tab with MSAS. You can minimize the browser.'; }
      if (pc.connectionState === 'failed') stop(session, 'Video connection failed. Reconnect from MSAS.');
      if (pc.connectionState === 'disconnected') session.disconnected = setTimeout(() => stop(session, 'MSAS disconnected.'), 5000);
    };
    await pc.setLocalDescription(await pc.createOffer());
    await new Promise((resolve, reject) => {
      if (pc.iceGatheringState === 'complete') return resolve();
      const limit = setTimeout(() => { pc.removeEventListener('icegatheringstatechange', changed); reject(new Error('Local connection setup timed out.')); }, 8000);
      function changed() { if (pc.iceGatheringState === 'complete') { clearTimeout(limit); pc.removeEventListener('icegatheringstatechange', changed); resolve(); } }
      pc.addEventListener('icegatheringstatechange', changed);
    });
    if (current !== session) return;
    await request(session, '/signal', { type: 'offer', sdp: pc.localDescription.sdp });
    poll(session);
  } catch (error) { stop(session, error.message); }
}
chrome.runtime.onMessage.addListener((message, sender, respond) => {
  if (sender.id !== chrome.runtime.id || message.target !== 'offscreen') return;
  if (message.type === 'status') { respond({ active: !!current, status: statusText }); return; }
  if (message.type === 'stop') { stop(current); respond({ ok: true }); return; }
  if (message.type === 'start') {
    if (current) { respond({ error: 'Already sharing. Stop first.' }); return; }
    start(message); respond({ ok: true });
  }
});
