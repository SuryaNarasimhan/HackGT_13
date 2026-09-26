const status = document.getElementById('status');
chrome.runtime.sendMessage({ target: 'background', type: 'status' }).then(result => { if (result?.status) status.textContent = result.status; }).catch(() => {});
document.getElementById('connect').addEventListener('click', async () => {
  const token = document.getElementById('code').value.trim();
  if (!/^[a-f0-9]{48}$/.test(token)) { status.textContent = 'Paste the complete pairing code from MSAS.'; return; }
  const button = document.getElementById('connect'); button.disabled = true;
  status.textContent = 'Connecting…';
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    const result = await chrome.runtime.sendMessage({ target: 'background', type: 'start', token, tabId: tab.id });
    status.textContent = result.error || 'Connecting to MSAS. Wait for its live preview, then minimize Edge.';
    if (!result.error) document.getElementById('code').value = '';
  } catch { status.textContent = 'Connection failed. Open MSAS and generate a new pairing code.'; }
  finally { button.disabled = false; }
});
document.getElementById('stop').addEventListener('click', async () => {
  await chrome.runtime.sendMessage({ target: 'background', type: 'stop' });
  status.textContent = 'Sharing stopped.';
});
