(() => {
  const $ = id => document.getElementById(id);
  const api = window.msas;
  let expanded = true;
  const statusText = { warming_up:'Learning this speaker', usual:'Near speaker baseline', used:'Compared with baseline', no_face:'No clear face', multiple:'Multiple faces', uncertain:'Uncertain', too_quiet:'Too quiet', no_words:'No words', unavailable:'Unavailable' };

  function topPrediction(values) {
    const entries = Object.entries(values || {}).filter(([, value]) => Number.isFinite(value));
    if (!entries.length || entries.every(([, value]) => value <= 0)) return { label:'Waiting', score:'—' };
    const [label, score] = entries.sort((a, b) => b[1] - a[1])[0];
    return { label, score:`${Math.round(score * 100)}%` };
  }
  function channel(name, values, state) {
    const top = topPrediction(values);
    $(`${name}-label`).textContent = state && state !== 'used' ? statusText[state] || top.label : top.label;
    $(`${name}-score`).textContent = top.score;
    $(`${name}-score`).title = state ? statusText[state] || state : 'Model score';
  }
  function handle(event) {
    if (!event) return;
    if (event.type === 'booting') $('status').textContent = 'Starting local analysis…';
    if (event.type === 'ready') { $('status').textContent = event.gemini ? 'Local signals · Gemini explanations' : 'Local signals · private fallback'; $('pulse').classList.add('live'); }
    if (event.type === 'status') $('status').textContent = event.status || 'Analyzing conversation';
    if (event.type === 'speech') { $('speech').textContent = event.active ? 'Speech detected' : 'Listening'; $('pulse').classList.toggle('live', event.active); }
    if (event.type === 'input') {
      $('audio-input').classList.toggle('on', event.audio); $('face-input').classList.toggle('on', event.face);
      $('audio-input').textContent = `${event.audio ? '●' : '○'} ${event.audio ? 'Call audio connected' : 'Enable call audio'}`;
      $('face-input').textContent = `${event.face ? '●' : '○'} ${event.face ? 'Participant selected' : 'Select participant'}`;
    }
    if (event.type === 'error') { $('status').textContent = event.message || 'Analysis unavailable'; $('pulse').classList.remove('live'); }
    if (event.type === 'face') channel('face', event.values, event.status === 'expression' ? 'used' : event.status.replace('-', '_'));
    if (event.type === 'result') {
      channel('face', event.channels?.face, event.channelStatus?.face);
      channel('tone', event.channels?.tone, event.channelStatus?.tone);
      channel('words', event.channels?.words, event.channelStatus?.words);
      $('transcript').textContent = event.transcript || 'No clear words detected.';
      $('cue-type').textContent = event.cue?.type || 'Likely meaning';
      $('cue-confidence').textContent = `${event.cue?.confidence || 'Low'} confidence`;
      $('explanation').textContent = event.cue?.explanation || 'Not enough evidence yet.';
      $('suggestion').textContent = event.cue?.suggestion || 'Keep listening for context.';
      $('source').textContent = event.cue?.source || 'Local analysis';
      $('jsd').textContent = `Signal difference ${Math.round((event.jsd || 0) * 100)}%`;
    }
  }
  $('collapse').addEventListener('click', async () => {
    expanded = !expanded; $('content').hidden = !expanded; $('collapse').textContent = expanded ? '−' : '+';
    await api.resizeAnalysis(expanded);
  });
  api.onAnalysisEvent(handle);
})();
