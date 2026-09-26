const $ = id => document.getElementById(id);
const border = new URLSearchParams(location.search).get('role') === 'border';
let expanded = false;
$(border ? 'border' : 'controls').hidden = false;
function render(state) {
  document.body.dataset.cue = state.cue?.category || 'active';
  $('label').textContent = state.cue?.label || (state.mode === 'starting' ? 'Connecting capture…' : 'Capture active');
  $('mode').textContent = state.mode === 'demo' ? 'SIMULATED · NO CALL CAPTURE' : 'AI NOT CONNECTED · LOCAL CAPTURE';
  $('quote').textContent = state.cue?.quote || '';
  $('meaning').textContent = state.cue?.meaning || 'Your selected source is being previewed locally. Live transcription and AI interpretation are not connected yet.';
  $('evidence').textContent = state.cue?.evidence || '';
  $('alternative').textContent = state.cue ? `Another possibility: ${state.cue.alternative}` : '';
  document.querySelector('details').hidden = !state.cue;
  $('next').hidden = state.mode !== 'demo';
}
window.msas.onState(render);
// The passive border receives state events but has no privileged UI actions.
if (!border) {
  window.msas.getState().then(render);
  $('expand').addEventListener('click', () => {
    expanded = !expanded; $('details').hidden = !expanded;
    $('expand').setAttribute('aria-expanded', String(expanded)); window.msas.expand(expanded);
  });
  $('stop').addEventListener('click', () => window.msas.stop());
  $('next').addEventListener('click', () => window.msas.nextDemo());
  $('open').addEventListener('click', () => window.msas.showMain());
}
