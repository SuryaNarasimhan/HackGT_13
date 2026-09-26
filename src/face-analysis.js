(function () {
  const $ = id => document.getElementById(id);
  const { ExpressionState, containedRect } = window.MsasFaceState;
  window.FaceAnalysis = {
    create(video) {
      const canvas = $('face-selection'), context = canvas.getContext('2d');
      const crop = document.createElement('canvas'), cropContext = crop.getContext('2d', { willReadFrequently: true });
      const smoother = new ExpressionState();
      let live = false, selecting = false, region = null, draft = null, anchor = null;
      let worker = null, timer = null, watchdog = null, busy = false, mesh = [];
      let sourceWidth = 0, sourceHeight = 0, lastVideoTime = -1, lastFrameTime = 0;
      function show(text, state = 'idle') { $('expression-result').textContent = text; $('expression-result').dataset.state = state; }
      function rect() { return containedRect(video.videoWidth, video.videoHeight, canvas.width, canvas.height); }
      function draw() {
        const bounds = $('preview').getBoundingClientRect();
        canvas.width = Math.round(bounds.width); canvas.height = Math.round(bounds.height);
        const area = rect(), selection = selecting ? draft : region;
        context.clearRect(0, 0, canvas.width, canvas.height);
        if (!area || !selection) return;
        const box = { x: area.x + selection.x * area.width, y: area.y + selection.y * area.height, width: selection.width * area.width, height: selection.height * area.height };
        context.strokeStyle = '#8dd9b2'; context.lineWidth = 2;
        context.strokeRect(box.x, box.y, box.width, box.height);
        if (!selecting && $('show-landmarks').checked) {
          context.fillStyle = '#b2f5cd';
          for (const point of mesh) {
            if (point[0] < 0 || point[0] > 1 || point[1] < 0 || point[1] > 1) continue;
            context.beginPath(); context.arc(box.x + point[0] * box.width, box.y + point[1] * box.height, 1, 0, 2 * Math.PI); context.fill();
          }
        }
      }
      function shutdown() {
        clearTimeout(timer); clearTimeout(watchdog); timer = watchdog = null;
        worker?.terminate(); worker = null; busy = false;
        mesh = []; smoother.reset(); crop.width = crop.height = 0;
      }
      function clear(text = 'Select a participant to analyze') {
        shutdown(); region = draft = anchor = null; selecting = false;
        canvas.dataset.selecting = 'false'; canvas.hidden = true;
        $('stop-analysis').disabled = true; $('select-participant').textContent = 'Select participant';
        $('face-help').textContent = 'In Meet, pin the other person. Select their video area here.';
        show(text); draw();
      }
      function fail(text) { shutdown(); mesh = []; show(text, 'error'); draw(); }
      function point(event) {
        const area = rect(), bounds = canvas.getBoundingClientRect();
        if (!area) return null;
        return { x: Math.max(0, Math.min(1, (event.clientX - bounds.left - area.x) / area.width)), y: Math.max(0, Math.min(1, (event.clientY - bounds.top - area.y) / area.height)) };
      }
      function analyze() {
        const instance = worker;
        if (!live || !region || !instance || busy) return;
        if (video.videoWidth !== sourceWidth || video.videoHeight !== sourceHeight) return clear('Call layout changed. Select the participant again.');
        if (video.readyState < 2 || video.currentTime === lastVideoTime) {
          if (performance.now() - lastFrameTime > 2000) { smoother.reset(); mesh = []; show('Waiting for live video', 'uncertain'); draw(); }
          timer = setTimeout(analyze, 250); return;
        }
        lastVideoTime = video.currentTime; lastFrameTime = performance.now();
        const width = region.width * video.videoWidth, height = region.height * video.videoHeight;
        const scale = Math.min(1, 480 / Math.max(width, height));
        crop.width = Math.max(1, Math.round(width * scale)); crop.height = Math.max(1, Math.round(height * scale));
        try {
          cropContext.drawImage(video, region.x * video.videoWidth, region.y * video.videoHeight, width, height, 0, 0, crop.width, crop.height);
          const frame = cropContext.getImageData(0, 0, crop.width, crop.height);
          busy = true;
          watchdog = setTimeout(() => { if (worker === instance) fail('Analysis timed out. Select participant to retry.'); }, 10000);
          instance.postMessage({ type: 'frame', pixels: frame.data.buffer, width: crop.width, height: crop.height, timestamp: performance.now() }, [frame.data.buffer]);
        } catch { fail('Could not read the selected area. Select participant to retry.'); }
      }
      function startWorker() {
        shutdown(); sourceWidth = video.videoWidth; sourceHeight = video.videoHeight;
        lastVideoTime = -1; lastFrameTime = performance.now();
        show('Loading local expression model…', 'loading');
        let instance;
        try { instance = new Worker('./face-worker.js'); }
        catch { fail('Could not start local analysis. Restart MSAS and try again.'); return; }
        worker = instance;
        watchdog = setTimeout(() => { if (worker === instance) fail('Model loading timed out. Select participant to retry.'); }, 30000);
        instance.onerror = () => { if (worker === instance) fail('Local model failed to load. Run npm install, then restart MSAS.'); };
        instance.onmessage = ({ data }) => {
          if (worker !== instance || !live || !region) return;
          clearTimeout(watchdog); watchdog = null;
          if (data.type === 'ready') { show('Looking for a face…', 'loading'); analyze(); }
          else if (data.type === 'error') fail(data.message);
          else if (data.type === 'result') {
            busy = false;
            const stale = performance.now() - data.timestamp > 2500;
            const result = stale ? smoother.unavailable('uncertain', 'Uncertain · analysis is too slow') : smoother.update(data.faces, data.timestamp);
            mesh = !stale && data.faces.length === 1 ? data.faces[0].mesh : [];
            show(result.text, result.status); draw();
            timer = setTimeout(analyze, 250);
          }
        };
        instance.postMessage({ type: 'init' });
      }
      function commit() {
        if (!draft || draft.width * video.videoWidth < 96 || draft.height * video.videoHeight < 96) {
          show('Select a larger area around the participant.', 'uncertain'); return;
        }
        region = { ...draft }; selecting = false; anchor = null;
        canvas.dataset.selecting = 'false'; $('stop-analysis').disabled = false;
        $('select-participant').textContent = 'Reselect participant';
        $('face-help').textContent = 'Keep the same person pinned. Reselect if Meet rearranges its tiles.';
        draw(); startWorker(); $('select-participant').focus();
      }
      $('select-participant').addEventListener('click', () => {
        if (!live) return;
        shutdown(); region = null; selecting = true; anchor = null;
        draft = { x: .1, y: .1, width: .8, height: .8 };
        canvas.hidden = false; canvas.dataset.selecting = 'true';
        $('stop-analysis').disabled = false;
        $('face-help').textContent = 'Drag around one participant. Keyboard: arrows move, Shift+arrows resize, Enter confirms, Esc cancels.';
        show('Select the other person’s video area'); draw(); canvas.focus();
      });
      canvas.addEventListener('pointerdown', event => {
        if (!selecting) return;
        event.preventDefault(); anchor = point(event); if (!anchor) return;
        canvas.setPointerCapture(event.pointerId); draft = { ...anchor, width: 0, height: 0 }; draw();
      });
      canvas.addEventListener('pointermove', event => {
        if (!selecting || !anchor) return;
        const end = point(event); if (!end) return;
        draft = { x: Math.min(anchor.x, end.x), y: Math.min(anchor.y, end.y), width: Math.abs(end.x - anchor.x), height: Math.abs(end.y - anchor.y) }; draw();
      });
      canvas.addEventListener('pointerup', event => {
        if (!selecting || !anchor) return;
        if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
        anchor = null; commit();
      });
      canvas.addEventListener('pointercancel', () => { anchor = null; });
      canvas.addEventListener('keydown', event => {
        if (!selecting) return;
        if (event.key === 'Escape') { event.preventDefault(); clear(); return; }
        if (event.key === 'Enter') { event.preventDefault(); commit(); return; }
        if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
        event.preventDefault();
        const dx = event.key === 'ArrowLeft' ? -.02 : event.key === 'ArrowRight' ? .02 : 0;
        const dy = event.key === 'ArrowUp' ? -.02 : event.key === 'ArrowDown' ? .02 : 0;
        if (event.shiftKey) { draft.width = Math.max(.1, Math.min(1 - draft.x, draft.width + dx)); draft.height = Math.max(.1, Math.min(1 - draft.y, draft.height + dy)); }
        else { draft.x = Math.max(0, Math.min(1 - draft.width, draft.x + dx)); draft.y = Math.max(0, Math.min(1 - draft.height, draft.y + dy)); }
        draw();
      });
      $('stop-analysis').addEventListener('click', () => clear('Analysis paused. Select participant to resume.'));
      $('show-landmarks').addEventListener('change', draw);
      new ResizeObserver(draw).observe($('preview'));
      return {
        setLive(value) {
          if (live === value) return;
          live = value; $('face-controls').hidden = !live;
          if (!live) { clear(); $('show-landmarks').checked = false; }
        },
        stop() { live = false; clear(); $('face-controls').hidden = true; $('show-landmarks').checked = false; }
      };
    }
  };
})();
