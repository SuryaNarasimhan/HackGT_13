(() => {
  function resample(input, sourceRate, targetRate = 16000) {
    if (sourceRate === targetRate) return input;
    const ratio = sourceRate / targetRate;
    const output = new Float32Array(Math.max(1, Math.floor(input.length / ratio)));
    for (let index = 0; index < output.length; index++) {
      const position = index * ratio;
      const left = Math.floor(position);
      const right = Math.min(input.length - 1, left + 1);
      const mix = position - left;
      output[index] = input[left] * (1 - mix) + input[right] * mix;
    }
    return output;
  }

  function create(video, faceAnalysis, api) {
    let active = false, context = null, source = null, processor = null, frameTimer = null;
    let buffered = new Float32Array(0), framePending = false, lastInput = '';
    const canvas = document.createElement('canvas');
    const paint = canvas.getContext('2d', { alpha: false });
    const faceNames = { happy: 'joy', surprise: 'surprise', sad: 'sadness', angry: 'anger', disgust: 'disgust', fear: 'fear', neutral: 'neutral' };
    window.addEventListener('msas:face-result', event => {
      if (!active) return;
      const face = event.detail?.faces?.length === 1 ? event.detail.faces[0] : null;
      const values = Object.fromEntries(Object.values(faceNames).map(name => [name, 0]));
      for (const item of face?.emotion || []) {
        const name = faceNames[item.emotion];
        if (name && Number.isFinite(item.score)) values[name] = item.score;
      }
      api.sendAnalysisFace({ values, status: event.detail?.status || 'uncertain' });
    });

    function reportInput(audio, face) {
      const key = `${audio}:${face}`;
      if (key !== lastInput) { lastInput = key; api.setAnalysisInput({ audio, face }); }
    }
    function pushSamples(samples, rate) {
      const converted = resample(samples, rate);
      const joined = new Float32Array(buffered.length + converted.length);
      joined.set(buffered); joined.set(converted, buffered.length); buffered = joined;
      while (buffered.length >= 512) {
        const chunk = buffered.slice(0, 512); buffered = buffered.slice(512);
        api.sendAnalysisAudio(chunk.buffer);
      }
    }
    async function startAudio(media) {
      if (!media.getAudioTracks().length) return false;
      context = new AudioContext({ latencyHint: 'interactive', sampleRate: 16000 });
      await context.audioWorklet.addModule('./audio-worklet.js');
      source = context.createMediaStreamSource(new MediaStream(media.getAudioTracks()));
      processor = new AudioWorkletNode(context, 'msas-audio-processor');
      const silent = context.createGain(); silent.gain.value = 0;
      processor.port.onmessage = event => {
        if (active && event.data instanceof ArrayBuffer) pushSamples(new Float32Array(event.data), context.sampleRate);
      };
      source.connect(processor); processor.connect(silent); silent.connect(context.destination);
      if (context.state === 'suspended') await context.resume();
      return true;
    }
    function sendFrame() {
      if (!active) return;
      const region = faceAnalysis.getRegion();
      const hasFace = !!region;
      reportInput(!!context, hasFace);
      if (!hasFace || framePending || video.readyState < 2 || !video.videoWidth) return;
      const width = region.width * video.videoWidth, height = region.height * video.videoHeight;
      const scale = Math.min(1, 480 / Math.max(width, height));
      canvas.width = Math.max(1, Math.round(width * scale));
      canvas.height = Math.max(1, Math.round(height * scale));
      try {
        paint.drawImage(video, region.x * video.videoWidth, region.y * video.videoHeight, width, height, 0, 0, canvas.width, canvas.height);
        framePending = true;
        canvas.toBlob(async blob => {
          try { if (active && blob) api.sendAnalysisFrame(await blob.arrayBuffer()); }
          finally { framePending = false; }
        }, 'image/jpeg', .72);
      } catch { framePending = false; }
    }
    async function start(media) {
      stop(); active = true; lastInput = '';
      await api.startAnalysis();
      let audio = false;
      try { audio = await startAudio(media); } catch { audio = false; }
      reportInput(audio, !!faceAnalysis.getRegion());
      frameTimer = setInterval(sendFrame, 1000);
    }
    function stop() {
      active = false; clearInterval(frameTimer); frameTimer = null; framePending = false;
      processor?.disconnect(); source?.disconnect(); processor = source = null;
      if (context) context.close().catch(() => {});
      context = null; buffered = new Float32Array(0); lastInput = '';
    }
    return { start, stop };
  }

  window.LiveAnalysis = { create, resample };
})();
