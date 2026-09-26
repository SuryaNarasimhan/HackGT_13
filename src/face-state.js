(function (root) {
  const labels = { happy: 'Appears happy', neutral: 'Neutral expression', surprise: 'Appears surprised', sad: 'Appears sad', angry: 'Appears angry', fear: 'Appears fearful', disgust: 'Appears disgusted' };
  function containedRect(width, height, containerWidth, containerHeight) {
    if (![width, height, containerWidth, containerHeight].every(v => Number.isFinite(v) && v > 0)) return null;
    const scale = Math.min(containerWidth / width, containerHeight / height);
    return { x: (containerWidth - width * scale) / 2, y: (containerHeight - height * scale) / 2, width: width * scale, height: height * scale };
  }
  function overlap(a, b) {
    const intersection = Math.max(0, Math.min(a[0] + a[2], b[0] + b[2]) - Math.max(a[0], b[0])) * Math.max(0, Math.min(a[1] + a[3], b[1] + b[3]) - Math.max(a[1], b[1]));
    return intersection / (a[2] * a[3] + b[2] * b[3] - intersection || 1);
  }
  class ExpressionState {
    constructor() { this.reset(); }
    reset() { this.average = null; this.candidate = null; this.count = 0; this.lastBox = null; this.lastTime = null; }
    unavailable(status, text) { this.reset(); return { status, text }; }
    update(faces, timestamp) {
      if (!Array.isArray(faces) || faces.length === 0) return this.unavailable('no-face', 'No face detected');
      if (faces.length !== 1) return this.unavailable('multiple', 'Select just one participant');
      const face = faces[0];
      if (!Array.isArray(face.box) || face.box.length !== 4 || !face.box.every(Number.isFinite) ||
          !Number.isFinite(face.boxScore) || !Number.isFinite(face.faceScore) ||
          face.boxScore < .65 || face.faceScore < .6 || Math.min(face.box[2], face.box[3]) < 64) {
        return this.unavailable('uncertain', 'Uncertain · face is unclear or too small');
      }
      if (!Number.isFinite(timestamp)) return this.unavailable('uncertain', 'Uncertain');
      if (this.lastTime !== null && (timestamp - this.lastTime > 2500 || timestamp < this.lastTime)) this.reset();
      if (this.lastBox && overlap(this.lastBox, face.box) < .25) this.reset();
      this.lastBox = [...face.box]; this.lastTime = timestamp;
      const scores = Object.fromEntries(Object.keys(labels).map(key => [key, 0]));
      for (const item of face.emotion || []) {
        if (Object.hasOwn(scores, item.emotion) && Number.isFinite(item.score) && item.score >= 0 && item.score <= 1) scores[item.emotion] = item.score;
      }
      if (Object.values(scores).reduce((a, b) => a + b, 0) < .5) return this.unavailable('uncertain', 'Uncertain');
      this.average = Object.fromEntries(Object.entries(scores).map(([key, value]) => [key, this.average ? .4 * value + .6 * this.average[key] : value]));
      const ranked = Object.entries(this.average).sort((a, b) => b[1] - a[1]);
      const [category, score] = ranked[0];
      if (score < .65 || score - ranked[1][1] < .15) {
        this.candidate = null; this.count = 0;
        return { status: 'uncertain', text: 'Uncertain' };
      }
      this.count = this.candidate === category ? this.count + 1 : 1;
      this.candidate = category;
      if (this.count < 3) return { status: 'uncertain', text: 'Uncertain · checking expression' };
      // These are model scores, not calibrated probabilities of a person's feelings.
      return { status: 'expression', category, text: labels[category] };
    }
  }
  const api = { ExpressionState, containedRect, overlap };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.MsasFaceState = api;
})(globalThis);
