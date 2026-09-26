const { test } = require('node:test');
const assert = require('node:assert/strict');
const { ExpressionState, containedRect } = require('../src/face-state.js');
const face = (emotion = [{ emotion: 'happy', score: .9 }, { emotion: 'neutral', score: .08 }]) => ({ box: [30, 20, 150, 180], boxScore: .9, faceScore: .9, emotion });
test('a clear expression needs three consistent observations; no probability is exposed', () => {
  const state = new ExpressionState();
  assert.equal(state.update([face()], 0).status, 'uncertain');
  assert.equal(state.update([face()], 500).status, 'uncertain');
  assert.deepEqual(state.update([face()], 1000), { status: 'expression', category: 'happy', text: 'Appears happy' });
});
test('face disappearance and multiple faces immediately clear the previous label', () => {
  const state = new ExpressionState(); [0, 500, 1000].forEach(t => state.update([face()], t));
  assert.equal(state.update([], 1500).status, 'no-face');
  assert.equal(state.update([face()], 2000).status, 'uncertain');
  assert.equal(state.update([face(), face()], 2500).status, 'multiple');
});
test('weak, ambiguous, tiny and malformed observations abstain', () => {
  for (const value of [face([{ emotion: 'happy', score: .48 }, { emotion: 'sad', score: .47 }]), { ...face(), boxScore: .2 }, { ...face(), faceScore: NaN }, { ...face(), box: [0, 0, 20, 20] }, face([{ emotion: 'happy', score: NaN }])]) {
    const state = new ExpressionState();
    for (let t = 0; t < 5; t++) assert.equal(state.update([value], t * 500).status, 'uncertain');
  }
});
test('a time gap or face location jump requires fresh stabilization', () => {
  const state = new ExpressionState(); [0, 500, 1000].forEach(t => state.update([face()], t));
  assert.equal(state.update([face()], 5000).status, 'uncertain');
  state.update([face()], 5500); state.update([face()], 6000);
  assert.equal(state.update([{ ...face(), box: [320, 300, 150, 180] }], 6500).status, 'uncertain');
});
test('participant selection accounts for letterboxing', () => {
  assert.deepEqual(containedRect(1920, 1080, 400, 400), { x: 0, y: 87.5, width: 400, height: 225 });
  assert.equal(containedRect(0, 0, 400, 400), null);
});
