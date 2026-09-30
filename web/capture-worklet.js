// Microphone capture: resamples to 16 kHz mono and posts 512-sample int16 frames.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.phase = 0;
    this.acc = 0;
    this.accN = 0;
    this.frame = new Int16Array(512);
    this.n = 0;
    this.energy = 0;
  }

  push(v) {
    this.energy += v * v;
    this.frame[this.n++] = Math.max(-32768, Math.min(32767, Math.round(v * 32767)));
    if (this.n === this.frame.length) {
      const level = Math.sqrt(this.energy / this.n);
      this.port.postMessage({ pcm: this.frame.buffer, level }, [this.frame.buffer]);
      this.frame = new Int16Array(512);
      this.n = 0;
      this.energy = 0;
    }
  }

  process(inputs) {
    const input = inputs[0] && inputs[0][0];
    if (!input) return true;
    if (this.ratio === 1) {
      for (let i = 0; i < input.length; i++) this.push(input[i]);
      return true;
    }
    // Box-filter decimation: average every input sample that falls inside one
    // output period (a cheap low-pass that is plenty for speech recognition).
    for (let i = 0; i < input.length; i++) {
      this.acc += input[i];
      this.accN++;
      this.phase += 1;
      if (this.phase >= this.ratio) {
        this.phase -= this.ratio;
        this.push(this.acc / this.accN);
        this.acc = 0;
        this.accN = 0;
      }
    }
    return true;
  }
}

registerProcessor("capture", CaptureProcessor);
