/* A mesh-gradient band, driven by the numbers in the note it sits under.
 *
 * The fragment shader is Paper Shaders' mesh gradient
 * (https://github.com/paper-design/shaders, Apache-2.0, (c) Paper Design), kept
 * verbatim; the vertex shader, the palette, the weights and the lifecycle
 * (visibility, reduced motion, DPR) are local.
 *
 * Markup contract, in _includes/mesh-band.html:
 *   <div class="mesh-band" data-colors="#91dbba,#c9b2ff,#93caff" data-weights="1.34,1.37,0.38">
 *     <canvas></canvas>
 *   </div>
 * weights become each spot's opacity, normalised against the largest: in these
 * pages a weight is a measured ratio from the note, so the brightest spot is the
 * largest one. The canvas stays hidden until a frame has been drawn, so a device
 * without WebGL 2 shows the CSS fallback instead of an empty box.
 */
(() => {
  "use strict";

  const FRAG = `#version 300 es
precision highp float;

uniform vec2 u_resolution;
uniform float u_time;
uniform vec4 u_colors[10];
uniform float u_colorsCount;
uniform float u_distortion;
uniform float u_swirl;
uniform float u_grainMixer;
uniform float u_grainOverlay;
uniform float u_intensity;
uniform float u_opacityGain;

in vec2 v_objectUV;
out vec4 fragColor;

vec2 rotate(vec2 uv, float th) {
  return mat2(cos(th), sin(th), -sin(th), cos(th)) * uv;
}

float hash21(vec2 p) {
  p = fract(p * vec2(0.3183099, 0.3678794)) + 0.1;
  p += dot(p, p + 19.19);
  return fract(p.x * p.y);
}

float valueNoise(vec2 st) {
  vec2 i = floor(st);
  vec2 f = fract(st);
  float a = hash21(i);
  float b = hash21(i + vec2(1.0, 0.0));
  float c = hash21(i + vec2(0.0, 1.0));
  float d = hash21(i + vec2(1.0, 1.0));
  vec2 u = f * f * (3.0 - 2.0 * f);
  float x1 = mix(a, b, u.x);
  float x2 = mix(c, d, u.x);
  return mix(x1, x2, u.y);
}

float noise(vec2 n, vec2 seedOffset) {
  return valueNoise(n + seedOffset);
}

vec2 getPosition(int i, float t) {
  float a = float(i) * .37;
  float b = .6 + fract(float(i) / 3.) * .9;
  float c = .8 + fract(float(i + 1) / 4.);

  float x = sin(t * b + a);
  float y = cos(t * c + a * 1.5);

  return .5 + .5 * vec2(x, y);
}

void main() {
  vec2 uv = v_objectUV;
  uv += .5;
  vec2 grainUV = uv * 1000.;

  float mixerGrain = 0.;
  if (u_grainMixer > 0.) {
    mixerGrain = .4 * u_grainMixer * (noise(grainUV, vec2(0.)) - .5);
  }

  const float firstFrameOffset = 41.5;
  float t = .5 * (u_time + firstFrameOffset);

  float radius = smoothstep(0., 1., length(uv - .5));
  float center = 1. - radius;
  for (float i = 1.; i <= 2.; i++) {
    uv.x += u_distortion * center / i * sin(t + i * .4 * smoothstep(.0, 1., uv.y)) * cos(.2 * t + i * 2.4 * smoothstep(.0, 1., uv.y));
    uv.y += u_distortion * center / i * cos(t + i * 2. * smoothstep(.0, 1., uv.x));
  }

  vec2 uvRotated = uv;
  uvRotated -= vec2(.5);
  float angle = 3. * u_swirl * radius;
  uvRotated = rotate(uvRotated, -angle);
  uvRotated += vec2(.5);

  vec3 color = vec3(0.);
  float opacity = 0.;
  float totalWeight = 0.;

  for (int i = 0; i < 10; i++) {
    if (i >= int(u_colorsCount)) break;

    vec2 pos = getPosition(i, t) + mixerGrain;
    vec3 colorFraction = u_colors[i].rgb * u_colors[i].a;
    float opacityFraction = u_colors[i].a;

    float dist = length(uvRotated - pos);

    dist = pow(dist, 3.5);
    float weight = 1. / (dist + 1e-3);
    color += colorFraction * weight;
    opacity += opacityFraction * weight;
    totalWeight += weight;
  }

  color /= max(1e-4, totalWeight);
  opacity /= max(1e-4, totalWeight);

  // Local, so a bright mesh sits on a dark page as a bloom rather than a wash.
  // Paper's shader averages the spot colours un-premultiplied; on the notebook's
  // #101217 backdrop that reads as white. These two knobs scale the result.
  color *= u_intensity;
  opacity *= u_opacityGain;

  if (u_grainOverlay > 0.) {
    float grainOverlay = valueNoise(rotate(grainUV, 1.) + vec2(3.));
    grainOverlay = mix(grainOverlay, valueNoise(rotate(grainUV, 2.) + vec2(-1.)), .5);
    grainOverlay = pow(grainOverlay, 1.3);

    float grainOverlayV = grainOverlay * 2. - 1.;
    vec3 grainOverlayColor = vec3(step(0., grainOverlayV));
    float grainOverlayStrength = u_grainOverlay * abs(grainOverlayV);
    grainOverlayStrength = pow(grainOverlayStrength, .8);
    color = mix(color, grainOverlayColor, .35 * grainOverlayStrength);

    opacity += .5 * grainOverlayStrength;
  }
  opacity = clamp(opacity, 0., 1.);

  fragColor = vec4(color, opacity);
}
`;

  const VERT = `#version 300 es
precision highp float;

uniform float u_aspect;
in vec2 a_position;
out vec2 v_objectUV;

void main() {
  v_objectUV = vec2(a_position.x * u_aspect, a_position.y);
  gl_Position = vec4(a_position * 2.0, 0.0, 1.0);
}
`;

  const BACKDROP = [0x10 / 255, 0x12 / 255, 0x17 / 255, 1.0];

  function rgb(hex) {
    const v = hex.trim().replace("#", "");
    return [
      parseInt(v.slice(0, 2), 16) / 255,
      parseInt(v.slice(2, 4), 16) / 255,
      parseInt(v.slice(4, 6), 16) / 255,
    ];
  }

  function compile(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(log || "shader failed to compile");
    }
    return shader;
  }

  function program(gl) {
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl, gl.VERTEX_SHADER, VERT));
    gl.attachShader(p, compile(gl, gl.FRAGMENT_SHADER, FRAG));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
      throw new Error(gl.getProgramInfoLog(p) || "program failed to link");
    }
    return p;
  }

  function start(band) {
    const canvas = band.querySelector("canvas");
    const gl = canvas.getContext("webgl2", {
      alpha: true,
      antialias: false,
      depth: false,
      powerPreference: "low-power",
      preserveDrawingBuffer: true, // lets a still be pulled out of the canvas
    });
    if (!gl) return;

    let prog;
    try {
      prog = program(gl);
    } catch (error) {
      console.warn("mesh-band:", error.message);
      return;
    }

    gl.useProgram(prog);

    const buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, 0.5]),
      gl.STATIC_DRAW
    );
    const aPosition = gl.getAttribLocation(prog, "a_position");
    gl.enableVertexAttribArray(aPosition);
    gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 0, 0);

    const colors = (band.dataset.colors || "#91dbba,#c9b2ff,#93caff")
      .split(",")
      .slice(0, 10)
      .map(rgb);
    const weights = (band.dataset.weights || "")
      .split(",")
      .map(Number)
      .filter((w) => Number.isFinite(w) && w > 0);
    const peak = weights.length ? Math.max(...weights) : 1;
    while (weights.length < colors.length) weights.push(peak);

    const spots = new Float32Array(40);
    colors.forEach((c, i) => {
      const alpha = Math.min(1, weights[i] / peak);
      spots.set([c[0], c[1], c[2], alpha], i * 4);
    });

    const u = {
      resolution: gl.getUniformLocation(prog, "u_resolution"),
      aspect: gl.getUniformLocation(prog, "u_aspect"),
      time: gl.getUniformLocation(prog, "u_time"),
      colors: gl.getUniformLocation(prog, "u_colors"),
      colorsCount: gl.getUniformLocation(prog, "u_colorsCount"),
      distortion: gl.getUniformLocation(prog, "u_distortion"),
      swirl: gl.getUniformLocation(prog, "u_swirl"),
      grainMixer: gl.getUniformLocation(prog, "u_grainMixer"),
      grainOverlay: gl.getUniformLocation(prog, "u_grainOverlay"),
      intensity: gl.getUniformLocation(prog, "u_intensity"),
      opacityGain: gl.getUniformLocation(prog, "u_opacityGain"),
    };

    gl.uniform4fv(u.colors, spots);
    gl.uniform1f(u.colorsCount, colors.length);
    gl.uniform1f(u.distortion, Number(band.dataset.distortion ?? 0.8));
    gl.uniform1f(u.swirl, Number(band.dataset.swirl ?? 0.55));
    gl.uniform1f(u.grainMixer, Number(band.dataset.grainMixer ?? 0.05));
    gl.uniform1f(u.grainOverlay, Number(band.dataset.grainOverlay ?? 0.04));
    gl.uniform1f(u.intensity, Number(band.dataset.intensity ?? 0.8));
    gl.uniform1f(u.opacityGain, Number(band.dataset.opacity ?? 0.6));
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(...BACKDROP);

    function resize() {
      const dpr = Math.min(window.devicePixelRatio || 1, 1.5);
      const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
      const height = Math.max(1, Math.round(canvas.clientHeight * dpr));
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      gl.viewport(0, 0, width, height);
      gl.uniform2f(u.resolution, width, height);
      gl.uniform1f(u.aspect, canvas.clientWidth / Math.max(1, canvas.clientHeight));
    }

    function frame(seconds) {
      resize();
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform1f(u.time, seconds);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }

    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)");
    let visible = true;
    let raf = 0;
    let t0 = 0;

    function loop(now) {
      if (!t0) t0 = now;
      frame((now - t0) / 1000);
      raf = requestAnimationFrame(loop);
    }

    function play() {
      if (raf || reduced.matches || !visible) return;
      raf = requestAnimationFrame(loop);
    }

    function pause() {
      if (!raf) return;
      cancelAnimationFrame(raf);
      raf = 0;
    }

    band.dataset.ready = "true";
    frame(0); // a valid first paint, before any rAF tick and regardless of visibility
    if (!reduced.matches) play();

    if ("IntersectionObserver" in window) {
      new IntersectionObserver(
        (entries) => {
          visible = entries.some((e) => e.isIntersecting);
          if (visible) play();
          else pause();
        },
        { rootMargin: "100px" }
      ).observe(band);
    }
    reduced.addEventListener("change", () => {
      if (reduced.matches) {
        pause();
        frame(0);
      } else play();
    });
    window.addEventListener("resize", () => {
      if (reduced.matches) frame(0);
    });
  }

  function boot() {
    document.querySelectorAll(".mesh-band").forEach(start);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
