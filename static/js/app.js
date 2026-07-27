const state = {
  file: null,
  documentId: null,
  document: null,
  history: [],
};

let activeRecognition = null;

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => document.querySelectorAll(selector);

const dropZone = $("#drop-zone");
const fileInput = $("#file-input");
const processButton = $("#process-button");

dropZone.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") fileInput.click();
});
fileInput.addEventListener("change", () => selectFile(fileInput.files[0]));

["dragenter", "dragover"].forEach((name) => {
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.add("dragging");
  });
});
["dragleave", "drop"].forEach((name) => {
  dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.remove("dragging");
  });
});
dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));

function selectFile(file) {
  hideError();
  if (!file || !file.name.toLowerCase().endsWith(".pdf")) {
    showError("Please select a PDF document.");
    return;
  }
  state.file = file;
  $("#selected-file").textContent = `${file.name} · ${formatBytes(file.size)}`;
  $("#selected-file").classList.remove("hidden");
  processButton.disabled = false;
}

processButton.addEventListener("click", async () => {
  if (!state.file) return;

  const formData = new FormData();
  formData.append("file", state.file);
  setProcessing(true);
  hideError();

  try {
    const response = await fetch("/api/documents", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Document processing failed.");
    state.documentId = data.document_id;
    state.document = data;
    renderDashboard(data);
  } catch (error) {
    showError(error.message);
  } finally {
    setProcessing(false);
  }
});

$("#new-document").addEventListener("click", () => window.location.reload());

$$(".tab").forEach((button) => {
  button.addEventListener("click", () => {
    $$(".tab").forEach((tab) => tab.classList.remove("active"));
    button.classList.add("active");
    $$(".tab-panel").forEach((panel) => panel.classList.add("hidden"));
    $(`#${button.dataset.tab}-tab`).classList.remove("hidden");
  });
});

function renderDashboard(data) {
  $("#upload-view").classList.add("hidden");
  $("#dashboard-view").classList.remove("hidden");
  $("#new-document").classList.remove("hidden");
  $("#document-name").textContent = data.filename;
  $("#document-meta").textContent =
    `${data.analytics.pages} pages · processed in ${data.processing_seconds}s`;

  const metrics = [
    ["Pages", data.analytics.pages],
    ["Words", formatNumber(data.analytics.words)],
    ["Text chunks", data.analytics.chunks],
    ["PDF images", data.analytics.images],
    ["Index time", `${data.processing_seconds}s`],
  ];
  $("#metric-grid").innerHTML = metrics.map(([label, value]) => `
    <article class="metric-card"><span>${escapeHtml(label)}</span><strong>${escapeHtml(value)}</strong></article>
  `).join("");

  $("#summary-text").textContent = data.overview.summary || "No summary is available.";
  $("#llm-state").textContent = data.llm_available ? "AI generated" : "Extractive preview";
  renderList("#findings-list", data.overview.key_findings, "Configure an LLM key to generate findings.");
  renderList("#risks-list", data.overview.risks, "Configure an LLM key to identify risks.");
  renderList("#opportunities-list", data.overview.opportunities, "Configure an LLM key to identify opportunities.");

  $("#term-list").innerHTML = data.analytics.top_terms.map((item) =>
    `<span class="term">${escapeHtml(item.term)} <b>${item.count}</b></span>`
  ).join("");

  const maxTopic = Math.max(...data.analytics.topics.map((item) => item.count), 1);
  $("#topic-chart").innerHTML = data.analytics.topics.map((item) => `
    <div class="bar-row">
      <span>${escapeHtml(item.topic)}</span>
      <div class="bar-track"><div class="bar-fill" style="width:${(item.count / maxTopic) * 100}%"></div></div>
      <b>${item.count}</b>
    </div>
  `).join("");

  const financialMetrics = data.analytics.financial_metrics;
  $("#metrics-list").innerHTML = financialMetrics.length
    ? financialMetrics.map((value) => `<span>${escapeHtml(value)}</span>`).join("")
    : `<p class="empty-copy">No obvious currency, percentage, or reporting-period markers were found.</p>`;

  renderVisualGallery(data.visual_pages || []);
}

function renderVisualGallery(visuals) {
  if (!visuals.length) return;
  $("#visuals-panel").classList.remove("hidden");
  $("#visual-gallery").innerHTML = visuals.map((visual) => `
    <button class="visual-card" data-image-url="${escapeHtml(visual.url)}" data-page="${visual.page}">
      <img src="${escapeHtml(visual.url)}" alt="Visual content on PDF page ${visual.page}" loading="lazy">
      <span>Page ${visual.page}</span>
    </button>
  `).join("");
  $$(".visual-card").forEach((card) => {
    card.addEventListener("click", () => openImage(card.dataset.imageUrl, `PDF page ${card.dataset.page}`));
  });
}

function renderList(selector, values, emptyMessage) {
  $(selector).innerHTML = values?.length
    ? values.map((value) => `<li>${escapeHtml(value)}</li>`).join("")
    : `<li class="empty-copy">${escapeHtml(emptyMessage)}</li>`;
}

$$(".suggestions button").forEach((button) => {
  button.addEventListener("click", () => {
    $("#question-input").value = button.textContent.trim();
    $("#question-input").focus();
  });
});

$("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();

  if (activeRecognition) {
    try { activeRecognition.stop(); } catch (e) {}
  }

  const input = $("#question-input");
  const question = input.value.trim();
  if (!question || !state.documentId) return;

  appendMessage("user", question);
  state.history.push({ role: "user", content: question });
  input.value = "";
  input.disabled = true;

  if (typeof avatar !== 'undefined' && avatar) {
    avatar.setState('thinking');
  }

  const loading = appendMessage("assistant", "Searching the document and checking the evidence…", [], true);

  try {
    const response = await fetch(`/api/documents/${state.documentId}/ask`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, history: state.history.slice(0, -1) }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "The question could not be answered.");
    loading.remove();
    appendMessage("assistant", data.answer, data.sources, false, data.visual_sources || []);
    state.history.push({ role: "assistant", content: data.answer });

    // Auto speak option
    const autoSpeak = $("#auto-speak-toggle")?.checked;
    if (autoSpeak) {
      speakText(data.answer, null);
    } else {
      if (typeof avatar !== 'undefined' && avatar) avatar.setState('idle');
    }
  } catch (error) {
    loading.remove();
    appendMessage("assistant", error.message);
    if (typeof avatar !== 'undefined' && avatar) avatar.setState('idle');
  } finally {
    input.disabled = false;
    input.focus();
  }
});

function appendMessage(role, content, sources = [], loading = false, visuals = []) {
  const wrapper = document.createElement("div");
  wrapper.className = `message ${role}${loading ? " loading" : ""}`;
  const sourceMarkup = sources.length ? `
    <div class="sources">
      ${sources.slice(0, 5).map((source) => `
        <div class="source">
          <strong>Page ${source.page}</strong> · chunk ${source.chunk_id}
          <div>${escapeHtml(source.excerpt)}${source.excerpt.length >= 420 ? "…" : ""}</div>
        </div>
      `).join("")}
    </div>` : "";
  const visualMarkup = visuals.length ? `
    <div class="visual-sources">
      ${visuals.map((visual) => `
        <button class="visual-source" data-image-url="${escapeHtml(visual.url)}" data-page="${visual.page}">
          <img src="${escapeHtml(visual.url)}" alt="PDF page ${visual.page}" loading="lazy">
          <span>Visual source · Page ${visual.page}</span>
        </button>
      `).join("")}
    </div>` : "";

  const isSpeechSupported = ('speechSynthesis' in window);
  const actionMarkup = (role === "assistant" && !loading && isSpeechSupported) ? `
    <div class="message-actions">
      <button class="tts-button" aria-label="Read aloud">
        <svg viewBox="0 0 24 24" width="16" height="16">
          <path fill="currentColor" d="M14,3.23V5.29C16.89,6.15 19,8.83 19,12C19,15.17 16.89,17.85 14,18.7V20.77C18,19.86 21,16.28 21,12C21,7.72 18,4.14 14,3.23M16.5,12C16.5,10.23 15.5,8.71 14,7.97V16C15.5,15.29 16.5,13.77 16.5,12M3,9V15H7L12,20V4L7,9H3Z" />
        </svg>
      </button>
    </div>` : "";

  wrapper.innerHTML = `<div class="message-bubble">${actionMarkup}${escapeHtml(content)}${visualMarkup}${sourceMarkup}</div>`;
  $("#chat-messages").appendChild(wrapper);

  if (role === "assistant" && !loading && isSpeechSupported) {
    const ttsBtn = wrapper.querySelector(".tts-button");
    if (ttsBtn) {
      ttsBtn.addEventListener("click", () => speakText(content, ttsBtn));
    }
  }

  wrapper.querySelectorAll(".visual-source").forEach((card) => {
    card.addEventListener("click", () => openImage(card.dataset.imageUrl, `PDF page ${card.dataset.page}`));
  });
  $("#chat-messages").scrollTop = $("#chat-messages").scrollHeight;
  return wrapper;
}

function setProcessing(active) {
  $("#processing-state").classList.toggle("hidden", !active);
  processButton.disabled = active || !state.file;
  dropZone.style.pointerEvents = active ? "none" : "auto";
}

function showError(message) {
  $("#upload-error").textContent = message;
  $("#upload-error").classList.remove("hidden");
}
function hideError() { $("#upload-error").classList.add("hidden"); }
function formatNumber(value) { return new Intl.NumberFormat().format(value); }
function formatBytes(bytes) {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  const index = Math.floor(Math.log(bytes) / Math.log(1024));
  return `${(bytes / Math.pow(1024, index)).toFixed(index ? 1 : 0)} ${units[index]}`;
}
function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function openImage(url, title) {
  $("#dialog-image").src = url;
  $("#dialog-title").textContent = title;
  $("#image-dialog").showModal();
}
$("#close-dialog").addEventListener("click", () => $("#image-dialog").close());
$("#image-dialog").addEventListener("click", (event) => {
  if (event.target === $("#image-dialog")) $("#image-dialog").close();
});

// Speech Recognition implementation
function initVoiceInput() {
  const micButton = $("#voice-input-btn");
  const questionInput = $("#question-input");
  
  if (!micButton || !questionInput) return;

  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {
    console.warn("Speech recognition is not supported in this browser.");
    return;
  }

  // Show microphone button if supported
  micButton.classList.remove("hidden");

  const recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = "en-US";

  let isListening = false;
  let originalPlaceholder = questionInput.placeholder;

  recognition.onstart = () => {
    isListening = true;
    activeRecognition = recognition;
    micButton.classList.add("listening");
    questionInput.placeholder = "Listening... Speak now.";
    questionInput.focus();
    if (typeof avatar !== 'undefined' && avatar) {
      avatar.setState('listening');
    }
  };

  recognition.onresult = (event) => {
    let resultText = "";
    for (let i = event.resultIndex; i < event.results.length; ++i) {
      resultText += event.results[i][0].transcript;
    }
    questionInput.value = resultText;
  };

  recognition.onerror = (event) => {
    console.error("Speech recognition error:", event.error);
    stopListening();
  };

  recognition.onend = () => {
    stopListening();
  };

  function startListening() {
    try {
      recognition.start();
    } catch (e) {
      console.error("Failed to start speech recognition:", e);
    }
  }

  function stopListening() {
    if (!isListening) return;
    isListening = false;
    if (activeRecognition === recognition) {
      activeRecognition = null;
    }
    micButton.classList.remove("listening");
    questionInput.placeholder = originalPlaceholder;
    if (typeof avatar !== 'undefined' && avatar) {
      avatar.setState('idle');
    }
    try {
      recognition.stop();
    } catch (e) {
      // already stopped
    }
  }

  micButton.addEventListener("click", () => {
    if (isListening) {
      stopListening();
    } else {
      startListening();
    }
  });
}

// Initialize voice input features
initVoiceInput();

// Speech Synthesis (Text-to-Speech) implementation
let currentUtterance = null;
let currentTTSButton = null;

function cleanTextForSpeech(text) {
  return text
    .replace(/\[Page \d+(?:, chunk \d+)?\]/g, "") // Strip citations like [Page X] or [Page X, chunk Y]
    .replace(/\s+/g, " ")
    .trim();
}

function speakText(text, button) {
  // If we are already speaking, stop it
  if (window.speechSynthesis && window.speechSynthesis.speaking) {
    window.speechSynthesis.cancel();
    if (currentTTSButton) {
      setButtonSpeakingState(currentTTSButton, false);
    }
    if (typeof avatar !== 'undefined' && avatar) {
      avatar.setState('idle');
    }
    // If the clicked button was the one speaking, we just stop
    if (currentTTSButton === button && button !== null) {
      currentTTSButton = null;
      currentUtterance = null;
      return;
    }
  }

  if (!window.speechSynthesis) return;

  const cleanedText = cleanTextForSpeech(text);
  if (!cleanedText) return;

  const utterance = new SpeechSynthesisUtterance(cleanedText);

  // Set voice preference if available
  const voiceSelect = document.getElementById("voice-select");
  if (voiceSelect && voiceSelect.value) {
    const voices = window.speechSynthesis.getVoices();
    const selectedVoice = voices.find(v => v.voiceURI === voiceSelect.value);
    if (selectedVoice) {
      utterance.voice = selectedVoice;
    }
  }

  utterance.onstart = () => {
    if (typeof avatar !== 'undefined' && avatar) {
      avatar.setState('speaking');
    }
  };

  utterance.onend = () => {
    if (typeof avatar !== 'undefined' && avatar) {
      avatar.setState('idle');
    }
    if (button) {
      setButtonSpeakingState(button, false);
    }
    if (currentTTSButton === button) {
      currentTTSButton = null;
      currentUtterance = null;
    }
  };

  utterance.onerror = (event) => {
    console.error("SpeechSynthesis error:", event);
    if (typeof avatar !== 'undefined' && avatar) {
      avatar.setState('idle');
    }
    if (button) {
      setButtonSpeakingState(button, false);
    }
    if (currentTTSButton === button) {
      currentTTSButton = null;
      currentUtterance = null;
    }
  };

  currentUtterance = utterance;
  currentTTSButton = button;

  if (button) {
    setButtonSpeakingState(button, true);
  }
  window.speechSynthesis.speak(utterance);
}

function setButtonSpeakingState(button, isSpeaking) {
  if (isSpeaking) {
    button.classList.add("speaking");
    button.setAttribute("aria-label", "Stop reading aloud");
    button.innerHTML = `
      <svg viewBox="0 0 24 24" width="16" height="16">
        <path fill="currentColor" d="M12,4L9.91,6.09L12,8.18M4.27,3L3,4.27L7.73,9H3V15H7L12,20V13.27L16.25,17.53C15.58,18.04 14.83,18.46 14,18.7V20.76C15.38,20.45 16.63,19.82 17.68,18.96L20,21.27L21.27,20L4.27,3M19,12C19,12.94 18.8,13.82 18.46,14.64L19.97,16.15C20.62,14.91 21,13.5 21,12C21,7.72 18,4.14 14,3.23V5.29C16.89,6.15 19,8.83 19,12M16.5,12C16.5,10.23 15.5,8.71 14,7.97V10.18L16.45,12.63C16.48,12.43 16.5,12.22 16.5,12Z" />
      </svg>
    `;
  } else {
    button.classList.remove("speaking");
    button.setAttribute("aria-label", "Read aloud");
    button.innerHTML = `
      <svg viewBox="0 0 24 24" width="16" height="16">
        <path fill="currentColor" d="M14,3.23V5.29C16.89,6.15 19,8.83 19,12C19,15.17 16.89,17.85 14,18.7V20.77C18,19.86 21,16.28 21,12C21,7.72 18,4.14 14,3.23M16.5,12C16.5,10.23 15.5,8.71 14,7.97V16C15.5,15.29 16.5,13.77 16.5,12M3,9V15H7L12,20V4L7,9H3Z" />
      </svg>
    `;
  }
}

// Stop speaking on page unload to prevent stuck voices
window.addEventListener("beforeunload", () => {
  if (window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
});

// Holographic AI Avatar Class Implementation
class HoloAvatar {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.width = this.canvas.width;
    this.height = this.canvas.height;
    
    this.state = 'idle'; // 'idle', 'thinking', 'listening', 'speaking'
    
    this.time = 0;
    this.blinkTimer = 0;
    this.isBlinking = false;
    this.blinkProgress = 0;
    
    this.mouse = { x: this.width / 2, y: this.height / 2 - 10 };
    this.targetMouse = { x: this.width / 2, y: this.height / 2 - 10 };
    this.setupMouseTracking();
    
    this.particles = [];
    this.maxParticles = 15;
    for (let i = 0; i < this.maxParticles; i++) {
      this.particles.push(this.createParticle(true));
    }
    
    this.animationFrameId = null;
  }

  setupMouseTracking() {
    window.addEventListener('mousemove', (e) => {
      const rect = this.canvas.getBoundingClientRect();
      const centerX = rect.left + rect.width / 2;
      const centerY = rect.top + rect.height / 2;
      
      const dx = e.clientX - centerX;
      const dy = e.clientY - centerY;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const maxDist = 300;
      
      const scale = Math.min(dist / maxDist, 1.0);
      const angle = Math.atan2(dy, dx);
      
      const maxEyeShift = 8;
      this.targetMouse.x = this.width / 2 + Math.cos(angle) * maxEyeShift * scale;
      this.targetMouse.y = this.height / 2 - 10 + Math.sin(angle) * maxEyeShift * scale;
    });
  }

  createParticle(randomStart = false) {
    return {
      x: (Math.random() - 0.5) * 120 + this.width / 2,
      y: randomStart ? Math.random() * 120 + this.height / 2 - 40 : this.height / 2 + 50,
      size: Math.random() * 2 + 1,
      speedY: -Math.random() * 0.8 - 0.2,
      alpha: randomStart ? Math.random() * 0.7 : 0.8,
      fadeSpeed: Math.random() * 0.005 + 0.003
    };
  }

  setState(state) {
    if (this.state === state) return;
    this.state = state;
    
    const statusIndicator = document.querySelector('.avatar-status-indicator');
    const statusText = document.getElementById('avatar-status-text');
    
    if (statusIndicator && statusText) {
      statusIndicator.className = 'avatar-status-indicator ' + state;
      const capitalized = state.charAt(0).toUpperCase() + state.slice(1);
      statusText.textContent = `Aria (${capitalized})`;
    }
  }

  start() {
    if (this.animationFrameId) return;
    const loop = () => {
      this.update();
      this.draw();
      this.animationFrameId = requestAnimationFrame(loop);
    };
    this.animationFrameId = requestAnimationFrame(loop);
  }

  stop() {
    if (this.animationFrameId) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
  }

  update() {
    this.time += 0.05;
    
    this.mouse.x += (this.targetMouse.x - this.mouse.x) * 0.1;
    this.mouse.y += (this.targetMouse.y - this.mouse.y) * 0.1;
    
    if (!this.isBlinking) {
      this.blinkTimer++;
      if (this.blinkTimer > 180 + Math.random() * 120) {
        this.isBlinking = true;
        this.blinkProgress = 0;
        this.blinkTimer = 0;
      }
    } else {
      this.blinkProgress += 0.15;
      if (this.blinkProgress >= 1) {
        this.isBlinking = false;
        this.blinkProgress = 0;
      }
    }
    
    for (let i = 0; i < this.particles.length; i++) {
      let p = this.particles[i];
      p.y += p.speedY;
      p.alpha -= p.fadeSpeed;
      if (p.alpha <= 0 || p.y < this.height / 2 - 80) {
        this.particles[i] = this.createParticle(false);
      }
    }
  }

  draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);
    
    const cx = this.width / 2;
    const cy = this.height / 2 - 10;
    
    // 1. Draw glowing background aura
    let auraGlow = ctx.createRadialGradient(cx, cy, 20, cx, cy, 90);
    if (this.state === 'listening') {
      auraGlow.addColorStop(0, 'rgba(157, 62, 56, 0.25)');
      auraGlow.addColorStop(1, 'rgba(157, 62, 56, 0)');
    } else if (this.state === 'speaking') {
      auraGlow.addColorStop(0, 'rgba(53, 96, 138, 0.25)');
      auraGlow.addColorStop(1, 'rgba(53, 96, 138, 0)');
    } else if (this.state === 'thinking') {
      auraGlow.addColorStop(0, 'rgba(199, 147, 66, 0.25)');
      auraGlow.addColorStop(1, 'rgba(199, 147, 66, 0)');
    } else {
      auraGlow.addColorStop(0, 'rgba(23, 79, 60, 0.15)');
      auraGlow.addColorStop(1, 'rgba(23, 79, 60, 0)');
    }
    ctx.fillStyle = auraGlow;
    ctx.beginPath();
    ctx.arc(cx, cy, 90, 0, Math.PI * 2);
    ctx.fill();
    
    // 2. Draw rotating/pulsing holographic rings
    ctx.lineWidth = 1;
    
    // Outer ring
    ctx.strokeStyle = this.state === 'speaking' ? 'rgba(53, 96, 138, 0.2)' : 'rgba(23, 79, 60, 0.15)';
    ctx.beginPath();
    let ringRadius = 75 + Math.sin(this.time * 2) * (this.state === 'speaking' ? 4 : 1.5);
    if (this.state === 'listening') {
      ringRadius = 75 + Math.sin(this.time * 4) * 5;
      ctx.strokeStyle = 'rgba(157, 62, 56, 0.3)';
    }
    ctx.arc(cx, cy, ringRadius, 0, Math.PI * 2);
    ctx.stroke();
    
    // Inner dashed spinning ring
    ctx.strokeStyle = this.state === 'thinking' ? 'rgba(199, 147, 66, 0.3)' : 'rgba(23, 79, 60, 0.1)';
    ctx.save();
    ctx.translate(cx, cy);
    if (this.state === 'thinking') {
      ctx.rotate(this.time);
    } else {
      ctx.rotate(this.time * 0.2);
    }
    ctx.setLineDash([4, 12]);
    ctx.beginPath();
    ctx.arc(0, 0, 68, 0, Math.PI * 2);
    ctx.stroke();
    ctx.restore();
    
    // 3. Draw head/capsule glassmorphic base
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, 55, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
    ctx.fill();
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.stroke();
    ctx.restore();
    
    // 4. Draw floating particles
    ctx.fillStyle = this.state === 'thinking' ? 'rgba(199, 147, 66, 0.6)' : 'rgba(23, 79, 60, 0.4)';
    this.particles.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
      ctx.globalAlpha = p.alpha;
      ctx.fill();
    });
    ctx.globalAlpha = 1.0;
    
    // 5. Draw Eyes
    let eyeSpacing = 16;
    let eyeYOffset = -8;
    let eyeWidth = 7;
    let eyeHeight = 10;
    
    let scaleY = 1.0;
    if (this.isBlinking) {
      scaleY = Math.abs(Math.sin(this.blinkProgress * Math.PI));
    }
    
    let leftEyeX = cx - eyeSpacing;
    let rightEyeX = cx + eyeSpacing;
    let eyeY = cy + eyeYOffset;
    
    let trackingOffsetX = 0;
    let trackingOffsetY = 0;
    
    if (this.state === 'thinking') {
      trackingOffsetX = 3;
      trackingOffsetY = -4;
      eyeHeight = 6;
    } else if (this.state === 'listening') {
      eyeHeight = 12;
    } else {
      trackingOffsetX = (this.mouse.x - cx) * 0.5;
      trackingOffsetY = (this.mouse.y - cy) * 0.5;
    }
    
    ctx.fillStyle = this.state === 'listening' ? 'var(--danger)' : 
                    this.state === 'speaking' ? '#35608a' :
                    this.state === 'thinking' ? 'var(--gold)' : 'var(--forest)';
    
    // Left Eye
    ctx.save();
    ctx.translate(leftEyeX + trackingOffsetX, eyeY + trackingOffsetY);
    ctx.scale(1, scaleY);
    ctx.beginPath();
    ctx.ellipse(0, 0, eyeWidth / 2, eyeHeight / 2, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    
    // Right Eye
    ctx.save();
    ctx.translate(rightEyeX + trackingOffsetX, eyeY + trackingOffsetY);
    ctx.scale(1, scaleY);
    ctx.beginPath();
    ctx.ellipse(0, 0, eyeWidth / 2, eyeHeight / 2, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    
    // 6. Draw Mouth
    ctx.save();
    ctx.beginPath();
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.strokeStyle = ctx.fillStyle;
    
    let mouthY = cy + 18;
    let mouthWidth = 24;
    
    if (this.state === 'speaking') {
      ctx.beginPath();
      let segments = 20;
      let startX = cx - mouthWidth / 2;
      for (let i = 0; i <= segments; i++) {
        let pct = i / segments;
        let x = startX + pct * mouthWidth;
        let envelope = Math.sin(pct * Math.PI);
        let amp = 5 * envelope * (Math.sin(this.time * 4) * 0.6 + Math.sin(this.time * 9 + pct * 5) * 0.4);
        let y = mouthY + amp;
        if (i === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      }
      ctx.stroke();
    } else if (this.state === 'thinking') {
      ctx.beginPath();
      ctx.moveTo(cx - 8, mouthY);
      ctx.lineTo(cx + 8, mouthY);
      ctx.stroke();
    } else if (this.state === 'listening') {
      ctx.beginPath();
      ctx.arc(cx, mouthY, 4, 0, Math.PI * 2);
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.moveTo(cx - mouthWidth / 2, mouthY);
      ctx.quadraticCurveTo(cx, mouthY + 5 + Math.sin(this.time) * 1.0, cx + mouthWidth / 2, mouthY);
      ctx.stroke();
    }
    ctx.restore();
  }
}

// Voice Selector Population Utility
function initVoiceSelector() {
  const voiceSelect = document.getElementById("voice-select");
  if (!voiceSelect) return;

  function populateVoices() {
    if (!window.speechSynthesis) return;
    const voices = window.speechSynthesis.getVoices();
    
    voiceSelect.innerHTML = '<option value="">Default Voice</option>';
    
    voices.forEach(voice => {
      const option = document.createElement("option");
      option.value = voice.voiceURI;
      option.textContent = `${voice.name} (${voice.lang})`;
      voiceSelect.appendChild(option);
    });
  }

  populateVoices();
  if (window.speechSynthesis && typeof window.speechSynthesis.onvoiceschanged !== 'undefined') {
    window.speechSynthesis.onvoiceschanged = populateVoices;
  }
}

// Initialize Avatar and Voice Selector
const avatar = new HoloAvatar('avatar-canvas');
if (avatar) {
  avatar.setState('idle');
  avatar.start();
}
initVoiceSelector();
