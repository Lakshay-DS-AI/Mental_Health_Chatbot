import os
import io
import base64
import threading
from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for, send_file
import cv2
import numpy as np
import pyttsx3
from fer import FER
import hs_module as hs

# ---------------- Groq API Client ----------------
# Make sure to set your API key here directly
from groq import Groq
chatbot_api = Groq(api_key=)

def groq_chatbot_reply(user_input, sentiment="neutral"):
    system_message = f"""
    You are a compassionate mental health support assistant.
    The user currently looks {sentiment}.
    Please consider this emotional state when replying.
    """
    try:
        response = chatbot_api.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("Groq error:", e)
        return "(I'm sorry, I couldn't connect right now.)"

# ---------------- Flask App ----------------
app = Flask(__name__)
app.secret_key = "supersecretkey"

# Initialize FER detector once
emotion_detector = FER(mtcnn=True)


# ===== HTML Templates =====

# === Login Page with Sun/Moon Toggle (Pure HTML + CSS) ===
login_html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Login</title>
<style>
body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: linear-gradient(135deg, #6a11cb, #2575fc);
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  margin: 0;
  transition: background 0.5s, color 0.5s;
}

/* === Login Container === */
.login-container {
  background: #fff;
  padding: 2rem;
  border-radius: 12px;
  box-shadow: 0 8px 20px rgba(0,0,0,0.2);
  width: 400px;
  text-align: center;
  animation: fadeIn 0.8s ease-in-out;
  position: relative;
}

.login-container h1 {
  margin-bottom: 1rem;
  color: #333;
}
.login-container p {
  margin-bottom: 1.5rem;
  color: #555;
}

/* ✅ Fixed Input Alignment */
.login-container form {
  display: flex;
  flex-direction: column;
  align-items: center;
  box-sizing: border-box;
  width: 100%;
}

.login-container input {
  width: calc(100% - 24px); /* Equal spacing on both sides */
  box-sizing: border-box;
  padding: 12px;
  margin: 10px 0;
  border-radius: 8px;
  border: 1px solid #ccc;
  font-size: 16px;
  transition: border 0.3s;
}
.login-container input:focus {
  border-color: #2575fc;
  outline: none;
}
.login-container button {
  width: 100%;
  padding: 12px;
  border: none;
  border-radius: 8px;
  background: #2575fc;
  color: white;
  font-size: 16px;
  cursor: pointer;
  transition: background 0.3s;
}
.login-container button:hover {
  background: #1a5edc;
}

/* === Fade Animation === */
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(-20px); }
  to { opacity: 1; transform: translateY(0); }
}

/* === Toggle Switch === */
.theme-toggle {
  position: absolute;
  top: 20px;
  right: 20px;
}
.theme-toggle input {
  display: none;
}
.theme-toggle label {
  font-size: 26px;
  cursor: pointer;
  user-select: none;
  transition: transform 0.4s ease;
}
.theme-toggle label:hover {
  transform: scale(1.2);
}

/* === Dark Mode Styles === */
body.dark-mode {
  background: #121212;
  color: #e0e0e0;
}
body.dark-mode .login-container {
  background: #1e1e1e;
  box-shadow: 0 8px 20px rgba(0,0,0,0.5);
}
body.dark-mode .login-container h1,
body.dark-mode .login-container p {
  color: #e0e0e0;
}
body.dark-mode .login-container input {
  background: #1e1e1e;
  color: #e0e0e0;
  border: 1px solid #333;
}
body.dark-mode .login-container button {
  background: #2979ff;
  color: white;
}

/* === Toggle Icons === */
.theme-toggle .sun {
  display: inline;
}
.theme-toggle .moon {
  display: none;
}
body.dark-mode .theme-toggle .sun {
  display: none;
}
body.dark-mode .theme-toggle .moon {
  display: inline;
}
</style>
</head>

<body>
<!-- === Theme Toggle === -->
<div class="theme-toggle">
  <input type="checkbox" id="modeSwitch">
  <label for="modeSwitch" class="sun">🌞</label>
  <label for="modeSwitch" class="moon">🌙</label>
</div>

<div class="login-container">
  <h1>Welcome Back 👋</h1>
  <p>Please log in to continue</p>
  <form action="/" method="POST">
    <input type="text" name="name" placeholder="Full Name" required>
    <input type="email" name="email" placeholder="Email Address" required>
    <input type="tel" name="phone" placeholder="Phone Number" required>
    <button type="submit">Login</button>
  </form>
</div>

<script>
  // Checkbox-based CSS toggle activation
  const modeSwitch = document.getElementById('modeSwitch');
  modeSwitch.addEventListener('change', function() {
    document.body.classList.toggle('dark-mode', this.checked);
    localStorage.setItem('dark-mode', this.checked);
  });
  // Restore last mode
  if (localStorage.getItem('dark-mode') === 'true') {
    document.body.classList.add('dark-mode');
    modeSwitch.checked = true;
  }
</script>

</body>
</html>
"""
normal_chat_html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Normal Chat</title>
<style>
body {
  font-family: Arial, sans-serif;
  background: #f4f7fc;
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: column;
  height: 100vh;
  transition: background 0.5s, color 0.5s;
}

/* Dark mode */
body.dark-mode { background: #121212; color: #e0e0e0; }
body.dark-mode .container .left, body.dark-mode .container .right { color:#e0e0e0; }
body.dark-mode .video-box { background:#1e1e1e; box-shadow:0 8px 20px rgba(0,0,0,0.5); }
body.dark-mode .messages { background:#2a2a2a; border:1px solid #444; color:#e0e0e0; }

/* Toggle */
.theme-toggle { position:absolute; top:20px; right:20px; }
.theme-toggle input { display:none; }
.theme-toggle label { font-size:26px; cursor:pointer; user-select:none; transition: transform 0.4s ease; }
.theme-toggle label:hover { transform: scale(1.2); }

/* Layout */
.container { display:flex; max-width:1000px; margin:auto; gap:20px; flex:1; padding:20px; }
.left { flex:1; display:flex; flex-direction:column; }
.right { flex:1; display:flex; flex-direction:column; }
.video-box { border:1px solid #ccc; border-radius:10px; overflow:hidden; height:360px; }
#camera { width:100%; height:100%; object-fit:cover; }
.status { margin-top:10px; font-weight:bold; }
.chat-container { flex:1; display:flex; flex-direction:column; }
.messages { flex:1; overflow-y:auto; border:1px solid #ccc; border-radius:10px; padding:15px; background:white; margin-bottom:15px; }
.message.user { text-align:right; color:blue; margin:10px 0; }
.message.bot { text-align:left; color:green; margin:10px 0; }
.input-area { display:flex; gap:10px; }
.input-area input { flex:1; border-radius:8px; border:1px solid #ccc; padding:10px; }
.input-area button { background:#2575fc; color:white; border:none; border-radius:8px; cursor:pointer; padding:10px; }
.input-area button:hover { background:#1a5edc; }
</style>
</head>
<body>
<div class="theme-toggle">
  <input type="checkbox" id="modeSwitch">
  <label for="modeSwitch" class="sun">🌞</label>
  <label for="modeSwitch" class="moon">🌙</label>
</div>

<div class="container">
  <div class="left">
    <h3>Live Emotion Detection</h3>
    <div class="video-box">
      <video id="camera" autoplay playsinline></video>
    </div>
    <div class="status">Detected Emotion: <span id="emotion">—</span></div>
  </div>
  <div class="right">
    <div class="chat-container">
      <h2>Welcome, {{name}}</h2>
      <div class="messages" id="chat-box"></div>
      <div class="input-area">
        <input type="text" id="user-input" placeholder="Type your message...">
        <button onclick="sendMessage()">Send</button>
      </div>
    </div>
  </div>
</div>
<canvas id="hiddenCanvas" style="display:none;"></canvas>

<script>
// Dark/light toggle
const modeSwitch = document.getElementById('modeSwitch');
function updateToggleIcons() {
    if(document.body.classList.contains('dark-mode')){
        document.querySelector('.theme-toggle .sun').style.display = 'none';
        document.querySelector('.theme-toggle .moon').style.display = 'inline';
    } else {
        document.querySelector('.theme-toggle .sun').style.display = 'inline';
        document.querySelector('.theme-toggle .moon').style.display = 'none';
    }
}

modeSwitch.addEventListener('change', function() {
    document.body.classList.toggle('dark-mode', this.checked);
    localStorage.setItem('dark-mode', this.checked);
    updateToggleIcons();
});

if(localStorage.getItem('dark-mode')==='true'){
    document.body.classList.add('dark-mode');
    modeSwitch.checked = true;
}
updateToggleIcons();

// Camera + Emotion polling
const video = document.getElementById('camera');
const emotionSpan = document.getElementById('emotion');
navigator.mediaDevices.getUserMedia({video:true}).then(stream => { video.srcObject = stream; }).catch(err => alert('Camera error:'+err.message));

async function pollEmotion(){
  const canvas = document.getElementById('hiddenCanvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video,0,0,canvas.width,canvas.height);
  const dataUrl = canvas.toDataURL('image/jpeg',0.6);
  try {
    const resp = await fetch('/api/emotion_detect',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({image:dataUrl})
    });
    const data = await resp.json();
    if(data.emotion) emotionSpan.innerText = data.emotion;
  } catch(e){ console.error(e); }
  setTimeout(pollEmotion, 500);
}
video.addEventListener('playing', pollEmotion);

// Chat send
async function sendMessage(){
  const input = document.getElementById('user-input');
  const message = input.value.trim(); if(!message) return;
  const chatBox = document.getElementById('chat-box');
  chatBox.innerHTML += `<div class='message user'><b>You:</b> ${message}</div>`;
  input.value='';
  try {
    const resp = await fetch("/api/chat",{method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({message})});
    const data = await resp.json();
    chatBox.innerHTML += `<div class='message bot'><b>Bot:</b> ${data.reply}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;
    // TTS
    const ttsResp = await fetch("/api/tts",{method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({text:data.reply})});
    const ttsBlob = await ttsResp.blob();
    const audioUrl = URL.createObjectURL(ttsBlob);
    const audio = new Audio(audioUrl); audio.play();
  } catch(e){ console.error(e); }
}
</script>
</body>
</html>
"""
hand_chat_html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Hand Sign Chat</title>
<style>
body { font-family: Arial, sans-serif; background:#f4f7fc; margin:0; padding:20px; display:flex; justify-content:center; align-items:flex-start; transition: background 0.5s, color 0.5s; }
body.dark-mode { background:#121212; color:#e0e0e0; }
body.dark-mode .video-box { background:#1e1e1e; box-shadow:0 8px 20px rgba(0,0,0,0.5); }
body.dark-mode .sentence-box, body.dark-mode #replyBox { background:#2a2a2a; border:1px solid #444; color:#e0e0e0; }

.theme-toggle { position:absolute; top:20px; right:20px; }
.theme-toggle input { display:none; }
.theme-toggle label { font-size:26px; cursor:pointer; user-select:none; transition: transform 0.4s ease; }
.theme-toggle label:hover { transform: scale(1.2); }

.container { width:1000px; display:flex; gap:20px; }
.left { flex:1; }
.right { width:420px; }
.video-box { background:#000; border-radius:10px; overflow:hidden; }
#camera { width:100%; height:480px; object-fit:cover; }
.status { margin-top:10px; font-weight:bold; }
.sentence-box { height:150px; border-radius:8px; background:white; padding:10px; border:1px solid #ccc; overflow:auto; }
.controls { display:flex; gap:10px; margin-top:10px; }
.btn { padding:10px 14px; border-radius:8px; background:#2575fc; color:white; border:none; cursor:pointer; }
.btn.secondary { background:#666; }
</style>
</head>
<body>
<div class="theme-toggle">
  <input type="checkbox" id="modeSwitch">
  <label for="modeSwitch" class="sun">🌞</label>
  <label for="modeSwitch" class="moon">🌙</label>
</div>

<div class="container">
  <div class="left">
    <div class="video-box"><video id="camera" autoplay playsinline></video></div>
    <div class="status">Detected word: <span id="detected">—</span></div>
    <div class="controls">
      <button class="btn" id="enterBtn">Send (Enter)</button>
      <button class="btn secondary" id="resetBtn">Reset</button>
    </div>
  </div>
  <div class="right">
    <h3>Constructed sentence</h3>
    <div class="sentence-box" id="sentenceBox"></div>
    <p style="color:#666; font-size:13px;">Tip: the client polls the server with captured frames. Press <b>Send</b> to send the sentence to bot.</p>
    <h4>Chat response</h4>
    <div id="replyBox" style="white-space:pre-wrap; background:white;border:1px solid #ccc;padding:10px;border-radius:8px;min-height:120px;"></div>
  </div>
</div>
<canvas id="hiddenCanvas" style="display:none;"></canvas>

<script>
// Dark/light toggle
const modeSwitch = document.getElementById('modeSwitch');
function updateToggleIcons() {
    if(document.body.classList.contains('dark-mode')){
        document.querySelector('.theme-toggle .sun').style.display='none';
        document.querySelector('.theme-toggle .moon').style.display='inline';
    } else {
        document.querySelector('.theme-toggle .sun').style.display='inline';
        document.querySelector('.theme-toggle .moon').style.display='none';
    }
}
modeSwitch.addEventListener('change', function(){
    document.body.classList.toggle('dark-mode', this.checked);
    localStorage.setItem('dark-mode', this.checked);
    updateToggleIcons();
});
if(localStorage.getItem('dark-mode')==='true'){
    document.body.classList.add('dark-mode');
    modeSwitch.checked=true;
}
updateToggleIcons();

// Camera + hand prediction
const video=document.getElementById('camera');
const detectedSpan=document.getElementById('detected');
const sentenceBox=document.getElementById('sentenceBox');
const replyBox=document.getElementById('replyBox');

navigator.mediaDevices.getUserMedia({video:true}).then(stream=>{video.srcObject=stream;}).catch(err=>alert('Camera error:'+err.message));

let sentence=sessionStorage.getItem('hand_sentence')||"";
sentenceBox.innerText=sentence;
let polling=true;
async function pollFrame(){
  if(!polling) return;
  const canvas=document.getElementById('hiddenCanvas');
  canvas.width=video.videoWidth||640;
  canvas.height=video.videoHeight||480;
  const ctx=canvas.getContext('2d');
  ctx.drawImage(video,0,0,canvas.width,canvas.height);
  const dataUrl=canvas.toDataURL('image/jpeg',0.6);
  try{
    const resp=await fetch('/api/hand_predict',{method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({image:dataUrl})});
    const data=await resp.json();
    if(data.predicted){
      detectedSpan.innerText=data.predicted;
      if(data.appended){sentence=data.sentence; sentenceBox.innerText=sentence; sessionStorage.setItem('hand_sentence',sentence);}
    } else { detectedSpan.innerText='—'; }
  } catch(e){console.error(e);}
  setTimeout(pollFrame,350);
}
video.addEventListener('playing',()=>{pollFrame();});

// Enter/send
document.getElementById('enterBtn').addEventListener('click',async ()=>{
  const s=sentence; if(!s.trim()) return alert('No sentence to send.');
  const resp=await fetch('/api/hand_enter',{method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({sentence:s})});
  const data=await resp.json();
  replyBox.innerText=data.reply;
  sentence=''; sentenceBox.innerText=''; sessionStorage.removeItem('hand_sentence');
});

// Reset
document.getElementById('resetBtn').addEventListener('click',async ()=>{
  await fetch('/api/hand_reset',{method:'POST'});
  sentence=''; sentenceBox.innerText=''; sessionStorage.removeItem('hand_sentence'); detectedSpan.innerText='—';
});
</script>
</body>
</html>
"""
mode_select_html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Choose Mode</title>
<style>
body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: linear-gradient(135deg,#f8f9ff,#e6f0ff);
  height:100vh;
  display:flex;
  flex-direction: column;
  align-items:center;
  justify-content:center;
  margin:0;
  transition: background 0.5s, color 0.5s;
}
.container {
  width:800px;
  display:flex;
  gap: 30px;
  justify-content:center;
}
.card {
  width:320px;
  padding: 30px;
  border-radius:16px;
  box-shadow: 0 8px 30px rgba(20,30,60,0.08);
  background:white;
  text-align:center;
  cursor:pointer;
  transition: transform .2s, box-shadow .2s;
}
.card:hover {
  transform: translateY(-6px);
  box-shadow: 0 12px 40px rgba(20,30,60,0.12);
}
.card h2 { margin:12px 0; }
.card p { color:#666; }

/* === Toggle Switch === */
.theme-toggle {
  position: absolute;
  top: 20px;
  right: 20px;
}
.theme-toggle input { display: none; }
.theme-toggle label {
  font-size: 26px;
  cursor: pointer;
  user-select: none;
  transition: transform 0.4s ease;
}
.theme-toggle label:hover { transform: scale(1.2); }

/* === Dark Mode === */
body.dark-mode {
  background: #121212;
  color: #e0e0e0;
}
body.dark-mode .card {
  background: #1e1e1e;
  box-shadow: 0 8px 30px rgba(0,0,0,0.5);
}
body.dark-mode .card p { color:#ccc; }

/* === Toggle Icons === */
.theme-toggle .sun { display: inline; }
.theme-toggle .moon { display: none; }
body.dark-mode .theme-toggle .sun { display: none; }
body.dark-mode .theme-toggle .moon { display: inline; }

</style>
</head>
<body>
<div class="theme-toggle">
  <input type="checkbox" id="modeSwitch">
  <label for="modeSwitch" class="sun">🌞</label>
  <label for="modeSwitch" class="moon">🌙</label>
</div>

<div style="text-align:center; margin-bottom:20px;">
  <h1>Welcome, {{name}}</h1>
  <p>Choose a chat mode</p>
</div>

<div class="container">
  <div class="card" onclick="location.href='{{ url_for('normal_chat') }}'">
    <img src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='80' height='80'><rect rx='12' width='80' height='80' fill='%232575fc'/></svg>" alt="">
    <h2>Normal Chat</h2>
    <p>Standard mental-health conversational mode</p>
  </div>
  <div class="card" onclick="location.href='{{ url_for('hand_chat') }}'">
    <img src="data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='80' height='80'><rect rx='12' width='80' height='80' fill='%2300b894'/></svg>" alt="">
    <h2>Hand Sign Chat</h2>
    <p>Use hand signs to form sentences and chat</p>
  </div>
</div>

<script>
const modeSwitch = document.getElementById('modeSwitch');
modeSwitch.addEventListener('change', function() {
  document.body.classList.toggle('dark-mode', this.checked);
  localStorage.setItem('dark-mode', this.checked);
});
if (localStorage.getItem('dark-mode') === 'true') {
  document.body.classList.add('dark-mode');
  modeSwitch.checked = true;
}
</script>

</body>
</html>
"""
# ---------------- Routes ----------------
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        session["name"] = request.form.get("name")
        session["email"] = request.form.get("email")
        session["phone"] = request.form.get("phone")
        return redirect(url_for("mode_select"))
    return render_template_string(login_html)

@app.route("/mode")
def mode_select():
    if "name" not in session:
        return redirect(url_for("home"))
    return render_template_string(mode_select_html, name=session["name"])

@app.route("/normal_chat")
def normal_chat():
    if "name" not in session:
        return redirect(url_for("home"))
    return render_template_string(normal_chat_html, name=session["name"])

@app.route("/hand_chat")
def hand_chat():
    if "name" not in session:
        return redirect(url_for("home"))
    session.setdefault("hand_sentence", "")
    return render_template_string(hand_chat_html, name=session["name"])

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# ---------------- Chat API ----------------
@app.route("/api/chat", methods=["POST"])
def chat_api():
    user_input = request.json.get("message", "").strip()
    sentiment = session.get("sentiment", "neutral")
    banned_words = ["sex","porn","kill","suicide","murder","violence","drugs","rape"]
    if any(word in user_input.lower() for word in banned_words):
        return jsonify({"reply": "⚠️ I’m here to support your mental health safely. Please reach out to a trusted person or helpline if needed."})
    ai_message = groq_chatbot_reply(user_input, sentiment)
    return jsonify({"reply": ai_message})

# ---------------- TTS API ----------------
@app.route("/api/tts", methods=["POST"])
def tts_api():
    text = request.json.get("text", "")
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)
    audio_fp = io.BytesIO()
    temp_file = "temp_tts.mp3"
    engine.save_to_file(text, temp_file)
    engine.runAndWait()
    with open(temp_file, "rb") as f:
        audio_fp.write(f.read())
    audio_fp.seek(0)
    os.remove(temp_file)
    return send_file(audio_fp, mimetype="audio/mpeg")

# ---------------- Hand-sign APIs ----------------
@app.route("/api/hand_predict", methods=["POST"])
def hand_predict():
    data = request.json.get("image")
    if not data:
        return jsonify({"predicted": None, "appended": False, "sentence": session.get("hand_sentence","")})
    try:
        image_data = data.split(",")[1] if "," in data else data
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except:
        return jsonify({"predicted": None, "appended": False, "sentence": session.get("hand_sentence","")})

    clf = hs.load_model()
    if clf is None:
        return jsonify({"predicted": None, "appended": False, "sentence": session.get("hand_sentence","")})

    pred = hs.predict_from_frame(frame, clf=clf)
    if pred is None:
        session['hand_last_pred'] = None
        session['hand_pred_count'] = 0
        return jsonify({"predicted": None, "appended": False, "sentence": session.get("hand_sentence","")})

    last = session.get('hand_last_pred', None)
    count = session.get('hand_pred_count', 0)
    stable_required = 4
    appended = False

    if last == pred:
        count += 1
    else:
        count = 1
        last = pred
    session['hand_last_pred'] = last
    session['hand_pred_count'] = count

    if count >= stable_required:
        s = session.get("hand_sentence", "")
        if pred in [".", ",", "?", "!"]:
            s = s.strip() + pred + " "
        else:
            words = s.strip().split()
            last_word = words[-1] if words else ""
            if pred != last_word:
                to_add = pred.capitalize() if s.strip()=="" else pred.lower()
                s = (s + " " + to_add).strip() + " "
        session['hand_sentence'] = s
        appended = True
        session['hand_pred_count'] = 0
        session['hand_last_pred'] = None

    return jsonify({"predicted": pred, "appended": appended, "sentence": session.get("hand_sentence","")})

@app.route("/api/hand_enter", methods=["POST"])
def hand_enter():
    sent = request.json.get("sentence", "").strip()
    if not sent:
        return jsonify({"reply": "No sentence provided."})
    reply = groq_chatbot_reply(sent, session.get("sentiment", "neutral"))
    session['hand_sentence'] = ""
    session['hand_last_pred'] = None
    session['hand_pred_count'] = 0
    return jsonify({"reply": reply})

@app.route("/api/hand_reset", methods=["POST"])
def hand_reset():
    session['hand_sentence'] = ""
    session['hand_last_pred'] = None
    session['hand_pred_count'] = 0
    return jsonify({"ok": True})

# ---------------- Emotion Detection ----------------
@app.route("/api/emotion_detect", methods=["POST"])
def emotion_detect():
    data = request.json.get("image")
    if not data:
        return jsonify({"emotion": None})
    try:
        image_data = data.split(",")[1] if "," in data else data
        img_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        emotions = emotion_detector.detect_emotions(frame_rgb)
        if emotions:
            top_emotion = max(emotions[0]["emotions"], key=emotions[0]["emotions"].get)
            session['sentiment'] = top_emotion
            return jsonify({"emotion": top_emotion})
        return jsonify({"emotion": "neutral"})
    except Exception as e:
        print("Emotion detect error:", e)
        return jsonify({"emotion": None})

# ---------------- Main ----------------
def start_flask():
    app.run(debug=False, threaded=True)

def start_cli_training():
    hs.run_cli()

if __name__ == "__main__":
    print("Choose an option:")
    print("1: Start Website")
    print("2: Hand Sign Training / CLI (HS module)")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        print("Starting Flask website...")
        start_flask()
    elif choice == "2":
        print("Starting HS CLI (data collection, training)...")
        start_cli_training()
    else:
        print("Invalid choice. Exiting.")

