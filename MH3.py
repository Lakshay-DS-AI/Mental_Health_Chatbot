from flask import Flask, render_template_string, request, jsonify, session, redirect, url_for, send_file
import random
import whisper
from gtts import gTTS
import io
import os
import tempfile
import imageio_ffmpeg as iio_ffmpeg  # type: ignore

# === Set ffmpeg path for Whisper (no sudo needed) ===
os.environ["IMAGEIO_FFMPEG_EXE"] = iio_ffmpeg.get_ffmpeg_exe()

# === Chatbot API Client (Groq) ===
from groq import Groq
chatbot_api = Groq(api_key="")

# === STT API (Whisper) ===
stt_model = whisper.load_model("base")  # separate model for STT

# === Flask App ===
app = Flask(__name__)
app.secret_key = "supersecretkey"

# === HTML Templates ===
login_html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Login - Mental Health Support</title>
<style>
body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background: linear-gradient(135deg, #6a11cb, #2575fc); height: 100vh; display: flex; justify-content: center; align-items: center; margin: 0; }
.login-container { background: #fff; padding: 2rem; border-radius: 12px; box-shadow: 0 8px 20px rgba(0,0,0,0.2); width: 400px; text-align: center; animation: fadeIn 0.8s ease-in-out; }
.login-container h1 { margin-bottom: 1rem; color: #333; }
.login-container p { margin-bottom: 1.5rem; color: #555; }
.login-container input { width: 100%; padding: 12px; margin: 10px 0; border-radius: 8px; border: 1px solid #ccc; font-size: 16px; transition: border 0.3s; }
.login-container input:focus { border-color: #2575fc; outline: none; }
.login-container button { width: 100%; padding: 12px; border: none; border-radius: 8px; background: #2575fc; color: white; font-size: 16px; cursor: pointer; transition: background 0.3s; }
.login-container button:hover { background: #1a5edc; }
@keyframes fadeIn { from { opacity: 0; transform: translateY(-20px); } to { opacity: 1; transform: translateY(0); } }
</style>
</head>
<body>
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
</body>
</html>
"""

chat_html = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Mental Health Chat</title>
<style>
body { font-family: Arial, sans-serif; background: #f4f7fc; margin: 0; padding: 0; display: flex; flex-direction: column; height: 100vh; }
.chat-container { flex: 1; display: flex; flex-direction: column; max-width: 600px; margin: auto; padding: 20px; }
.messages { flex: 1; overflow-y: auto; border: 1px solid #ccc; border-radius: 10px; padding: 15px; background: white; margin-bottom: 15px; }
.message { margin: 10px 0; }
.user { text-align: right; color: blue; }
.bot { text-align: left; color: green; }
.input-area { display: flex; gap: 10px; }
.input-area input { flex: 1; border-radius: 8px; border: 1px solid #ccc; padding:10px; }
.input-area button { background: #2575fc; color: white; border: none; border-radius: 8px; cursor: pointer; padding:10px; }
.input-area button:hover { background: #1a5edc; }
</style>
</head>
<body>
<div class="chat-container">
<h2>Welcome, {{name}} 👋</h2>
<div class="messages" id="chat-box"></div>
<div class="input-area">
<input type="text" id="user-input" placeholder="Type your message...">
<button onclick="sendMessage()">Send</button>
<button onclick="startRecording()">🎤</button>
</div>
<audio id="tts-audio" controls style="display:none;"></audio>
</div>

<script>
async function sendMessage(text=null){
    const input = document.getElementById("user-input");
    const message = text || input.value.trim();
    if(!message) return;
    const chatBox = document.getElementById("chat-box");
    chatBox.innerHTML += `<div class='message user'><b>You:</b> ${message}</div>`;
    input.value = "";

    const resp = await fetch("/api/chat", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({message})
    });
    const data = await resp.json();
    chatBox.innerHTML += `<div class='message bot'><b>Bot:</b> ${data.reply}</div>`;
    chatBox.scrollTop = chatBox.scrollHeight;

    // Play TTS
    const ttsResp = await fetch("/api/tts", {
        method:"POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({text: data.reply})
    });
    const ttsBlob = await ttsResp.blob();
    const audioURL = URL.createObjectURL(ttsBlob);
    const audio = document.getElementById("tts-audio");
    audio.src = audioURL;
    audio.style.display="block";
    audio.play();
}

function startRecording(){
    let mediaRecorder;
    let audioChunks = [];
    navigator.mediaDevices.getUserMedia({audio:true}).then(stream=>{
        mediaRecorder = new MediaRecorder(stream);
        audioChunks=[];
        mediaRecorder.ondataavailable = e => audioChunks.push(e.data);
        mediaRecorder.onstop = async ()=>{
            const audioBlob = new Blob(audioChunks,{type:'audio/webm'});
            const formData = new FormData();
            formData.append("audio_file", audioBlob, "recording.webm");
            const resp = await fetch("/api/stt", {method:"POST", body: formData});
            const data = await resp.json();
            sendMessage(data.text);
        };
        mediaRecorder.start();
        setTimeout(()=>mediaRecorder.stop(),5000);
    });
}
</script>
</body>
</html>
"""

# === Routes ===
@app.route("/", methods=["GET","POST"])
def home():
    if request.method=="POST":
        session["name"]=request.form.get("name")
        session["email"]=request.form.get("email")
        session["phone"]=request.form.get("phone")
        return redirect(url_for("chat"))
    return render_template_string(login_html)

@app.route("/chat")
def chat():
    if "name" not in session:
        return redirect(url_for("home"))
    return render_template_string(chat_html, name=session["name"])

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("home"))

# === Chatbot API ===
@app.route("/api/chat", methods=["POST"])
def chat_api():
    user_input = request.json.get("message","").lower()
    banned_words = ["sex","porn","kill","suicide","murder","violence","drugs","rape"]
    if any(word in user_input for word in banned_words):
        return jsonify({"reply":"⚠️ I’m here to support your mental health safely. Please reach out to a trusted person or helpline if needed."})

    response = chatbot_api.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role":"system","content":"You are a compassionate mental health support assistant."},
            {"role":"user","content":user_input}
        ],
        temperature=0.7,
        max_tokens=200
    )
    ai_message = response.choices[0].message.content.strip()
    return jsonify({"reply": ai_message})

# === TTS API ===
@app.route("/api/tts", methods=["POST"])
def tts_api():
    text = request.json.get("text","")
    tts = gTTS(text=text, lang="en")
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return send_file(audio_fp, mimetype="audio/mpeg")

# === STT API ===
@app.route("/api/stt", methods=["POST"])
def stt_api():
    if "audio_file" not in request.files:
        return jsonify({"text":""})
    audio_file=request.files["audio_file"]
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as tmp:
        audio_file.save(tmp.name)
        result=stt_model.transcribe(tmp.name)
        os.unlink(tmp.name)
    return jsonify({"text": result["text"]})

# === Run App ===
if __name__=="__main__":
    app.run(debug=True)
