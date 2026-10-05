@app.get("/app", response_class=HTMLResponse)
async def chat_ui():
    return """
    <!DOCTYPE html>
    <html dir="rtl" lang="ar">
    <head>
        <meta charset="UTF-8">
        <title>منصة موجة البيان</title>
        <style>
            body { font-family: sans-serif; background: #0f172a; color: white; display: flex; justify-content: center; align-items: center; height: 100vh; margin: 0; }
            .chat-card { width: 90%; max-width: 450px; height: 600px; background: #1e293b; border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; box-shadow: 0 4px 20px rgba(0,0,0,0.4); }
            .header { background: #3b82f6; padding: 15px; text-align: center; font-weight: bold; font-size: 1.1rem; }
            .messages { flex: 1; padding: 15px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
            .msg { padding: 10px 14px; border-radius: 8px; max-width: 80%; line-height: 1.4; font-size: 0.95rem; }
            .user { background: #2563eb; align-self: flex-start; }
            .bot { background: #334155; align-self: flex-end; }
            .input-box { display: flex; padding: 10px; background: #0f172a; gap: 8px; }
            input { flex: 1; padding: 10px; border-radius: 6px; border: 1px solid #475569; background: #1e293b; color: white; outline: none; }
            button { background: #3b82f6; color: white; border: none; padding: 10px 18px; border-radius: 6px; cursor: pointer; font-weight: bold; }
            button:hover { background: #1d4ed8; }
        </style>
    </head>
    <body>
        <div class="chat-card">
            <div class="header">مستشار موجة البيان الذكي</div>
            <div class="messages" id="chat">
                <div class="msg bot">مرحباً بك في منصة موجة البيان! كيف يمكنني مساعدتك اليوم؟</div>
            </div>
            <div class="input-box">
                <input type="text" id="userInput" placeholder="اكتب سؤالك هنا..." onkeydown="if(event.key==='Enter') send()">
                <button onclick="send()">إرسال</button>
            </div>
        </div>

        <script>
            async function send() {
                const input = document.getElementById('userInput');
                const chat = document.getElementById('chat');
                const text = input.value.trim();
                if (!text) return;

                chat.innerHTML += `<div class="msg user">${text}</div>`;
                input.value = '';
                chat.scrollTop = chat.scrollHeight;

                const botMsg = document.createElement('div');
                botMsg.className = 'msg bot';
                botMsg.innerText = 'جاري التفكير...';
                chat.appendChild(botMsg);
                chat.scrollTop = chat.scrollHeight;

                try {
                    const res = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message: text, model: 'auto' })
                    });
                    const data = await res.json();
                    botMsg.innerText = data.response;
                } catch(e) {
                    botMsg.innerText = 'حدث خطأ في الاتصال بالخادم.';
                }
                chat.scrollTop = chat.scrollHeight;
            }
        </script>
    </body>
    </html>
    """
