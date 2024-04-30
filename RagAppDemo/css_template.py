css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #b8d4ff
}
.chat-message.bot {
    background-color: #f9bdbd
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://t3.ftcdn.net/jpg/05/73/14/38/360_F_573143889_NVvKlj8AGINKQyT7Pr3tkvCScXShff0F.jpg">
    </div>
    <div class="message">{{message}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://storage.prompt-hunt.workers.dev/clgtupr4u003elj08xll7nnpt_1">
    </div>    
    <div class="message">{{message}}</div>
</div>
'''