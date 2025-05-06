from flask import Flask, render_template, request, jsonify
from Deploy1 import get_chatbot_response

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json.get("message")
    print(f"Received message: {user_message}")  # Debugging log

    if not user_message:
        return jsonify({"response": "Error: Empty message received!"})

    bot_reply = get_chatbot_response(user_message)
    print(f"Chatbot response: {bot_reply}")  # Debugging log

    return jsonify({"response": bot_reply})

if __name__ == "__main__":
    app.run(debug=True, port=8000)
