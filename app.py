from flask import Flask, jsonify, request
import Process

app = Flask(__name__)

@app.route('/chatbot', methods=["GET", "POST"])
def chatbot_response():
    if request.method == 'GET':
        the_question = request.args.get('question', '')
    elif request.method == 'POST':
        the_question = request.form['question']
    else:
        return jsonify({'error': 'Invalid request method'})

    response = Process.chatbot_response(the_question)

    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='5000', debug=True)
