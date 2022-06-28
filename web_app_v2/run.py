from web_experiment import create_app, socketio

app = create_app(debug=False)

if __name__ == '__main__':
  socketio.run(app)
