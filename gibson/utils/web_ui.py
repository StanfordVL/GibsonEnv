from flask import Flask, render_template, Response
import sys
import pickle
import socket
import zmq

app = Flask(__name__)

port = "5552"
if len(sys.argv) > 1:
    port =  sys.argv[1]

if len(sys.argv) > 2:
    port_web =  sys.argv[2]
    port_web = int(port_web)
else:
    port_web = 5001
# Socket to talk to server
context = zmq.Context()
socket = context.socket(zmq.SUB)

print("Collecting updates from server...")
socket.connect ("tcp://localhost:%s" % port)


topicfilter = b"ui"
socket.setsockopt(zmq.SUBSCRIBE, topicfilter)

@app.route('/')
def index():
    return render_template('index.html')

def gen():
    while True:
        string = socket.recv()
        
        data = string[2:]
        #from IPython import embed; embed()
        frame = pickle.loads(data)[-1]
        
        frame = frame.tobytes()
        #print(frame.shape)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port_web, debug=False)
