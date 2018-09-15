from flask import Flask, request, jsonify
from PIL import Image
import numpy

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload():
    framecount = int(request.form['framecount'])
    frames = [None] * framecount
    for i in range(framecount):
        img = Image.open(request.files[str(i)]).convert('1')
        try: 
            assert img.size[0] == 1397
            assert img.size[1] == 512
        except:
            return jsonify({
                'success': False,
                'error': 'Invalid image size',
                'index': i
            }), 400
        frames[i] = numpy.array(img.getdata()).reshape(img.size[0], img.size[1], 1)
    return jsonify({'success': True}), 200

if __name__ == '__main__':
    app.run()
