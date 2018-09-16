from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import base64
import sqlite3
import hashlib

app = Flask(__name__)
conn = sqlite3.connect('frames.db')
c = conn.cursor()
c.execute('CREATE TABLE frames (index real, filename text)')
c.commit()

@app.route('/upload', methods=['POST'])
def upload():
    json = request.get_json()
    index = int(json['index'])
    raw = BytesIo(base64.b64decode(json['uri'].split(',')[1]))
    img = Image.open(raw)
    checksum = hashlib.sha256()
    checksum.update(raw)
    name = checksum.hexdigest()
    img.save('/var/www/static/{:s}.png'.format(name))
    c.execute('INSERT INTO frames (?,?)', (index, name,))
    c.commit()
    return jsonify({'file': '{:s}.png'.format(name)}), 200

if __name__ == '__main__':
    app.run()
