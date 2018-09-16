from flask import Flask, request, jsonify, g
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import base64
import sqlite3
import hashlib

app = Flask(__name__)
# CORS(app)

conn = sqlite3.connect('frames.db')
c = conn.cursor()
c.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='frames'")
if not c.fetchone():
    c.execute('CREATE TABLE frames (frame real, filename text)')
conn.commit()
c.close()
conn.close()

def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect('frames.db')
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.commit()
        db.close()

@app.route('/upload', methods=['POST'])
def upload():
    data = request.get_json()
    index = int(data['index'])
    raw = base64.b64decode(data['uri'].split(',')[1])
    checksum = hashlib.sha256()
    checksum.update(raw)
    name = checksum.hexdigest()
    img = Image.open(BytesIO(raw))
    bg = Image.new('1', (img.size[0], img.size[1]), 256)
    bg.paste(img, mask=img)
    bg.save('/var/www/static/{:s}.png'.format(name), 'PNG')
    c = get_db().cursor()
    c.execute('INSERT INTO frames VALUES (?,?)', (index, name,))
    c.close()
    return jsonify({'file': 'http://api.piepiper.1lab.me/static/{:s}.png'.format(name)}), 200

if __name__ == '__main__':
    app.run()
