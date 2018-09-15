# middle_out_backend


## REST server
Small flask app for communicating with the client

### Setup
```
# Setup virtualenv
python -m venv venv
source venv/bin/activate

# Install deps
pip install -r requirements.txt

# Run server
(sudo) python server.py
```

### Commands
```
POST /upload
form-data:
framecount: (# of frames)
n (0 <= n < framecount): (frame image)
```
