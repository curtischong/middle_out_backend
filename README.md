# middle_out_backend

Interested in the front-end? Visit it [here](https://github.com/marceloneil/piepiper)

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
