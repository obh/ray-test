"""Launch Ray Serve with explicit host binding for Docker."""
import ray
from ray import serve

ray.init(address="auto")
serve.start(http_options={"host": "0.0.0.0", "port": 8000})

from server import app
serve.run(app)

# Block forever
import signal
signal.pause()
