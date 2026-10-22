
# SignBridge

A limited-vocabulary ISL recognition prototype for browser-based
communication.

## Current status
Step 0: project contracts and environment.

## Privacy
Webcam frames and landmarks should remain on-device during inference.
Only the minimum required text should be sent to cloud services.

## Development
Activate the Python virtual environment, then run:

    python -m pytest -q

See contracts/ for the landmark and prediction-event specifications.
