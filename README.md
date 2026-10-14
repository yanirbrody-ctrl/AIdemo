# Head Dodge

A browser game you control by moving or tilting your head (webcam + MediaPipe Face Landmarker, all running locally in the browser). Arrow keys / A-D also work at any time.

## Run locally

The camera only works over HTTPS or `localhost`, so serve the folder rather than opening `index.html` directly:

```sh
python3 -m http.server 8000
```

Then open http://localhost:8000.

Static files only — it also runs as-is on GitHub Pages.
