import argparse
import json
from pathlib import Path

from config_utils import load_config
from eval_autoencoder import first_layer_pca, load_year
from utils import load_model_and_preprocessor


def write_html(pc1, pc2, scores, lat, lon, date, output_path):
    payload = {
        "date": date,
        "lat": lat.tolist(),
        "lon": lon.tolist(),
        "pc1Map": pc1.tolist(),
        "pc2Map": pc2.tolist(),
        "scores": scores.tolist(),
    }
    html = HTML_TEMPLATE.replace("__DATA__", json.dumps(payload))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>First-layer PCA</title>
  <style>
    body { margin: 0; font-family: system-ui, sans-serif; color: #223; }
    main { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 16px; }
    canvas { width: 100%; height: auto; border: 1px solid #ccc; }
    .label { margin: 0 0 8px; font-weight: 600; }
    #info { grid-column: 1 / -1; min-height: 1.5em; }
  </style>
</head>
<body>
<main>
  <section>
    <p class="label" id="pc1-map-title"></p>
    <canvas id="pc1-map" width="900" height="450"></canvas>
  </section>
  <section>
    <p class="label" id="pc2-map-title"></p>
    <canvas id="pc2-map" width="900" height="450"></canvas>
  </section>
  <section>
    <p class="label">First-layer PCA</p>
    <canvas id="pca" width="600" height="600"></canvas>
  </section>
  <div id="info"></div>
</main>
<script>
const data = __DATA__;
const pc1MapCanvas = document.getElementById("pc1-map");
const pc2MapCanvas = document.getElementById("pc2-map");
const pcaCanvas = document.getElementById("pca");
const pc1MapCtx = pc1MapCanvas.getContext("2d");
const pc2MapCtx = pc2MapCanvas.getContext("2d");
const pcaCtx = pcaCanvas.getContext("2d");
const pc1MapBase = document.createElement("canvas");
const pc2MapBase = document.createElement("canvas");
const pcaBase = document.createElement("canvas");
pc1MapBase.width = pc1MapCanvas.width;
pc1MapBase.height = pc1MapCanvas.height;
pc2MapBase.width = pc2MapCanvas.width;
pc2MapBase.height = pc2MapCanvas.height;
pcaBase.width = pcaCanvas.width;
pcaBase.height = pcaCanvas.height;
const pc1MapBaseCtx = pc1MapBase.getContext("2d");
const pc2MapBaseCtx = pc2MapBase.getContext("2d");
const pcaBaseCtx = pcaBase.getContext("2d");
const info = document.getElementById("info");
document.getElementById("pc1-map-title").textContent = `PC1 map; ${data.date}`;
document.getElementById("pc2-map-title").textContent = `PC2 map; ${data.date}`;

const lat = data.lat, lon = data.lon, scores = data.scores;
const pc1Map = data.pc1Map, pc2Map = data.pc2Map;
const margin = 36;
const pc1 = scores.map(s => s[0]), pc2 = scores.map(s => s[1]);
const pc1Min = pc1.reduce((a, b) => Math.min(a, b), Infinity);
const pc1Max = pc1.reduce((a, b) => Math.max(a, b), -Infinity);
const pc2Min = pc2.reduce((a, b) => Math.min(a, b), Infinity);
const pc2Max = pc2.reduce((a, b) => Math.max(a, b), -Infinity);
const mapAbsMax = [pc1Map, pc2Map].reduce(
  (best, map) => map.reduce(
    (mapBest, row) => row.reduce((rowBest, value) => Math.max(rowBest, Math.abs(value)), mapBest),
    best
  ),
  0
) || 1;
let selected = {i: Math.floor(lat.length / 2), j: Math.floor(lon.length / 2)};

function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
function lerp(a, b, t) { return a + (b - a) * t; }
function color(value) {
  const t = clamp((value / mapAbsMax + 1) / 2, 0, 1);
  const lo = [49, 54, 149], mid = [255, 255, 255], hi = [165, 0, 38];
  const a = t < 0.5 ? lo : mid, b = t < 0.5 ? mid : hi, u = t < 0.5 ? t * 2 : (t - 0.5) * 2;
  return `rgb(${lerp(a[0], b[0], u)},${lerp(a[1], b[1], u)},${lerp(a[2], b[2], u)})`;
}
function xMap(canvas, j) { return (j + 0.5) / lon.length * canvas.width; }
function yMap(canvas, i) { return (i + 0.5) / lat.length * canvas.height; }
function xPca(x) { return margin + (x - pc1Min) / (pc1Max - pc1Min || 1) * (pcaCanvas.width - 2 * margin); }
function yPca(y) { return pcaCanvas.height - margin - (y - pc2Min) / (pc2Max - pc2Min || 1) * (pcaCanvas.height - 2 * margin); }
function flatIndex(i, j) { return i * lon.length + j; }

function drawMapBase(canvas, ctx, values) {
  const cellW = canvas.width / lon.length, cellH = canvas.height / lat.length;
  for (let i = 0; i < lat.length; i++) {
    for (let j = 0; j < lon.length; j++) {
      ctx.fillStyle = color(values[i][j]);
      ctx.fillRect(j * cellW, i * cellH, Math.ceil(cellW), Math.ceil(cellH));
    }
  }
}
function drawPcaBase() {
  pcaBaseCtx.clearRect(0, 0, pcaCanvas.width, pcaCanvas.height);
  pcaBaseCtx.strokeStyle = "#bbb";
  pcaBaseCtx.strokeRect(margin, margin, pcaCanvas.width - 2 * margin, pcaCanvas.height - 2 * margin);
  pcaBaseCtx.fillStyle = "rgba(30, 90, 170, 0.25)";
  for (const s of scores) pcaBaseCtx.fillRect(xPca(s[0]), yPca(s[1]), 2, 2);
}
function drawMap(canvas, ctx, base) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(base, 0, 0);
  ctx.strokeStyle = "black";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.arc(xMap(canvas, selected.j), yMap(canvas, selected.i), 7, 0, 2 * Math.PI);
  ctx.stroke();
}
function drawPca() {
  pcaCtx.clearRect(0, 0, pcaCanvas.width, pcaCanvas.height);
  pcaCtx.drawImage(pcaBase, 0, 0);
  const s = scores[flatIndex(selected.i, selected.j)];
  pcaCtx.strokeStyle = "black";
  pcaCtx.lineWidth = 2;
  pcaCtx.beginPath();
  pcaCtx.arc(xPca(s[0]), yPca(s[1]), 8, 0, 2 * Math.PI);
  pcaCtx.stroke();
}
function updateInfo() {
  const s = scores[flatIndex(selected.i, selected.j)];
  info.textContent = `lat=${lat[selected.i].toFixed(2)}, lon=${lon[selected.j].toFixed(2)}, PC1=${s[0].toPrecision(4)}, PC2=${s[1].toPrecision(4)}`;
}
function draw() {
  drawMap(pc1MapCanvas, pc1MapCtx, pc1MapBase);
  drawMap(pc2MapCanvas, pc2MapCtx, pc2MapBase);
  drawPca();
  updateInfo();
}

function selectFromMap(canvas, event) {
  const rect = canvas.getBoundingClientRect();
  const x = (event.clientX - rect.left) / rect.width * canvas.width;
  const y = (event.clientY - rect.top) / rect.height * canvas.height;
  selected = {
    i: Math.max(0, Math.min(lat.length - 1, Math.floor(y / canvas.height * lat.length))),
    j: Math.max(0, Math.min(lon.length - 1, Math.floor(x / canvas.width * lon.length))),
  };
  draw();
}

pc1MapCanvas.addEventListener("click", event => selectFromMap(pc1MapCanvas, event));
pc2MapCanvas.addEventListener("click", event => selectFromMap(pc2MapCanvas, event));

drawMapBase(pc1MapCanvas, pc1MapBaseCtx, pc1Map);
drawMapBase(pc2MapCanvas, pc2MapBaseCtx, pc2Map);
drawPcaBase();
draw();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", required=True)
    parser.add_argument("--checkpoint_path")
    parser.add_argument("--date", required=True, help="YYYY-MM")
    parser.add_argument("--data_path", default=None)
    parser.add_argument("--output_path", default="autoencoder_interactive.html")
    args = parser.parse_args()

    config = load_config(args.config_file)
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path
        else Path(config.train.checkpoint_dir) / "best_model.pt"
    )
    model, preprocessor, _ = load_model_and_preprocessor(
        config, checkpoint_path, downscaling=False
    )
    ds = load_year(
        config, args.date, Path(args.data_path or config.dataset.era5.raw_path)
    )
    scores, pc1, pc2, lat, lon = first_layer_pca(
        ds, model, preprocessor, config, args.date
    )
    write_html(pc1, pc2, scores, lat, lon, args.date, Path(args.output_path))


if __name__ == "__main__":
    main()
