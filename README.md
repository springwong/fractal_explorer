# Fractal Explorer

A real-time interactive fractal explorer built with Rust, wgpu, and egui. Runs natively and in the browser via WebAssembly + WebGPU.

## Features

- 6 fractal types: Mandelbrot, Julia, Burning Ship, Tricorn, Buddhabrot, Nova
- GPU-accelerated rendering via compute shaders (WGSL)
- f64 emulated double precision for deep zoom
- Perturbation theory for Mandelbrot/Julia at extreme zoom levels
- Custom palette editor with GPU LUT-based coloring
- Mandelbrot/Julia linked split-view mode
- PNG export (up to 8K) and video recording (native only)
- WebAssembly support for browser deployment

## Running Natively

```bash
cargo run --release
```

## Running in Browser (Development)

```bash
rustup target add wasm32-unknown-unknown
cargo install trunk --locked
trunk serve
```

Open http://127.0.0.1:8080 in a WebGPU-capable browser (Chrome 113+, Edge 113+).

## Deploying to GitHub Pages

### Prerequisites

- `trunk` installed (`cargo install trunk --locked`)
- `wasm32-unknown-unknown` target installed (`rustup target add wasm32-unknown-unknown`)
- A GitHub repository with Pages enabled on the `gh-pages` branch

### Steps

1. **Build the release WASM bundle:**

   ```bash
   trunk build --release
   ```

   This produces optimized files in the `dist/` directory.

2. **Switch to the `gh-pages` branch:**

   ```bash
   git stash            # stash any uncommitted changes on main
   git switch gh-pages
   ```

3. **Replace old files with new build:**

   ```bash
   git rm -f *.js *.wasm index.html 2>/dev/null
   cp dist/* .
   git add index.html fractal-explorer-*.js fractal-explorer-*_bg.wasm
   ```

4. **Commit and push:**

   ```bash
   git commit -m "Deploy fractal explorer to GitHub Pages"
   git push origin gh-pages
   ```

5. **Switch back to main:**

   ```bash
   git switch main
   git stash pop        # restore stashed changes if any
   ```

### One-liner

```bash
trunk build --release && git stash && git switch gh-pages && git rm -f *.js *.wasm index.html 2>/dev/null && cp dist/* . && git add index.html fractal-explorer-*.js fractal-explorer-*_bg.wasm && git commit -m "Deploy" && git push origin gh-pages && git switch main && git stash pop
```

### GitHub Pages Setup (first time only)

1. Go to your repo **Settings > Pages**
2. Under **Source**, select **Deploy from a branch**
3. Pick branch **`gh-pages`** / **`/ (root)`**
4. Click **Save**

The site will be available at `https://<username>.github.io/<repo-name>/`.

### Note on `public_url`

`Trunk.toml` has `public_url = "/fractal_explorer/"` so asset paths match the GitHub Pages subdirectory. If you deploy to a different path, update this value accordingly. For local development with `trunk serve`, you can override it:

```bash
trunk serve --public-url /
```

## Controls

| Key | Action |
|-----|--------|
| `1`-`6` | Switch fractal type |
| `+` / `-` | Increase / decrease max iterations |
| `R` | Reset view |
| `P` | Export PNG (1080p) |
| `C` | Cycle color scheme |
| `Q` / `E` | Rotate view |
| `T` / `G` | Zoom in / out |
| `L` | Toggle linked mode |
| Mouse drag | Pan |
| Scroll wheel | Zoom at cursor |
| Right-click | Set Julia c parameter |

## License

MIT License
