# Fractal Explorer - Phase 1

一個用 Rust + wgpu 實現的高性能即時 Mandelbrot 集探索器。

## 功能特性

✅ **已實現 (Phase 1):**
- GPU 加速的 Mandelbrot 渲染 (Compute Shader)
- 即時互動式縮放和平移
- 平滑著色算法（消除色帶）
- 鍵盤快捷鍵控制
- 60 FPS @ 1920×1080 性能

## 編譯與運行

### 編譯
```bash
cargo build --release
```

### 運行
```bash
cargo run --release
```

程式會開啟一個 1920×1080 的視窗，顯示經典的 Mandelbrot 集合。

## 操作方式

### 滑鼠控制
- **左鍵拖拽**: 平移視圖
- **滾輪**: 以游標為中心縮放
  - 向上滾動: 放大
  - 向下滾動: 縮小

### 鍵盤快捷鍵
- **R**: 重置視圖到初始位置 (-0.5, 0.0)
- **↑**: 增加最大迭代次數 (+64)
- **↓**: 減少最大迭代次數 (-64)
- **Esc**: 退出程式

## 技術細節

### 架構
- **視窗管理**: winit 0.30
- **GPU 渲染**: wgpu 22 (Metal backend on macOS)
- **著色器**: WGSL (WebGPU Shading Language)
- **數學庫**: glam 0.29

### 渲染管線
1. **Compute Shader**: 計算每個像素的迭代次數
   - Workgroup size: 16×16
   - 平滑迭代算法: `iter + 1 - log2(log2(|z|))`
2. **Fullscreen Quad**: 將計算結果渲染到螢幕
   - Texture format: RGBA8Unorm
   - Sampling: Linear filtering

### 性能
- **解析度**: 1920×1080
- **目標幀率**: 60 FPS
- **最大迭代數**: 256 (可調整至 4096)
- **縮放精度**: f64 內部精度，f32 GPU 精度

### 已知限制
- 縮放超過 ~1e6 會出現精度問題 (Phase 3 將實現 emulated double)
- Phase 1 只支援 Mandelbrot 集 (Phase 2 將加入 Julia、Burning Ship 等)
- 無 UI 控制面板 (Phase 2 將整合 egui)

## 專案結構
```
src/
├── main.rs              # 入口點、事件循環
├── camera.rs            # 視圖變換 (screen ↔ complex plane)
├── renderer/
│   ├── mod.rs
│   ├── gpu.rs           # wgpu 初始化
│   ├── compute.rs       # Compute pipeline
│   ├── render.rs        # Render pipeline
│   └── uniforms.rs      # GPU uniform 定義
└── shaders/
    ├── mandelbrot.wgsl  # Mandelbrot compute shader
    └── fullscreen.wgsl  # Fullscreen quad shader
```

## 下一步 (Phase 2)

- [ ] 整合 egui UI 控制面板
- [ ] 多種 fractal 類型 (Julia、Burning Ship、Tricorn)
- [ ] 配色方案選擇器
- [ ] 參數即時調整
- [ ] 更多鍵盤快捷鍵

## 授權

MIT License

---

**Phase 1 完成日期**: 2026-02-16
**開發環境**: macOS, Apple M1, Rust 2021 Edition
