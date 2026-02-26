# Fractal Explorer — Claude Code 計劃書

## 專案概覽

一個用 Rust 實現的即時互動式 Fractal 探索器。
目標是高性能 GPU 渲染 + 直觀的 UI 控制面板，讓用戶可以即時探索不同 fractal 的細節。

---

## 技術棧

| 層次 | 技術 | 用途 |
|------|------|------|
| 視窗管理 | `winit` | 跨平台視窗、輸入事件 |
| GPU 渲染 | `wgpu` | Compute shader 計算 fractal |
| UI 框架 | `egui` + `egui-wgpu` + `egui-winit` | 控制面板、參數調整 |
| CPU 並行 | `rayon` | CPU fallback，多核心並行 |
| 數學庫 | `glam` | Vec2、Mat4 等數學運算 |
| 圖像輸出 | `image` | 導出高解析度 PNG |
| 影片編碼 | `ffmpeg-next`（Phase 3 加入） | 錄製深度縮放動畫 |

### Cargo.toml
```toml
[package]
name = "fractal-explorer"
version = "0.1.0"
edition = "2021"

[dependencies]
winit = "0.29"
wgpu = "0.18"
egui = "0.29"
egui-wgpu = "0.29"
egui-winit = "0.29"
rayon = "1.8"
glam = "0.25"
image = "0.24"
pollster = "0.3"
bytemuck = { version = "1.14", features = ["derive"] }
log = "0.4"
env_logger = "0.10"
```

---

## 專案結構
```
fractal-explorer/
├── CLAUDE.md
├── Cargo.toml
├── src/
│   ├── main.rs
│   ├── app.rs                 # 主應用狀態、事件循環
│   ├── camera.rs              # 視圖變換：center、zoom
│   ├── export.rs              # PNG 導出、影片錄製
│   ├── renderer/
│   │   ├── mod.rs
│   │   ├── gpu.rs             # wgpu 初始化、pipeline
│   │   ├── compute.rs         # Compute shader 管理
│   │   └── texture.rs         # Output texture
│   ├── fractals/
│   │   ├── mod.rs             # FractalType enum + Fractal trait
│   │   ├── mandelbrot.rs
│   │   ├── julia.rs
│   │   ├── burning_ship.rs
│   │   ├── tricorn.rs
│   │   ├── buddhabrot.rs      # 不同渲染邏輯
│   │   ├── nova.rs
│   │   └── lyapunov.rs
│   ├── ui/
│   │   ├── mod.rs
│   │   ├── control_panel.rs   # egui 側邊欄
│   │   ├── viewport.rs        # 滑鼠互動
│   │   ├── palette_editor.rs   # 自訂 Palette 編輯器
│   │   └── color_editor.rs
│   ├── coloring/
│   │   ├── mod.rs             # ColorScheme enum (Preset/Custom)
│   │   ├── palette.rs         # Palette、ColorStop、LUT 生成
│   │   └── presets.rs         # 內建配色方案預設
│   └── shaders/
│       ├── common.wgsl        # 共用函數（smooth iter、colorize）
│       ├── fullscreen.wgsl    # Fullscreen quad
│       ├── mandelbrot.wgsl
│       ├── julia.wgsl
│       └── burning_ship.wgsl
├── assets/
│   └── palettes/
└── tests/
    ├── fractal_correctness.rs
    └── performance.rs
```

---

## 核心設計

### Fractal Trait
```rust
pub trait Fractal: Send + Sync {
    fn shader_source(&self) -> &str;
    fn uniform_data(&self) -> Vec<u8>;
    fn iterate_cpu(&self, cx: f64, cy: f64, max_iter: u32) -> u32;
    fn default_center(&self) -> glam::DVec2;
    fn default_zoom(&self) -> f64;
    fn name(&self) -> &str;
    fn parameters(&self) -> Vec<Parameter> { vec![] }
}
```

### FractalType Enum
```rust
#[derive(Clone, PartialEq)]
pub enum FractalType {
    Mandelbrot,
    Julia { c_real: f64, c_imag: f64 },
    BurningShip,
    Tricorn,
    Nova { c_real: f64, c_imag: f64 },
    Lyapunov { sequence: String },
    Buddhabrot { samples: u32 },
}
```

### GPU Uniform 結構
```rust
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct FractalUniforms {
    center: [f32; 2],
    zoom: f32,
    aspect_ratio: f32,
    max_iter: u32,
    fractal_type: u32,
    c_real: f32,
    c_imag: f32,
    color_scheme: u32,
    _padding: [u32; 3],    // 必須維持 16-byte 對齊
}
```

### Camera
```rust
pub struct Camera {
    pub center: glam::DVec2,   // f64 保持深度縮放精度
    pub zoom: f64,
    pub screen_size: glam::UVec2,
}

impl Camera {
    pub fn screen_to_complex(&self, screen_pos: glam::Vec2) -> glam::DVec2;
    pub fn zoom_at(&mut self, screen_pos: glam::Vec2, delta: f64);
    pub fn pan(&mut self, delta: glam::Vec2);
}
```

---

## 渲染管線
```
每幀流程：

1. 收集輸入事件（winit）
2. 更新 Camera（縮放、平移）
3. 更新 FractalUniforms buffer（queue.write_buffer）
4. Dispatch compute shader → storage texture（RGBA8）
5. Fullscreen quad render（texture → 螢幕）
6. 渲染 egui UI（疊加）
7. Present
```

### Compute Shader 架構（WGSL）
```wgsl
@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var output_texture: texture_storage_2d<rgba8unorm, write>;

fn smooth_iter(iter: u32, zx: f32, zy: f32, max_iter: u32) -> f32 {
    if iter == max_iter { return 0.0; }
    let log_zn = log2(zx * zx + zy * zy) / 2.0;
    let nu = log2(log_zn / log2(2.0));
    return f32(iter) + 1.0 - nu;
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dims = textureDimensions(output_texture);
    if id.x >= dims.x || id.y >= dims.y { return; }

    let uv = (vec2<f32>(id.xy) - vec2<f32>(dims) / 2.0)
             / f32(dims.y) / uniforms.zoom;
    let c = uniforms.center + uv * vec2<f32>(uniforms.aspect_ratio, -1.0);

    var smooth_val: f32;
    switch uniforms.fractal_type {
        case 0u: { smooth_val = mandelbrot(c.x, c.y); }
        case 1u: { smooth_val = julia(c.x, c.y, uniforms.c_real, uniforms.c_imag); }
        case 2u: { smooth_val = burning_ship(c.x, c.y); }
        case 3u: { smooth_val = tricorn(c.x, c.y); }
        default: { smooth_val = 0.0; }
    }

    textureStore(output_texture, id.xy, colorize(smooth_val, uniforms.color_scheme));
}
```

---

## UI 佈局
```
┌──────────────┬────────────────────────────────┐
│ 控制面板      │                                │
│              │                                │
│ Fractal Type │        Fractal 渲染區           │
│ ○ Mandelbrot │                                │
│ ● Julia Set  │                                │
│ ○ Burning..  │                                │
│              │                                │
│ Parameters   │                                │
│ c_real ───── │                                │
│ c_imag ───── │                                │
│ [Animate c]  │                                │
│              │                                │
│ Max Iter ─── │                                │
│              │                                │
│ Color Scheme │                                │
│ [Smooth]     │                                │
│ [Fire]       │                                │
│ [Ocean]      │                                │
│              │                                │
│ Center:      │                                │
│ (-0.5, 0.0)  │                                │
│ Zoom: 1.0x   │                                │
│ [Reset View] │                                │
│              │                                │
│ [Save PNG]   │                                │
│ [Record]     │                                │
└──────────────┴────────────────────────────────┘
```

### 鍵盤快捷鍵

| 按鍵 | 動作 |
|------|------|
| `1` | Mandelbrot Set |
| `2` | Julia Set |
| `3` | Burning Ship |
| `4` | Tricorn |
| `5` | Buddhabrot |
| `+` / `-` | 增減最大迭代次數 (+/- 64) |
| `R` | 重置視圖 |
| `P` | 截圖 PNG (1080p) |
| `C` | 切換配色方案 |
| `Q` / `E` | 旋轉視圖 (左/右) |
| `T` / `G` | 縮放 (放大/縮小) |
| `J` / `L` | Julia c 實部 -/+ (Julia/Nova模式) |
| `I` / `K` | Julia c 虛部 +/- |
| `L` | 切換 Mandelbrot/Julia 聯動模式 (非Julia/Nova模式時) |

### 滑鼠操作

| 操作 | 動作 |
|------|------|
| 左鍵拖拉 | 平移視圖 |
| 滾輪 | 以游標為中心縮放 |
| 雙擊 | 以點擊位置縮放 2x |
| 右鍵（Julia 模式） | 設定 c 為游標位置 |

---

## 實現階段

### Phase 1：基礎（能跑 Mandelbrot）
- [x] wgpu 初始化（instance, adapter, device, queue, surface）
- [x] Camera：screen → complex 座標轉換
- [x] mandelbrot.wgsl compute shader
- [x] Compute pipeline + storage texture
- [x] Fullscreen quad render pipeline
- [x] winit 事件循環
- [x] 滑鼠縮放 + 拖拉平移

**完成標準：** 60 FPS @ 1080p 流暢探索 Mandelbrot

### Phase 2：多 Fractal + UI
- [x] 整合 egui
- [x] FractalType enum + shader dispatch
- [x] Julia Set（互動式 c 參數）
- [x] Burning Ship、Tricorn
- [x] Smooth Coloring（消除色帶）
- [x] 配色方案選擇器
- [x] 鍵盤快捷鍵

**完成標準：** 即時切換 fractal，Julia c 參數可用滑鼠調整

### Phase 3：進階功能
- [x] 高解析度 PNG 導出（4K / 8K）
- [x] f64 精度 / emulated double（深度縮放）
- [x] Buddhabrot（accumulation buffer 模式）
- [x] Nova Fractal
- [x] 自訂 Palette 編輯器
- [x] Mandelbrot / Julia 聯動模式
- [x] 影片錄製

**完成標準：** 支援至少 6 種 fractal，可導出高品質輸出

**Buddhabrot 實現細節：**
- 雙通道渲染：accumulation pass + tonemap pass
- 原子累加緩衝區（atomic u32 per pixel）
- 每幀 65536 個隨機樣本漸進式累積
- 視圖變更時自動清除累積緩衝區
- 支援旋轉、縮放、平移
- 專用配色方案（Nebula、Fire、Ocean、Grayscale）

**自訂 Palette 編輯器實現細節：**
- GPU storage buffer LUT：256 個 packed RGBA8 u32（1024 bytes）
- CPU 線性插值生成 LUT，上傳至 GPU storage buffer
- 所有 11 個 shader 使用 `sample_palette()` 從 LUT 取樣，取代硬編碼 colorize 函數
- binding 配置：標準 layout binding 2、perturbation/tonemap layout binding 3
- 7 個內建預設 + 自訂編輯器（egui Window，支援拖曳色標、色彩選擇器）
- 調色盤變更時即時更新，無需重新編譯 shader

### Phase 4：優化
- [ ] Perturbation Theory（超深度縮放 1e-100+）
- [ ] 漸進式渲染（低解析度先顯示，逐步細化）
- [ ] 書籤系統（保存 / 載入位置）
- [ ] 參數動畫
- [ ] WebAssembly 支援

### Phase 5：低優先級（Low Priority）
- [ ] Burning Ship perturbation theory f64（目前使用 Dekker emulated double，精度有限）
- [ ] Tricorn perturbation theory f64（目前使用 Dekker emulated double，精度有限）

---

## 精度問題

f32 在縮放超過約 **1e6** 時會出現像素化失真，解決方案：

1. **f64 CPU fallback**：自動切換，簡單但較慢
2. **Emulated double（GPU）**：兩個 f32 模擬 f64，性能損失約 4x
3. **Perturbation Theory**：支援無限深度，最複雜

建議：Phase 1-2 用 f32 → Phase 3 加 emulated double → Phase 4 考慮 perturbation

---

## 性能目標

| 場景 | 目標 | 解析度 |
|------|------|--------|
| 即時探索 | 60 FPS | 1920×1080 |
| 即時探索 | 30 FPS | 2560×1440 |
| 靜態渲染 | < 1 秒 | 3840×2160 |
| 靜態渲染 | < 5 秒 | 7680×4320 |

---

## 程式碼規範

- Public API 必須有 `///` doc comment
- `FractalUniforms` 必須維持 16-byte 對齊
- Shader 改動時同步更新 Rust `bytemuck` 結構

**新增 fractal 的 checklist：**
1. `FractalType` enum 加 variant
2. 對應 `.wgsl` shader
3. `Fractal` trait 實現
4. UI 控制欄加選項
5. 鍵盤快捷鍵
6. `tests/fractal_correctness.rs` 加測試

---

## 已知限制

- **Mandelbulb**（3D）需要 ray marching，是獨立大項目，不列入本計劃
- **Lyapunov** 不是逃逸時間演算法，渲染邏輯需單獨處理
- **Buddhabrot** 需 Monte Carlo 取樣，用 accumulation buffer 而非 per-pixel shader

---

*版本：1.0 ｜ Rust 2021 Edition ｜ 目標平台：macOS / Linux / Windows*
