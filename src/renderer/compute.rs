use super::fractal_pass::{
    create_compute_pipeline, create_palette_buffer, create_pipeline_layout, output_texture_entry, storage_entry,
    uniform_entry, workgroups, FractalPipelines, FractalTarget, PALETTE_LUT_BYTES,
};
use super::shader;
use super::uniforms::{FractalUniforms, View};
use crate::fractals::FractalType;
use glam::Vec2;

/// Random orbits traced per frame by the Buddhabrot accumulation pass
const BUDDHABROT_SAMPLES_PER_FRAME: u32 = 65536;
/// Workgroup size of the accumulation shader (one orbit per invocation)
const BUDDHABROT_WORKGROUP_SIZE: u32 = 256;

/// All compute resources of the live view: the main fractal target, the Julia target
/// of the linked split view, and the Buddhabrot accumulation passes.
pub struct ComputePipeline {
    pipelines: FractalPipelines,
    /// Palette LUT (256 packed RGBA8 entries), shared by every pass
    palette_buffer: wgpu::Buffer,
    /// Main view; `main.view` is what the render pipeline samples
    pub main: FractalTarget,
    /// Julia side of the linked Mandelbrot / Julia split view
    pub julia: FractalTarget,
    buddhabrot: BuddhabrotPasses,
    /// Total accumulated sample batches for Buddhabrot
    pub accum_sample_count: u32,
    /// Whether the main view was last rendered with its deep-zoom shader
    pub using_f64: bool,
}

impl ComputePipeline {
    /// Create all compute resources for a `width` x `height` view
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        let pipelines = FractalPipelines::new(device);
        let palette_buffer = create_palette_buffer(device);
        let (main, julia) = Self::create_targets(device, &pipelines, &palette_buffer, width, height);
        let buddhabrot = BuddhabrotPasses::new(device, &main, &palette_buffer);

        log::info!("Compute pipeline created: {}x{}", width, height);

        Self { pipelines, palette_buffer, main, julia, buddhabrot, accum_sample_count: 0, using_f64: false }
    }

    fn create_targets(
        device: &wgpu::Device,
        pipelines: &FractalPipelines,
        palette_buffer: &wgpu::Buffer,
        width: u32,
        height: u32,
    ) -> (FractalTarget, FractalTarget) {
        let usage = wgpu::TextureUsages::TEXTURE_BINDING;
        (
            FractalTarget::new(device, pipelines, palette_buffer, "Main View", width, height, usage),
            FractalTarget::new(device, pipelines, palette_buffer, "Linked Julia View", width, height, usage),
        )
    }

    /// Upload palette LUT data to GPU
    pub fn upload_palette(&self, queue: &wgpu::Queue, data: &[u8; PALETTE_LUT_BYTES as usize]) {
        queue.write_buffer(&self.palette_buffer, 0, data);
    }

    /// Render `fractal` into the main view, switching precision automatically with zoom
    pub fn render_main(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        fractal: &FractalType,
        view: &View,
    ) {
        match self.main.render(device, queue, encoder, &mut self.pipelines, fractal, view) {
            Ok(deep) => {
                if self.using_f64 != deep {
                    log::info!("Precision mode switched to: {}", if deep { "deep zoom" } else { "f32" });
                }
                self.using_f64 = deep;
            }
            Err(e) => log::error!("Fractal pass failed: {e}"),
        }
    }

    /// Render the Julia set for `c` into the linked view's Julia side
    pub fn render_julia(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        c: Vec2,
        view: &View,
    ) {
        let fractal = FractalType::Julia { c };
        if let Err(e) = self.julia.render(device, queue, encoder, &mut self.pipelines, &fractal, view) {
            log::error!("Linked Julia pass failed: {e}");
        }
    }

    /// Clear the Buddhabrot accumulation buffer (zero-fill)
    pub fn clear_accum_buffer(&mut self, queue: &wgpu::Queue) {
        let zeros = vec![0u8; self.buddhabrot.accum_buffer.size() as usize];
        queue.write_buffer(&self.buddhabrot.accum_buffer, 0, &zeros);
        self.accum_sample_count = 0;
        log::info!("Buddhabrot accumulation buffer cleared");
    }

    /// Add one batch of Buddhabrot samples for `view` and tonemap the result into the
    /// main view. `seed` must change every frame.
    pub fn render_buddhabrot(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &View,
        seed: u32,
    ) {
        let (width, height) = (self.main.width(), self.main.height());
        let mut uniforms = FractalUniforms::for_view(&FractalType::Buddhabrot, view, width, height);
        uniforms.ref_escape_iter = seed;
        uniforms.extra = [width, height, self.accum_sample_count];
        queue.write_buffer(&self.main.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        self.buddhabrot.dispatch(device, encoder, width, height);
        self.accum_sample_count += 1;
        self.using_f64 = false;
    }

    /// Recreate textures on resize
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        (self.main, self.julia) = Self::create_targets(device, &self.pipelines, &self.palette_buffer, width, height);
        self.buddhabrot.rebind(device, &self.main, &self.palette_buffer);
        self.accum_sample_count = 0;

        log::info!("Compute pipeline resized to {}x{}", width, height);
    }
}

/// Buddhabrot: an accumulation pass adding random orbits to a per-pixel counter
/// buffer, and a tonemap pass turning the counters into colours.
struct BuddhabrotPasses {
    /// One u32 hit counter per pixel
    accum_buffer: wgpu::Buffer,
    accum_layout: wgpu::BindGroupLayout,
    accum_bind_group: wgpu::BindGroup,
    accum_pipeline_layout: wgpu::PipelineLayout,
    tonemap_layout: wgpu::BindGroupLayout,
    tonemap_bind_group: wgpu::BindGroup,
    tonemap_pipeline_layout: wgpu::PipelineLayout,
    /// (accumulation, tonemap), compiled on first use
    pipelines: Option<(wgpu::ComputePipeline, wgpu::ComputePipeline)>,
}

impl BuddhabrotPasses {
    fn new(device: &wgpu::Device, target: &FractalTarget, palette_buffer: &wgpu::Buffer) -> Self {
        // Accumulation: uniforms + counters (read_write)
        let accum_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Buddhabrot Accum Bind Group Layout"),
            entries: &[uniform_entry(0), storage_entry(1, false)],
        });
        // Tonemap: uniforms + output texture + counters (read) + palette
        let tonemap_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Buddhabrot Tonemap Bind Group Layout"),
            entries: &[uniform_entry(0), output_texture_entry(1), storage_entry(2, true), storage_entry(3, true)],
        });
        let (accum_buffer, accum_bind_group, tonemap_bind_group) =
            Self::bind(device, &accum_layout, &tonemap_layout, target, palette_buffer);

        Self {
            accum_pipeline_layout: create_pipeline_layout(device, "Buddhabrot Accum Pipeline Layout", &accum_layout),
            tonemap_pipeline_layout: create_pipeline_layout(device, "Buddhabrot Tonemap Pipeline Layout", &tonemap_layout),
            accum_buffer,
            accum_layout,
            accum_bind_group,
            tonemap_layout,
            tonemap_bind_group,
            pipelines: None,
        }
    }

    /// Create the counter buffer for `target`'s size and the bind groups around it
    fn bind(
        device: &wgpu::Device,
        accum_layout: &wgpu::BindGroupLayout,
        tonemap_layout: &wgpu::BindGroupLayout,
        target: &FractalTarget,
        palette_buffer: &wgpu::Buffer,
    ) -> (wgpu::Buffer, wgpu::BindGroup, wgpu::BindGroup) {
        let accum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buddhabrot Accumulation Buffer"),
            size: target.width() as u64 * target.height() as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let accum_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Buddhabrot Accum Bind Group"),
            layout: accum_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: target.uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: accum_buffer.as_entire_binding() },
            ],
        });
        let tonemap_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Buddhabrot Tonemap Bind Group"),
            layout: tonemap_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: target.uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&target.view) },
                wgpu::BindGroupEntry { binding: 2, resource: accum_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: palette_buffer.as_entire_binding() },
            ],
        });
        (accum_buffer, accum_bind_group, tonemap_bind_group)
    }

    /// Point the passes at a new (resized) target, starting from empty counters
    fn rebind(&mut self, device: &wgpu::Device, target: &FractalTarget, palette_buffer: &wgpu::Buffer) {
        (self.accum_buffer, self.accum_bind_group, self.tonemap_bind_group) =
            Self::bind(device, &self.accum_layout, &self.tonemap_layout, target, palette_buffer);
    }

    /// Record the accumulation pass and the tonemap pass
    fn dispatch(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, width: u32, height: u32) {
        let (accum_pipeline, tonemap_pipeline) = self.pipelines.get_or_insert_with(|| {
            log::info!("Creating Buddhabrot pipelines");
            (
                create_compute_pipeline(
                    device,
                    "Buddhabrot Accum",
                    &self.accum_pipeline_layout,
                    &shader::buddhabrot_accum_source(),
                ),
                create_compute_pipeline(
                    device,
                    "Buddhabrot Tonemap",
                    &self.tonemap_pipeline_layout,
                    &shader::buddhabrot_tonemap_source(),
                ),
            )
        });

        // Accumulation pass: 1D dispatch, each invocation traces one random orbit
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Buddhabrot Accum Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(accum_pipeline);
            pass.set_bind_group(0, &self.accum_bind_group, &[]);
            pass.dispatch_workgroups(BUDDHABROT_SAMPLES_PER_FRAME.div_ceil(BUDDHABROT_WORKGROUP_SIZE), 1, 1);
        }

        // Tonemap pass: 2D dispatch, one invocation per pixel
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Buddhabrot Tonemap Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(tonemap_pipeline);
            pass.set_bind_group(0, &self.tonemap_bind_group, &[]);
            pass.dispatch_workgroups(workgroups(width), workgroups(height), 1);
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use glam::DVec2;

    /// The live-view passes (main, linked Julia at perturbation depth, Buddhabrot)
    /// record and submit without GPU validation errors, before and after a resize.
    #[test]
    fn live_passes_are_valid() {
        let instance = wgpu::Instance::default();
        let Some(adapter) = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default()))
        else {
            eprintln!("no GPU adapter available; skipping");
            return;
        };
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default(), None)).unwrap();

        device.push_error_scope(wgpu::ErrorFilter::Validation);
        let mut compute = ComputePipeline::new(&device, 96, 64);
        compute.upload_palette(&queue, &[0x80; PALETTE_LUT_BYTES as usize]);

        for size in [(96, 64), (50, 70)] {
            compute.resize(&device, size.0, size.1);
            compute.clear_accum_buffer(&queue);
            for fractal in &FractalType::ALL {
                let view = View {
                    center: fractal.default_center(),
                    zoom: fractal.default_zoom(),
                    rotation: 0.0,
                    max_iter: 128,
                };
                let deep_julia = View { center: DVec2::new(0.1, 0.2), zoom: 1.0e6, ..view };

                let mut encoder = device.create_command_encoder(&Default::default());
                if fractal.is_buddhabrot() {
                    compute.render_buddhabrot(&device, &queue, &mut encoder, &view, 7);
                    assert_eq!(compute.accum_sample_count, 1);
                } else {
                    compute.render_main(&device, &queue, &mut encoder, fractal, &view);
                }
                compute.render_julia(&device, &queue, &mut encoder, Vec2::new(-0.7, 0.27015), &deep_julia);
                queue.submit(std::iter::once(encoder.finish()));
            }
        }
        device.poll(wgpu::Maintain::Wait);
        let error = pollster::block_on(device.pop_error_scope());
        assert!(error.is_none(), "{error:?}");
    }
}
