//! The per-pixel fractal compute pass, shared by the live view, the linked Julia
//! view, PNG export and video recording.
//!
//! `FractalPipelines` owns the bind group layouts and the compiled pipelines;
//! a `FractalTarget` is one output texture with the buffers a pass into it needs.

use super::uniforms::{FractalUniforms, View};
use crate::fractals::{is_deep_zoom, FractalType, MAX_ITERATIONS};
use std::collections::HashMap;

/// Size of the palette LUT buffer: 256 packed RGBA8 entries
pub const PALETTE_LUT_BYTES: u64 = 1024;

/// Compute shaders run in 16 x 16 pixel workgroups
const WORKGROUP_SIZE: u32 = 16;

/// Reference orbit buffer: (max_iter + 1) [zx, zy] f32 pairs
const ORBIT_BUFFER_BYTES: u64 = (MAX_ITERATIONS as u64 + 1) * 2 * 4;

/// Number of workgroups needed to cover `pixels` along one axis
pub fn workgroups(pixels: u32) -> u32 {
    pixels.div_ceil(WORKGROUP_SIZE)
}

/// Layout entry for the uniform block
pub fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    buffer_entry(binding, wgpu::BufferBindingType::Uniform)
}

/// Layout entry for a storage buffer
pub fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    buffer_entry(binding, wgpu::BufferBindingType::Storage { read_only })
}

fn buffer_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer { ty, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

/// Layout entry for the RGBA8 output texture
pub fn output_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::StorageTexture {
            access: wgpu::StorageTextureAccess::WriteOnly,
            format: wgpu::TextureFormat::Rgba8Unorm,
            view_dimension: wgpu::TextureViewDimension::D2,
        },
        count: None,
    }
}

/// Create a compute pipeline (entry point `main`) from WGSL source
pub fn create_compute_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    source: &str,
) -> wgpu::ComputePipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        module: &module,
        entry_point: "main",
        compilation_options: Default::default(),
        cache: None,
    })
}

/// Create a pipeline layout around a single bind group layout
pub fn create_pipeline_layout(
    device: &wgpu::Device,
    label: &str,
    bind_group_layout: &wgpu::BindGroupLayout,
) -> wgpu::PipelineLayout {
    device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[bind_group_layout],
        push_constant_ranges: &[],
    })
}

/// Create the palette LUT buffer
pub fn create_palette_buffer(device: &wgpu::Device) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Palette LUT Buffer"),
        size: PALETTE_LUT_BYTES,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

/// Bind group layouts and compiled pipelines for the per-pixel fractal shaders
pub struct FractalPipelines {
    /// uniforms + output texture + palette
    standard_layout: wgpu::BindGroupLayout,
    standard_pipeline_layout: wgpu::PipelineLayout,
    /// uniforms + output texture + reference orbit + palette
    orbit_layout: wgpu::BindGroupLayout,
    orbit_pipeline_layout: wgpu::PipelineLayout,
    /// Compiled pipelines, keyed by (fractal id, deep zoom)
    cache: HashMap<(u32, bool), wgpu::ComputePipeline>,
}

impl FractalPipelines {
    /// Create the layouts; pipelines are compiled on first use
    pub fn new(device: &wgpu::Device) -> Self {
        let standard_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fractal Bind Group Layout"),
            entries: &[uniform_entry(0), output_texture_entry(1), storage_entry(2, true)],
        });
        let orbit_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Fractal Perturbation Bind Group Layout"),
            entries: &[uniform_entry(0), output_texture_entry(1), storage_entry(2, true), storage_entry(3, true)],
        });
        Self {
            standard_pipeline_layout: create_pipeline_layout(device, "Fractal Pipeline Layout", &standard_layout),
            orbit_pipeline_layout: create_pipeline_layout(device, "Fractal Perturbation Pipeline Layout", &orbit_layout),
            standard_layout,
            orbit_layout,
            cache: HashMap::new(),
        }
    }

    /// The pipeline for `fractal` at `zoom` and whether it reads a reference orbit.
    /// Errors for fractals that are not rendered per pixel.
    fn get(
        &mut self,
        device: &wgpu::Device,
        fractal: &FractalType,
        zoom: f64,
    ) -> Result<(&wgpu::ComputePipeline, bool), String> {
        let shader = fractal
            .shader(zoom)
            .ok_or_else(|| format!("{} is not rendered by a per-pixel shader", fractal.name()))?;
        let deep = is_deep_zoom(zoom);
        let layout = if shader.uses_orbit() { &self.orbit_pipeline_layout } else { &self.standard_pipeline_layout };

        let pipeline = self.cache.entry((fractal.id(), deep)).or_insert_with(|| {
            let label = format!("{} Compute ({})", fractal.name(), if deep { "deep" } else { "f32" });
            log::info!("Creating pipeline: {label}");
            create_compute_pipeline(device, &label, layout, &shader.source())
        });
        Ok((pipeline, shader.uses_orbit()))
    }
}

/// One output texture plus everything a fractal pass into it needs: its own uniform
/// buffer, its own reference orbit buffer and the bind groups tying them together.
pub struct FractalTarget {
    /// The RGBA8 texture the pass writes
    pub texture: wgpu::Texture,
    /// Default view of `texture`
    pub view: wgpu::TextureView,
    /// Uniform block for passes into this target
    pub uniform_buffer: wgpu::Buffer,
    orbit_buffer: wgpu::Buffer,
    standard_bind_group: wgpu::BindGroup,
    orbit_bind_group: wgpu::BindGroup,
    width: u32,
    height: u32,
}

impl FractalTarget {
    /// Create a `width` x `height` target. `extra_usage` is what the texture is used
    /// for besides being written by the compute pass (sampling, or copying out).
    pub fn new(
        device: &wgpu::Device,
        pipelines: &FractalPipelines,
        palette_buffer: &wgpu::Buffer,
        label: &str,
        width: u32,
        height: u32,
        extra_usage: wgpu::TextureUsages,
    ) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(&format!("{label} Texture")),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | extra_usage,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label} Uniforms")),
            size: std::mem::size_of::<FractalUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let orbit_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("{label} Reference Orbit")),
            size: ORBIT_BUFFER_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let standard_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label} Bind Group")),
            layout: &pipelines.standard_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 2, resource: palette_buffer.as_entire_binding() },
            ],
        });
        let orbit_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&format!("{label} Perturbation Bind Group")),
            layout: &pipelines.orbit_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniform_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 2, resource: orbit_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: palette_buffer.as_entire_binding() },
            ],
        });

        Self { texture, view, uniform_buffer, orbit_buffer, standard_bind_group, orbit_bind_group, width, height }
    }

    /// Target width in pixels
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Target height in pixels
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Render `fractal` at `view` into this target: upload the uniforms (and the
    /// reference orbit when the shader wants one) and record the compute pass.
    ///
    /// Returns whether the deep-zoom shader was used. The uploads are queued writes,
    /// so submit `encoder` before rendering into this target again.
    pub fn render(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        pipelines: &mut FractalPipelines,
        fractal: &FractalType,
        view: &View,
    ) -> Result<bool, String> {
        if view.max_iter > MAX_ITERATIONS {
            return Err(format!("max_iter {} exceeds the supported {MAX_ITERATIONS}", view.max_iter));
        }
        let (pipeline, uses_orbit) = pipelines.get(device, fractal, view.zoom)?;

        let mut uniforms = FractalUniforms::for_view(fractal, view, self.width, self.height);
        if uses_orbit {
            let (orbit, escape_iter) = fractal
                .reference_orbit(view.center, view.max_iter)
                .ok_or_else(|| format!("{} has no reference orbit", fractal.name()))?;
            queue.write_buffer(&self.orbit_buffer, 0, bytemuck::cast_slice(&orbit));
            uniforms.ref_escape_iter = escape_iter;
        }
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&uniforms));

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Fractal Compute Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, if uses_orbit { &self.orbit_bind_group } else { &self.standard_bind_group }, &[]);
        pass.dispatch_workgroups(workgroups(self.width), workgroups(self.height), 1);

        Ok(is_deep_zoom(view.zoom))
    }
}
