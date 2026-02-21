use super::uniforms::FractalUniforms;
use crate::fractals::FractalType;
use std::collections::HashMap;
use wgpu;

/// Zoom threshold for automatic f64 precision switching
const F64_ZOOM_THRESHOLD: f64 = 5.0e3;

/// Compute pipeline for fractal rendering with dynamic shader support
pub struct ComputePipeline {
    // Standard bind group (bindings 0,1: uniform + texture) for f32 and non-Mandelbrot f64
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniform_buffer: wgpu::Buffer,
    pub storage_texture: wgpu::Texture,
    pub texture_view: wgpu::TextureView,

    // Perturbation bind group (bindings 0,1,2: uniform + texture + orbit buffer) for Mandelbrot f64
    orbit_buffer: wgpu::Buffer,
    perturbation_bind_group_layout: wgpu::BindGroupLayout,
    perturbation_bind_group: wgpu::BindGroup,
    perturbation_pipeline_layout: wgpu::PipelineLayout,

    // Pipeline caching: key is (fractal_type_id, is_f64)
    pipeline_cache: HashMap<(u32, bool), wgpu::ComputePipeline>,
    pipeline_layout: wgpu::PipelineLayout,
    current_fractal_type: u32,
    /// Whether the current frame is using emulated f64 precision
    pub using_f64: bool,
}

/// Initial orbit buffer size (max_iter+1 entries * 2 floats * 4 bytes)
const ORBIT_BUFFER_SIZE: u64 = (4097 * 2 * 4) as u64;

impl ComputePipeline {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fractal Uniforms Buffer"),
            size: std::mem::size_of::<FractalUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create orbit buffer for perturbation theory
        let orbit_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Reference Orbit Buffer"),
            size: ORBIT_BUFFER_SIZE,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Create storage texture (compute shader output)
        let storage_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Compute Output Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        let texture_view = storage_texture.create_view(&wgpu::TextureViewDescriptor::default());

        // Standard bind group layout (2 bindings: uniform + texture)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
            ],
        });

        // Perturbation bind group layout (3 bindings: uniform + texture + orbit)
        let perturbation_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Perturbation Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        // Standard bind group
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
            ],
        });

        // Perturbation bind group
        let perturbation_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Perturbation Bind Group"),
            layout: &perturbation_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: orbit_buffer.as_entire_binding(),
                },
            ],
        });

        // Pipeline layouts
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let perturbation_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Perturbation Pipeline Layout"),
            bind_group_layouts: &[&perturbation_bind_group_layout],
            push_constant_ranges: &[],
        });

        log::info!("Compute pipeline created: {}x{}", width, height);

        Self {
            bind_group,
            bind_group_layout,
            uniform_buffer,
            storage_texture,
            texture_view,
            orbit_buffer,
            perturbation_bind_group_layout,
            perturbation_bind_group,
            perturbation_pipeline_layout,
            pipeline_cache: HashMap::new(),
            pipeline_layout,
            current_fractal_type: 0,
            using_f64: false,
        }
    }

    /// Upload reference orbit data to GPU
    pub fn upload_orbit(&self, _device: &wgpu::Device, queue: &wgpu::Queue, orbit_data: &[f32]) {
        let bytes = bytemuck::cast_slice(orbit_data);
        queue.write_buffer(&self.orbit_buffer, 0, bytes);
    }

    /// Check if the given fractal+precision uses perturbation (needs 3-binding layout)
    fn uses_perturbation(fractal_type: &FractalType, use_f64: bool) -> bool {
        use_f64 && (fractal_type.type_id() == 0 || fractal_type.type_id() == 1) // Mandelbrot & Julia
    }

    /// Get or create a compute pipeline for the given fractal type and precision
    fn get_or_create_pipeline(
        &mut self,
        device: &wgpu::Device,
        fractal_type: &FractalType,
        use_f64: bool,
    ) -> &wgpu::ComputePipeline {
        let type_id = fractal_type.type_id();
        let cache_key = (type_id, use_f64);

        // Select the correct pipeline layout based on whether perturbation is used
        let layout = if Self::uses_perturbation(fractal_type, use_f64) {
            &self.perturbation_pipeline_layout
        } else {
            &self.pipeline_layout
        };

        self.pipeline_cache.entry(cache_key).or_insert_with(|| {
            let shader_source = if use_f64 {
                fractal_type.shader_source_f64()
            } else {
                fractal_type.shader_source()
            };
            let precision_label = if use_f64 { "f64" } else { "f32" };
            let label = format!("{} Compute Shader ({})", fractal_type.name(), precision_label);

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&label),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{} Compute Pipeline ({})", fractal_type.name(), precision_label)),
                layout: Some(layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            });

            log::info!("Created compute pipeline for: {} ({})", fractal_type.name(), precision_label);
            pipeline
        })
    }

    /// Update uniform buffer
    pub fn update_uniforms(&self, queue: &wgpu::Queue, uniforms: &FractalUniforms) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Dispatch compute shader with dynamic fractal type and automatic precision switching
    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        width: u32,
        height: u32,
        fractal_type: &FractalType,
        zoom: f64,
    ) {
        let type_id = fractal_type.type_id();
        let use_f64 = zoom >= F64_ZOOM_THRESHOLD;

        // Track state changes for logging
        if self.current_fractal_type != type_id {
            self.current_fractal_type = type_id;
            log::debug!("Switched to fractal type: {}", fractal_type.name());
        }
        if self.using_f64 != use_f64 {
            self.using_f64 = use_f64;
            let mode = if use_f64 { "f64 (perturbation)" } else { "f32" };
            log::info!("Precision mode switched to: {}", mode);
        }

        // Get or create pipeline for this fractal type and precision
        let pipeline = self.get_or_create_pipeline(device, fractal_type, use_f64);

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Fractal Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(pipeline);

        // Use perturbation bind group (with orbit buffer) for Mandelbrot f64
        if Self::uses_perturbation(fractal_type, use_f64) {
            compute_pass.set_bind_group(0, &self.perturbation_bind_group, &[]);
        } else {
            compute_pass.set_bind_group(0, &self.bind_group, &[]);
        }

        // Calculate dispatch size (16x16 workgroups)
        let workgroup_count_x = (width + 15) / 16;
        let workgroup_count_y = (height + 15) / 16;

        compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
    }

    /// Recreate textures on resize
    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        // Recreate storage texture
        self.storage_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Compute Output Texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });

        self.texture_view = self
            .storage_texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // Recreate standard bind group
        self.bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute Bind Group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.texture_view),
                },
            ],
        });

        // Recreate perturbation bind group
        self.perturbation_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Perturbation Bind Group"),
            layout: &self.perturbation_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.orbit_buffer.as_entire_binding(),
                },
            ],
        });

        log::info!("Compute pipeline resized to {}x{}", width, height);
    }
}
