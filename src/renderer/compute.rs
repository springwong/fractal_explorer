use super::uniforms::FractalUniforms;
use crate::fractals::FractalType;
use std::collections::HashMap;
use wgpu;

/// Zoom threshold for automatic f64 precision switching
const F64_ZOOM_THRESHOLD: f64 = 5.0e3;

/// Compute pipeline for fractal rendering with dynamic shader support
pub struct ComputePipeline {
    // Standard bind group (bindings 0,1,2: uniform + texture + palette) for f32 and non-Mandelbrot f64
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniform_buffer: wgpu::Buffer,
    pub storage_texture: wgpu::Texture,
    pub texture_view: wgpu::TextureView,

    // Palette LUT buffer (256 packed RGBA8 entries = 1024 bytes)
    palette_buffer: wgpu::Buffer,

    // Perturbation bind group (bindings 0,1,2,3: uniform + texture + orbit + palette) for Mandelbrot f64
    orbit_buffer: wgpu::Buffer,
    perturbation_bind_group_layout: wgpu::BindGroupLayout,
    perturbation_bind_group: wgpu::BindGroup,
    perturbation_pipeline_layout: wgpu::PipelineLayout,

    // Buddhabrot accumulation buffer and pipelines
    accum_buffer: wgpu::Buffer,
    buddhabrot_accum_bind_group_layout: wgpu::BindGroupLayout,
    buddhabrot_accum_bind_group: wgpu::BindGroup,
    buddhabrot_accum_pipeline_layout: wgpu::PipelineLayout,
    buddhabrot_accum_pipeline: Option<wgpu::ComputePipeline>,
    buddhabrot_tonemap_bind_group_layout: wgpu::BindGroupLayout,
    buddhabrot_tonemap_bind_group: wgpu::BindGroup,
    buddhabrot_tonemap_pipeline_layout: wgpu::PipelineLayout,
    buddhabrot_tonemap_pipeline: Option<wgpu::ComputePipeline>,
    accum_width: u32,
    accum_height: u32,
    /// Total accumulated sample batches for Buddhabrot
    pub accum_sample_count: u32,

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

        // Create palette LUT buffer (256 RGBA8 entries = 1024 bytes)
        let palette_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Palette LUT Buffer"),
            size: 1024,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
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

        // Standard bind group layout (3 bindings: uniform + texture + palette)
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

        // Perturbation bind group layout (4 bindings: uniform + texture + orbit + palette)
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: palette_buffer.as_entire_binding(),
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: palette_buffer.as_entire_binding(),
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

        // --- Buddhabrot accumulation buffer and bind groups ---
        let accum_buffer_size = (width * height * 4) as u64; // one u32 per pixel
        let accum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buddhabrot Accumulation Buffer"),
            size: accum_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Accumulation pass bind group layout: uniform + accum_buf (read_write)
        let buddhabrot_accum_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Buddhabrot Accum Bind Group Layout"),
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
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let buddhabrot_accum_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Buddhabrot Accum Bind Group"),
            layout: &buddhabrot_accum_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: accum_buffer.as_entire_binding(),
                },
            ],
        });

        let buddhabrot_accum_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Buddhabrot Accum Pipeline Layout"),
            bind_group_layouts: &[&buddhabrot_accum_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Tonemap pass bind group layout: uniform + output_texture + accum_buf (read) + palette
        let buddhabrot_tonemap_bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Buddhabrot Tonemap Bind Group Layout"),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
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

        let buddhabrot_tonemap_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Buddhabrot Tonemap Bind Group"),
            layout: &buddhabrot_tonemap_bind_group_layout,
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
                    resource: accum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: palette_buffer.as_entire_binding(),
                },
            ],
        });

        let buddhabrot_tonemap_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Buddhabrot Tonemap Pipeline Layout"),
            bind_group_layouts: &[&buddhabrot_tonemap_bind_group_layout],
            push_constant_ranges: &[],
        });

        log::info!("Compute pipeline created: {}x{}", width, height);

        Self {
            bind_group,
            bind_group_layout,
            uniform_buffer,
            storage_texture,
            texture_view,
            palette_buffer,
            orbit_buffer,
            perturbation_bind_group_layout,
            perturbation_bind_group,
            perturbation_pipeline_layout,
            accum_buffer,
            buddhabrot_accum_bind_group_layout,
            buddhabrot_accum_bind_group,
            buddhabrot_accum_pipeline_layout,
            buddhabrot_accum_pipeline: None,
            buddhabrot_tonemap_bind_group_layout,
            buddhabrot_tonemap_bind_group,
            buddhabrot_tonemap_pipeline_layout,
            buddhabrot_tonemap_pipeline: None,
            accum_width: width,
            accum_height: height,
            accum_sample_count: 0,
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

    /// Upload palette LUT data to GPU
    pub fn upload_palette(&self, queue: &wgpu::Queue, data: &[u8; 1024]) {
        queue.write_buffer(&self.palette_buffer, 0, data);
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

    /// Clear the Buddhabrot accumulation buffer (zero-fill)
    pub fn clear_accum_buffer(&mut self, queue: &wgpu::Queue) {
        let size = (self.accum_width * self.accum_height * 4) as usize;
        let zeros = vec![0u8; size];
        queue.write_buffer(&self.accum_buffer, 0, &zeros);
        self.accum_sample_count = 0;
        log::info!("Buddhabrot accumulation buffer cleared");
    }

    /// Dispatch Buddhabrot accumulation + tonemap passes
    pub fn dispatch_buddhabrot(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        width: u32,
        height: u32,
    ) {
        // Lazily create accumulation pipeline
        if self.buddhabrot_accum_pipeline.is_none() {
            let shader_source = include_str!("../shaders/buddhabrot.wgsl");
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Buddhabrot Accum Shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });
            self.buddhabrot_accum_pipeline = Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Buddhabrot Accum Pipeline"),
                layout: Some(&self.buddhabrot_accum_pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            }));
            log::info!("Created Buddhabrot accumulation pipeline");
        }

        // Lazily create tonemap pipeline
        if self.buddhabrot_tonemap_pipeline.is_none() {
            let shader_source = include_str!("../shaders/buddhabrot_tonemap.wgsl");
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Buddhabrot Tonemap Shader"),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });
            self.buddhabrot_tonemap_pipeline = Some(device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Buddhabrot Tonemap Pipeline"),
                layout: Some(&self.buddhabrot_tonemap_pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            }));
            log::info!("Created Buddhabrot tonemap pipeline");
        }

        // Accumulation pass: dispatch 1D, each thread = one random sample
        // Use 65536 samples per frame for progressive refinement
        let num_samples: u32 = 65536;
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Buddhabrot Accum Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(self.buddhabrot_accum_pipeline.as_ref().unwrap());
            compute_pass.set_bind_group(0, &self.buddhabrot_accum_bind_group, &[]);
            let workgroup_count = (num_samples + 255) / 256;
            compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        }

        // Tonemap pass: 2D dispatch, one thread per pixel
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Buddhabrot Tonemap Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(self.buddhabrot_tonemap_pipeline.as_ref().unwrap());
            compute_pass.set_bind_group(0, &self.buddhabrot_tonemap_bind_group, &[]);
            let workgroup_count_x = (width + 15) / 16;
            let workgroup_count_y = (height + 15) / 16;
            compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
        }

        self.accum_sample_count += 1;
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.palette_buffer.as_entire_binding(),
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.palette_buffer.as_entire_binding(),
                },
            ],
        });

        // Recreate Buddhabrot accumulation buffer and bind groups
        let accum_buffer_size = (width * height * 4) as u64;
        self.accum_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Buddhabrot Accumulation Buffer"),
            size: accum_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.accum_width = width;
        self.accum_height = height;
        self.accum_sample_count = 0;

        self.buddhabrot_accum_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Buddhabrot Accum Bind Group"),
            layout: &self.buddhabrot_accum_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.accum_buffer.as_entire_binding(),
                },
            ],
        });

        self.buddhabrot_tonemap_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Buddhabrot Tonemap Bind Group"),
            layout: &self.buddhabrot_tonemap_bind_group_layout,
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
                    resource: self.accum_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.palette_buffer.as_entire_binding(),
                },
            ],
        });

        log::info!("Compute pipeline resized to {}x{}", width, height);
    }
}
