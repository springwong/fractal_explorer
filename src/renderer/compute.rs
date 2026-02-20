use super::uniforms::FractalUniforms;
use crate::fractals::FractalType;
use std::collections::HashMap;
use wgpu;

/// Compute pipeline for fractal rendering with dynamic shader support
pub struct ComputePipeline {
    pub bind_group: wgpu::BindGroup,
    pub bind_group_layout: wgpu::BindGroupLayout,
    pub uniform_buffer: wgpu::Buffer,
    pub storage_texture: wgpu::Texture,
    pub texture_view: wgpu::TextureView,

    // Pipeline caching
    pipeline_cache: HashMap<u32, wgpu::ComputePipeline>,
    pipeline_layout: wgpu::PipelineLayout,
    current_fractal_type: u32,
}

impl ComputePipeline {
    pub fn new(device: &wgpu::Device, width: u32, height: u32) -> Self {
        // Create uniform buffer
        let uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Fractal Uniforms Buffer"),
            size: std::mem::size_of::<FractalUniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
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

        // Create bind group layout
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Compute Bind Group Layout"),
            entries: &[
                // Uniforms
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
                // Storage texture
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

        // Create bind group
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

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        log::info!("Compute pipeline created: {}x{}", width, height);

        Self {
            bind_group,
            bind_group_layout,
            uniform_buffer,
            storage_texture,
            texture_view,
            pipeline_cache: HashMap::new(),
            pipeline_layout,
            current_fractal_type: 0, // Default to Mandelbrot
        }
    }

    /// Get or create a compute pipeline for the given fractal type
    fn get_or_create_pipeline(
        &mut self,
        device: &wgpu::Device,
        fractal_type: &FractalType,
    ) -> &wgpu::ComputePipeline {
        let type_id = fractal_type.type_id();

        self.pipeline_cache.entry(type_id).or_insert_with(|| {
            let shader_source = fractal_type.shader_source();
            let label = format!("{} Compute Shader", fractal_type.name());

            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&label),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

            let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{} Compute Pipeline", fractal_type.name())),
                layout: Some(&self.pipeline_layout),
                module: &shader,
                entry_point: "main",
                compilation_options: Default::default(),
                cache: None,
            });

            log::info!("Created compute pipeline for: {}", fractal_type.name());
            pipeline
        })
    }

    /// Update uniform buffer
    pub fn update_uniforms(&self, queue: &wgpu::Queue, uniforms: &FractalUniforms) {
        queue.write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(uniforms));
    }

    /// Dispatch compute shader with dynamic fractal type
    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        width: u32,
        height: u32,
        fractal_type: &FractalType,
    ) {
        let type_id = fractal_type.type_id();

        // Track current type for debugging (before borrowing)
        if self.current_fractal_type != type_id {
            self.current_fractal_type = type_id;
            log::debug!("Switched to fractal type: {}", fractal_type.name());
        }

        // Get or create pipeline for this fractal type
        let pipeline = self.get_or_create_pipeline(device, fractal_type);

        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Fractal Compute Pass"),
            timestamp_writes: None,
        });

        compute_pass.set_pipeline(pipeline);
        compute_pass.set_bind_group(0, &self.bind_group, &[]);

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

        // Recreate bind group
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

        log::info!("Compute pipeline resized to {}x{}", width, height);
    }
}
