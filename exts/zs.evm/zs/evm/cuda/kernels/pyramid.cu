/*
 * CUDA Kernels for Gaussian/Laplacian Pyramid Construction
 *
 * Target: <10ms latency for full EVM pipeline
 * Implements efficient pyramid construction with shared memory optimization
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// 5x5 Gaussian kernel coefficients (normalized)
__constant__ float d_gaussian_kernel[25] = {
    0.003765, 0.015019, 0.023792, 0.015019, 0.003765,
    0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
    0.023792, 0.094907, 0.150342, 0.094907, 0.023792,
    0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
    0.003765, 0.015019, 0.023792, 0.015019, 0.003765
};

// Tile size for shared memory
#define TILE_WIDTH 16
#define KERNEL_RADIUS 2
#define SHARED_WIDTH (TILE_WIDTH + 2 * KERNEL_RADIUS)

/**
 * Gaussian blur kernel with shared memory optimization
 * Processes a single channel of the image
 */
__global__ void gaussian_blur_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int width,
    int height,
    int pitch
) {
    __shared__ float tile[SHARED_WIDTH][SHARED_WIDTH];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;

    // Global position
    int x = bx * TILE_WIDTH + tx;
    int y = by * TILE_WIDTH + ty;

    // Load tile into shared memory with halo
    for (int dy = 0; dy < SHARED_WIDTH; dy += blockDim.y) {
        for (int dx = 0; dx < SHARED_WIDTH; dx += blockDim.x) {
            int load_x = bx * TILE_WIDTH + dx - KERNEL_RADIUS;
            int load_y = by * TILE_WIDTH + dy + ty - KERNEL_RADIUS;

            // Clamp to valid range
            load_x = max(0, min(width - 1, load_x));
            load_y = max(0, min(height - 1, load_y));

            if (dx + tx < SHARED_WIDTH && dy < SHARED_WIDTH) {
                tile[dy][dx + tx] = input[load_y * pitch + load_x];
            }
        }
    }

    __syncthreads();

    // Compute convolution
    if (x < width && y < height) {
        float sum = 0.0f;

        #pragma unroll
        for (int ky = 0; ky < 5; ky++) {
            #pragma unroll
            for (int kx = 0; kx < 5; kx++) {
                int tile_x = tx + kx;
                int tile_y = ty + ky;
                sum += tile[tile_y][tile_x] * d_gaussian_kernel[ky * 5 + kx];
            }
        }

        output[y * pitch + x] = sum;
    }
}

/**
 * Downsample image by 2x (take every other pixel)
 */
__global__ void downsample_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_width,
    int in_height,
    int out_width,
    int out_height,
    int in_pitch,
    int out_pitch
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < out_width && y < out_height) {
        // Sample from 2x position in input
        int in_x = x * 2;
        int in_y = y * 2;

        // Clamp to valid range
        in_x = min(in_x, in_width - 1);
        in_y = min(in_y, in_height - 1);

        output[y * out_pitch + x] = input[in_y * in_pitch + in_x];
    }
}

/**
 * Upsample image by 2x (duplicate pixels) with Gaussian interpolation
 */
__global__ void upsample_kernel(
    const float* __restrict__ input,
    float* __restrict__ output,
    int in_width,
    int in_height,
    int out_width,
    int out_height,
    int in_pitch,
    int out_pitch
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < out_width && y < out_height) {
        // Source coordinates in input (half resolution)
        float src_x = (float)x / 2.0f;
        float src_y = (float)y / 2.0f;

        int x0 = (int)floorf(src_x);
        int y0 = (int)floorf(src_y);
        int x1 = min(x0 + 1, in_width - 1);
        int y1 = min(y0 + 1, in_height - 1);

        x0 = max(0, min(x0, in_width - 1));
        y0 = max(0, min(y0, in_height - 1));

        // Bilinear interpolation
        float fx = src_x - x0;
        float fy = src_y - y0;

        float v00 = input[y0 * in_pitch + x0];
        float v01 = input[y0 * in_pitch + x1];
        float v10 = input[y1 * in_pitch + x0];
        float v11 = input[y1 * in_pitch + x1];

        float v0 = v00 * (1.0f - fx) + v01 * fx;
        float v1 = v10 * (1.0f - fx) + v11 * fx;
        float v = v0 * (1.0f - fy) + v1 * fy;

        output[y * out_pitch + x] = v * 4.0f; // Scale by 4 for energy conservation
    }
}

/**
 * Compute Laplacian level: L[i] = G[i] - upsample(G[i+1])
 */
__global__ void laplacian_level_kernel(
    const float* __restrict__ current_level,
    const float* __restrict__ upsampled_next,
    float* __restrict__ laplacian,
    int width,
    int height,
    int pitch
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * pitch + x;
        laplacian[idx] = current_level[idx] - upsampled_next[idx];
    }
}

/**
 * Collapse Laplacian pyramid: reconstruct from bottom up
 * out[i] = L[i] + upsample(out[i+1])
 */
__global__ void collapse_add_kernel(
    const float* __restrict__ laplacian_level,
    const float* __restrict__ upsampled_prev,
    float* __restrict__ output,
    int width,
    int height,
    int pitch
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * pitch + x;
        output[idx] = laplacian_level[idx] + upsampled_prev[idx];
    }
}

/**
 * Amplify spatial frequencies with wavelength attenuation
 * amplified = filtered * alpha * attenuation(level)
 */
__global__ void amplify_kernel(
    const float* __restrict__ filtered,
    float* __restrict__ amplified,
    float alpha,
    float attenuation_factor,
    int width,
    int height,
    int pitch
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * pitch + x;
        amplified[idx] = filtered[idx] * alpha * attenuation_factor;
    }
}

/**
 * Multi-channel processing: process RGB frame
 */
__global__ void process_rgb_kernel(
    const float* __restrict__ input_r,
    const float* __restrict__ input_g,
    const float* __restrict__ input_b,
    float* __restrict__ output_r,
    float* __restrict__ output_g,
    float* __restrict__ output_b,
    int width,
    int height,
    int pitch
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * pitch + x;
        output_r[idx] = input_r[idx];
        output_g[idx] = input_g[idx];
        output_b[idx] = input_b[idx];
    }
}

// Host interface functions (extern "C" for Python ctypes)
extern "C" {

/**
 * Build Gaussian pyramid level by level
 */
void cuda_build_gaussian_pyramid(
    const float* d_input,
    float** d_pyramid_levels,
    int* widths,
    int* heights,
    int* pitches,
    int num_levels,
    cudaStream_t stream
) {
    dim3 block(16, 16);

    for (int level = 0; level < num_levels; level++) {
        if (level == 0) {
            // Copy input to first level
            cudaMemcpy2DAsync(
                d_pyramid_levels[0],
                pitches[0] * sizeof(float),
                d_input,
                widths[0] * sizeof(float),
                widths[0] * sizeof(float),
                heights[0],
                cudaMemcpyDeviceToDevice,
                stream
            );
        } else {
            // Blur previous level
            dim3 grid_blur(
                (widths[level - 1] + block.x - 1) / block.x,
                (heights[level - 1] + block.y - 1) / block.y
            );

            gaussian_blur_kernel<<<grid_blur, block, 0, stream>>>(
                d_pyramid_levels[level - 1],
                d_pyramid_levels[level - 1], // In-place for temp
                widths[level - 1],
                heights[level - 1],
                pitches[level - 1]
            );

            // Downsample
            dim3 grid_down(
                (widths[level] + block.x - 1) / block.x,
                (heights[level] + block.y - 1) / block.y
            );

            downsample_kernel<<<grid_down, block, 0, stream>>>(
                d_pyramid_levels[level - 1],
                d_pyramid_levels[level],
                widths[level - 1],
                heights[level - 1],
                widths[level],
                heights[level],
                pitches[level - 1],
                pitches[level]
            );
        }
    }
}

} // extern "C"
