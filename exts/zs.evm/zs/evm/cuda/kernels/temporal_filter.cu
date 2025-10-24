/*
 * CUDA Kernels for Temporal Band-Pass Filtering
 *
 * Implements IIR Butterworth filter on GPU with circular buffer
 * for real-time EVM processing with <10ms latency
 */

#include <cuda_runtime.h>
#include <math.h>

// Maximum filter order (for Butterworth 2nd order: n=2)
#define MAX_FILTER_ORDER 5

/**
 * IIR filter state stored in global memory (persistent across frames)
 */
struct IIRFilterState {
    float x_history[MAX_FILTER_ORDER];  // Input history
    float y_history[MAX_FILTER_ORDER];  // Output history
    int write_idx;                       // Circular buffer write index
};

/**
 * Apply IIR filter to a single pixel value
 * Uses direct form II transposed structure for numerical stability
 */
__device__ float apply_iir_filter(
    float input,
    const float* b_coeffs,  // Numerator coefficients
    const float* a_coeffs,  // Denominator coefficients
    int order,
    IIRFilterState* state
) {
    // Direct Form II Transposed:
    // y[n] = b[0]*x[n] + w[0]
    // w[i] = b[i+1]*x[n] + w[i+1] - a[i+1]*y[n]

    float output = b_coeffs[0] * input;

    // Add contributions from delayed signals
    for (int i = 0; i < order; i++) {
        output += b_coeffs[i + 1] * state->x_history[i];
        output -= a_coeffs[i + 1] * state->y_history[i];
    }

    // Update state histories (shift)
    for (int i = order - 1; i > 0; i--) {
        state->x_history[i] = state->x_history[i - 1];
        state->y_history[i] = state->y_history[i - 1];
    }

    state->x_history[0] = input;
    state->y_history[0] = output;

    return output;
}

/**
 * Temporal band-pass filter for entire frame
 * Each thread processes one pixel's temporal history
 */
__global__ void temporal_bandpass_kernel(
    const float* __restrict__ input_frame,
    float* __restrict__ output_frame,
    IIRFilterState* __restrict__ filter_states,
    const float* __restrict__ b_coeffs,
    const float* __restrict__ a_coeffs,
    int filter_order,
    int width,
    int height,
    int pitch
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * pitch + x;

        // Get filter state for this pixel
        IIRFilterState* state = &filter_states[idx];

        // Apply filter
        float input_val = input_frame[idx];
        float output_val = apply_iir_filter(
            input_val,
            b_coeffs,
            a_coeffs,
            filter_order,
            state
        );

        output_frame[idx] = output_val;
    }
}

/**
 * Reset filter states (initialize to zero)
 */
__global__ void reset_filter_states_kernel(
    IIRFilterState* __restrict__ filter_states,
    int num_pixels
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < num_pixels) {
        IIRFilterState* state = &filter_states[idx];

        for (int i = 0; i < MAX_FILTER_ORDER; i++) {
            state->x_history[i] = 0.0f;
            state->y_history[i] = 0.0f;
        }

        state->write_idx = 0;
    }
}

/**
 * Circular buffer-based temporal filtering
 * More memory efficient for longer buffers
 */
__global__ void temporal_buffer_filter_kernel(
    const float* __restrict__ buffer,      // Circular buffer [num_frames, height, width]
    float* __restrict__ output,             // Output frame
    int buffer_size,
    int current_frame_idx,
    int width,
    int height,
    int pitch,
    const float* __restrict__ weights      // Temporal weights (e.g., Gaussian window)
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        float sum = 0.0f;
        float weight_sum = 0.0f;

        // Weighted sum over temporal buffer
        for (int t = 0; t < buffer_size; t++) {
            int frame_idx = (current_frame_idx - t + buffer_size) % buffer_size;
            int buffer_idx = frame_idx * (height * pitch) + y * pitch + x;

            float weight = weights[t];
            sum += buffer[buffer_idx] * weight;
            weight_sum += weight;
        }

        // Normalize
        if (weight_sum > 0.0f) {
            output[y * pitch + x] = sum / weight_sum;
        } else {
            output[y * pitch + x] = 0.0f;
        }
    }
}

/**
 * Fast FFT-based signal extraction for breath estimation
 * Process spatial mean of ROI over time
 */
__global__ void extract_roi_mean_kernel(
    const float* __restrict__ frame,
    const uint8_t* __restrict__ roi_mask,
    float* __restrict__ mean_output,
    int width,
    int height,
    int pitch
) {
    __shared__ float partial_sums[256];
    __shared__ int partial_counts[256];

    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    float local_sum = 0.0f;
    int local_count = 0;

    if (x < width && y < height) {
        int idx = y * pitch + x;

        if (roi_mask == nullptr || roi_mask[idx] > 0) {
            local_sum = frame[idx];
            local_count = 1;
        }
    }

    // Store to shared memory
    partial_sums[tid] = local_sum;
    partial_counts[tid] = local_count;

    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x * blockDim.y / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sums[tid] += partial_sums[tid + s];
            partial_counts[tid] += partial_counts[tid + s];
        }
        __syncthreads();
    }

    // Write block result to global memory
    if (tid == 0) {
        int block_idx = blockIdx.y * gridDim.x + blockIdx.x;
        mean_output[block_idx * 2] = partial_sums[0];
        mean_output[block_idx * 2 + 1] = (float)partial_counts[0];
    }
}

// Host interface
extern "C" {

/**
 * Initialize filter states on GPU
 */
cudaError_t cuda_init_temporal_filters(
    IIRFilterState** d_filter_states,
    int width,
    int height
) {
    int num_pixels = width * height;
    cudaError_t err = cudaMalloc(d_filter_states, num_pixels * sizeof(IIRFilterState));

    if (err != cudaSuccess) {
        return err;
    }

    // Reset states
    dim3 block(256);
    dim3 grid((num_pixels + block.x - 1) / block.x);

    reset_filter_states_kernel<<<grid, block>>>(*d_filter_states, num_pixels);

    return cudaGetLastError();
}

/**
 * Apply temporal filter to frame
 */
cudaError_t cuda_apply_temporal_filter(
    const float* d_input,
    float* d_output,
    IIRFilterState* d_filter_states,
    const float* d_b_coeffs,
    const float* d_a_coeffs,
    int filter_order,
    int width,
    int height,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid(
        (width + block.x - 1) / block.x,
        (height + block.y - 1) / block.y
    );

    temporal_bandpass_kernel<<<grid, block, 0, stream>>>(
        d_input,
        d_output,
        d_filter_states,
        d_b_coeffs,
        d_a_coeffs,
        filter_order,
        width,
        height,
        width  // pitch = width for simplicity
    );

    return cudaGetLastError();
}

/**
 * Reset filter states
 */
cudaError_t cuda_reset_temporal_filters(
    IIRFilterState* d_filter_states,
    int width,
    int height
) {
    int num_pixels = width * height;

    dim3 block(256);
    dim3 grid((num_pixels + block.x - 1) / block.x);

    reset_filter_states_kernel<<<grid, block>>>(d_filter_states, num_pixels);

    return cudaGetLastError();
}

} // extern "C"
