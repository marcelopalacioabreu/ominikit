const std = @import("std");
const tensorImpl = @import("TensorImplementacao.zig");

// Use shared BackendInstance from TensorImplementacao
const Backend = tensorImpl.BackendInstance;

// Use shared user-data types from TensorImplementacao

pub fn simd_get(impl_ptr: *Backend, i: usize) f64 {
    const s = impl_ptr.*;
    return s.data[i];
}

pub fn simd_set(impl_ptr: *Backend, i: usize, v: f64) void {
    const s = impl_ptr.*;
    s.data[i] = v;
}

pub fn simd_toArray(impl_ptr: *Backend, allocator: *std.mem.Allocator) anyerror![]f64 {
    const s = impl_ptr.*;
    var out = try allocator.alloc(f64, s.count);
    for (0..s.count) |i| out[i] = s.data[i];
    return out;
}

pub fn simd_destroy(allocator: *std.mem.Allocator, impl_ptr: *Backend) void {
    const s = impl_ptr.*;
    allocator.free(s.data);
    allocator.free(s.grad);
    allocator.destroy(impl_ptr);
}

const simd_vtable = tensorImpl.TensorImplementacao{
    .get = &simd_get,
    .set = &simd_set,
    .toArray = &simd_toArray,
    .destroy = &simd_destroy,
};

pub const VTABLE = simd_vtable;

pub fn create_impl(allocator: *std.mem.Allocator, total: usize) !*Backend {
    var impl = try allocator.create(Backend);
    impl.data = try allocator.alloc(f64, total);
    impl.grad = try allocator.alloc(f64, total);
    impl.count = total;
    for (0..total) |i| {
        impl.data[i] = 0.0;
        impl.grad[i] = 0.0;
    }
    // attach vtable for dispatch
    impl.vtable = &VTABLE;
    return impl;
}

// Old constructors removed: use Tensor.init / Tensor.initFromArray in Tensor.zig

pub const TensorError = error{ OutOfMemory, InvalidArgument };

// Optimized (naive, cache-friendly) matMul for CPUSIMD backend.
pub fn simd_matMul(allocator: *std.mem.Allocator, a_impl: *Backend, b_impl: *Backend, m: usize, n: usize, p: usize) !*Backend {
    const a = a_impl.*;
    const b = b_impl.*;
    var out = try create_impl(allocator, m * p);

    // Multiply with loop ordering i,k,j for cache locality on b
    for (0..m) |i| {
        for (0..n) |k| {
            const a_ik = a.data[i * n + k];
            const b_row = k * p;
            const out_row = i * p;
            for (0..p) |j| {
                out.data[out_row + j] += a_ik * b.data[b_row + j];
            }
        }
    }
    return out;
}

pub fn simd_matmul_backward(impl_ptr: *Backend, _: *std.mem.Allocator, grad: []const f64) void {
    const udptr = impl_ptr.user orelse return;
    const ud = udptr.*;
    const a = ud.matmul.a.*;
    const b = ud.matmul.b.*;
    const m = ud.matmul.m;
    const n = ud.matmul.n;
    const p = ud.matmul.p;

    // Compute gradients: a_grad = grad * b^T  (m x p) * (p x n) -> (m x n)
    // and b_grad = a^T * grad  (n x m) * (m x p) -> (n x p)
    // Zero accumulators
    for (0..m * n) |i| a.grad[i] = 0.0;
    for (0..n * p) |i| b.grad[i] = 0.0;

    const has_scalar_upstream = grad.len == 1;

    // a_grad
    for (0..m) |i| {
        for (0..n) |k| {
            var sum: f64 = 0.0;
            for (0..p) |j| {
                const upstream = if (has_scalar_upstream) grad[0] else grad[i * p + j];
                sum += upstream * b.data[k * p + j];
            }
            a.grad[i * n + k] += sum;
        }
    }

    // b_grad
    for (0..n) |k| {
        for (0..p) |j| {
            var sum: f64 = 0.0;
            for (0..m) |i| {
                const upstream = if (has_scalar_upstream) grad[0] else grad[i * p + j];
                sum += a.data[i * n + k] * upstream;
            }
            b.grad[k * p + j] += sum;
        }
    }
}

// Very small, generic 2D convolution helper. Assumes input is 2D stored row-major
// with dimensions (hin x win) and kernel (kh x kw). Returns output backend sized (hout x wout).
pub fn simd_conv(allocator: *std.mem.Allocator, input_impl: *Backend, hin: usize, win: usize, kernel_impl: *Backend, kh: usize, kw: usize) !*Backend {
    const inb = input_impl.*;
    const kb = kernel_impl.*;
    if (hin < kh or win < kw) return error.InvalidArgument;
    const hout = hin - kh + 1;
    const wout = win - kw + 1;
    var out = try create_impl(allocator, hout * wout);

    for (0..hout) |i| {
        for (0..wout) |j| {
            var sum: f64 = 0.0;
            for (0..kh) |ii| {
                for (0..kw) |jj| {
                    const in_r = i + ii;
                    const in_c = j + jj;
                    const in_idx = in_r * win + in_c;
                    const k_idx = ii * kw + jj;
                    sum += inb.data[in_idx] * kb.data[k_idx];
                }
            }
            out.data[i * wout + j] = sum;
        }
    }
    return out;
}

pub fn simd_conv_backward(impl_ptr: *Backend, _: *std.mem.Allocator, grad: []const f64) void {
    const udptr = impl_ptr.user orelse return;
    const ud = udptr.*;
    const inb = ud.conv.input.*;
    const kb = ud.conv.kernel.*;
    const hin = ud.conv.hin;
    const win = ud.conv.win;
    const kh = ud.conv.kh;
    const kw = ud.conv.kw;
    const hout = hin - kh + 1;
    const wout = win - kw + 1;

    // Zero accumulators
    for (0..hin * win) |i| inb.grad[i] = 0.0;
    for (0..kh * kw) |i| kb.grad[i] = 0.0;

    const has_scalar_upstream = grad.len == 1;
    // kernel gradients: sum over output positions
    for (0..kh) |ii| {
        for (0..kw) |jj| {
            var sum: f64 = 0.0;
            for (0..hout) |i| {
                for (0..wout) |j| {
                    const in_r = i + ii;
                    const in_c = j + jj;
                    const upstream = if (has_scalar_upstream) grad[0] else grad[i * wout + j];
                    sum += inb.data[in_r * win + in_c] * upstream;
                }
            }
            kb.grad[ii * kw + jj] += sum;
        }
    }

    // input gradients: distribute kernel * grad to input positions
    for (0..hout) |i| {
        for (0..wout) |j| {
            const g = if (has_scalar_upstream) grad[0] else grad[i * wout + j];
            for (0..kh) |ii| {
                for (0..kw) |jj| {
                    const in_r = i + ii;
                    const in_c = j + jj;
                    inb.grad[in_r * win + in_c] += kb.data[ii * kw + jj] * g;
                }
            }
        }
    }
}

// Batch-normalize in-place: normalize whole tensor to zero mean and unit variance with eps.
pub fn simd_batchnorm_inplace(impl_ptr: *Backend, epsilon: f64) void {
    const b = impl_ptr.*;
    var mean: f64 = 0.0;
    const n = b.count;
    for (0..n) |i| mean += b.data[i];
    mean /= @as(f64, n);
    var varacc: f64 = 0.0;
    for (0..n) |i| {
        const d = b.data[i] - mean;
        varacc += d * d;
    }
    varacc /= @as(f64, n);
    const denom = std.math.sqrt(varacc + epsilon);
    for (0..n) |i| b.data[i] = (b.data[i] - mean) / denom;
}

// Create a new normalized backend (out) and return it; caller may attach backward userdata.
pub fn simd_batchnorm(allocator: *std.mem.Allocator, input_impl: *Backend, epsilon: f64) !*Backend {
    const inb = input_impl.*;
    const n = inb.count;
    var mean: f64 = 0.0;
    for (0..n) |i| mean += inb.data[i];
    mean /= @as(f64, n);
    var varacc: f64 = 0.0;
    for (0..n) |i| {
        const d = inb.data[i] - mean;
        varacc += d * d;
    }
    varacc /= @as(f64, n);
    const denom = std.math.sqrt(varacc + epsilon);

    var out = try create_impl(allocator, n);
    for (0..n) |i| out.data[i] = (inb.data[i] - mean) / denom;

    return out;
}

pub fn simd_batchnorm_backward(impl_ptr: *Backend, _: *std.mem.Allocator, grad: []const f64) void {
    const udptr = impl_ptr.user orelse return;
    const ud = udptr.*;
    const inb = ud.batchnorm.input.*;
    const outb = ud.batchnorm.out.*;
    const n = ud.batchnorm.n;
    const denom = ud.batchnorm.denom;

    // compute sums
    var sum_dy: f64 = 0.0;
    var sum_dy_y: f64 = 0.0;
    for (0..n) |i| {
        const dy = grad[i];
        const y = outb.data[i];
        sum_dy += dy;
        sum_dy_y += dy * y;
    }

    var denom_f: f64 = 0.0;
    for (0..n) |_| denom_f += 1.0;
    const inv_n = 1.0 / denom_f;
    const inv_denom = 1.0 / denom;

    for (0..n) |i| {
        const dy = grad[i];
        const y = outb.data[i];
        const dx = inv_denom * inv_n * ((@as(f64, n) * dy) - sum_dy - (y * sum_dy_y));
        inb.grad[i] += dx;
    }
}

pub fn simd_mse_backward(impl_ptr: *Backend, _: *std.mem.Allocator, grad: []const f64) void {
    const udptr = impl_ptr.user orelse return;
    const ud = udptr.*;
    const pred = ud.mse.pred.*;
    const target = ud.mse.target.*;
    const n = ud.mse.n;
    const has_scalar_upstream = grad.len == 1;
    var denom_f2: f64 = 0.0;
    for (0..n) |_| denom_f2 += 1.0;
    const inv_n = 1.0 / denom_f2;
    for (0..n) |i| {
        const p = pred.data[i];
        const t = target.data[i];
        const upstream = if (has_scalar_upstream) grad[0] else grad[i];
        const dp = 2.0 * (p - t) * inv_n;
        pred.grad[i] += upstream * dp;
    }
}

pub fn simd_focal_backward(impl_ptr: *Backend, _: *std.mem.Allocator, grad: []const f64) void {
    const udptr = impl_ptr.user orelse return;
    const ud = udptr.*;
    const pred = ud.focal.pred.*;
    const target = ud.focal.target.*;
    const n = ud.focal.n;
    const alpha = ud.focal.alpha;
    const gamma = ud.focal.gamma;
    const has_scalar_upstream = grad.len == 1;
    const eps: f64 = 1e-6;

    for (0..n) |i| {
        const orig = pred.data[i];
        pred.data[i] = orig + eps;
        var sum_plus: f64 = 0.0;
        for (0..n) |k| {
            const p = pred.data[k];
            const t = target.data[k];
            const pt = if (t == 1.0) p else (1.0 - p);
            const logpt = if (pt <= 0.0) -1e12 else std.math.log(f64, 2.718281828459045, pt);
            sum_plus += -alpha * std.math.pow(f64, 1.0 - pt, gamma) * logpt;
        }
        pred.data[i] = orig - eps;
        var sum_minus: f64 = 0.0;
        for (0..n) |k| {
            const p = pred.data[k];
            const t = target.data[k];
            const pt = if (t == 1.0) p else (1.0 - p);
            const logpt = if (pt <= 0.0) -1e12 else std.math.log(f64, 2.718281828459045, pt);
            sum_minus += -alpha * std.math.pow(f64, 1.0 - pt, gamma) * logpt;
        }
        pred.data[i] = orig;
        const dp = (sum_plus - sum_minus) / (2.0 * eps);
        const upstream = if (has_scalar_upstream) grad[0] else grad[i];
        pred.grad[i] += upstream * dp;
    }
}

pub fn simd_bce_backward(impl_ptr: *Backend, _: *std.mem.Allocator, grad: []const f64) void {
    const udptr = impl_ptr.user orelse return;
    const ud = udptr.*;
    const pred = ud.bce.pred.*;
    const target = ud.bce.target.*;
    const n = ud.bce.n;

    const has_scalar_upstream = grad.len == 1;
    for (0..n) |i| {
        const p = pred.data[i];
        const t = target.data[i];
        const upstream = if (has_scalar_upstream) grad[0] else grad[i];
        const eps = 1e-12;
        const pc = if (p < eps) eps else if (p > 1.0 - eps) 1.0 - eps else p;
        const dp = -(t / pc) + ((1.0 - t) / (1.0 - pc));
        pred.grad[i] += upstream * dp;
    }
}

pub fn simd_smoothl1_backward(impl_ptr: *Backend, _: *std.mem.Allocator, grad: []const f64) void {
    const udptr = impl_ptr.user orelse return;
    const ud = udptr.*;
    const pred = ud.smoothl1.pred.*;
    const target = ud.smoothl1.target.*;
    const n = ud.smoothl1.n;

    const has_scalar_upstream = grad.len == 1;
    for (0..n) |i| {
        const p = pred.data[i];
        const t = target.data[i];
        const d = p - t;
        const absd = if (d < 0.0) -d else d;
        const upstream = if (has_scalar_upstream) grad[0] else grad[i];
        var dp: f64 = 0.0;
        if (absd < 1.0) {
            dp = d;
        } else {
            dp = if (d < 0.0) -1.0 else 1.0;
        }
        pred.grad[i] += upstream * dp;
    }
}
