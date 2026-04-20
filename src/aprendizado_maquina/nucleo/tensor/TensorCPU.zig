const std = @import("std");
const tensorImpl = @import("TensorImplementacao.zig");

// Use shared BackendInstance from TensorImplementacao
const Backend = tensorImpl.BackendInstance;

pub fn cpu_get(impl_ptr: *Backend, i: usize) f64 {
    const s = impl_ptr.*;
    return s.data[i];
}

pub fn cpu_set(impl_ptr: *Backend, i: usize, v: f64) void {
    const s = impl_ptr.*;
    s.data[i] = v;
}

pub fn cpu_toArray(impl_ptr: *Backend, allocator: *std.mem.Allocator) anyerror![]f64 {
    const s = impl_ptr.*;
    var out = try allocator.alloc(f64, s.count);
    for (0..s.count) |i| out[i] = s.data[i];
    return out;
}

pub fn cpu_destroy(allocator: *std.mem.Allocator, impl_ptr: *Backend) void {
    const s = impl_ptr.*;
    allocator.free(s.data);
    allocator.free(s.grad);
    allocator.destroy(impl_ptr);
}

const cpu_vtable = tensorImpl.TensorImplementacao{
    .get = &cpu_get,
    .set = &cpu_set,
    .toArray = &cpu_toArray,
    .destroy = &cpu_destroy,
};

pub const VTABLE = cpu_vtable;

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

pub fn cpu_matmul_backward(impl_ptr: *Backend, _: *std.mem.Allocator, grad: []const f64) void {
    const udptr = impl_ptr.user orelse return;
    const ud = udptr.*;
    const a = ud.matmul.a.*;
    const b = ud.matmul.b.*;
    const m = ud.matmul.m;
    const n = ud.matmul.n;
    const p = ud.matmul.p;

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

pub fn cpu_conv_backward(impl_ptr: *Backend, _: *std.mem.Allocator, grad: []const f64) void {
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

    for (0..hin * win) |i| inb.grad[i] = 0.0;
    for (0..kh * kw) |i| kb.grad[i] = 0.0;

    const has_scalar_upstream = grad.len == 1;
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

pub fn cpu_batchnorm_backward(impl_ptr: *Backend, _: *std.mem.Allocator, grad: []const f64) void {
    const udptr = impl_ptr.user orelse return;
    const ud = udptr.*;
    const inb = ud.batchnorm.input.*;
    const outb = ud.batchnorm.out.*;
    const n = ud.batchnorm.n;
    const denom = ud.batchnorm.denom;

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
        const dx = inv_denom * inv_n * ((denom_f * dy) - sum_dy - (y * sum_dy_y));
        inb.grad[i] += dx;
    }
}

pub fn cpu_bce_backward(impl_ptr: *Backend, _: *std.mem.Allocator, grad: []const f64) void {
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

pub fn cpu_mse_backward(impl_ptr: *Backend, _: *std.mem.Allocator, grad: []const f64) void {
    const udptr = impl_ptr.user orelse return;
    const ud = udptr.*;
    const pred = ud.mse.pred.*;
    const target = ud.mse.target.*;
    const n = ud.mse.n;
    const has_scalar_upstream = grad.len == 1;
    var denom_f2: f64 = 0.0;
    for (0..n) |_| denom_f2 += 1.0;
    const inv_n2 = 1.0 / denom_f2;
    for (0..n) |i| {
        const p = pred.data[i];
        const t = target.data[i];
        const upstream = if (has_scalar_upstream) grad[0] else grad[i];
        const dp = 2.0 * (p - t) * inv_n2;
        pred.grad[i] += upstream * dp;
    }
}

pub fn cpu_focal_backward(impl_ptr: *Backend, _: *std.mem.Allocator, grad: []const f64) void {
    const udptr = impl_ptr.user orelse return;
    const ud = udptr.*;
    const pred = ud.focal.pred.*;
    const target = ud.focal.target.*;
    const n = ud.focal.n;
    const alpha = ud.focal.alpha;
    const gamma = ud.focal.gamma;
    const has_scalar_upstream = grad.len == 1;
    const eps: f64 = 1e-6;

    // numeric gradient per element (finite differences)
    for (0..n) |i| {
        const orig = pred.data[i];
        // L(p+eps)
        pred.data[i] = orig + eps;
        var sum_plus: f64 = 0.0;
        for (0..n) |k| {
            const p = pred.data[k];
            const t = target.data[k];
            const pt = if (t == 1.0) p else (1.0 - p);
            const logpt = if (pt <= 0.0) -1e12 else std.math.log(f64, 2.718281828459045, pt);
            sum_plus += -alpha * std.math.pow(f64, 1.0 - pt, gamma) * logpt;
        }

        // L(p-eps)
        pred.data[i] = orig - eps;
        var sum_minus: f64 = 0.0;
        for (0..n) |k| {
            const p = pred.data[k];
            const t = target.data[k];
            const pt = if (t == 1.0) p else (1.0 - p);
            const logpt = if (pt <= 0.0) -1e12 else std.math.log(f64, 2.718281828459045, pt);
            sum_minus += -alpha * std.math.pow(f64, 1.0 - pt, gamma) * logpt;
        }

        // restore
        pred.data[i] = orig;

        const dp = (sum_plus - sum_minus) / (2.0 * eps);
        const upstream = if (has_scalar_upstream) grad[0] else grad[i];
        pred.grad[i] += upstream * dp;
    }
}

pub fn cpu_smoothl1_backward(impl_ptr: *Backend, _: *std.mem.Allocator, grad: []const f64) void {
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
        const upstream = if (has_scalar_upstream) grad[0] else grad[i];
        var dp: f64 = 0.0;
        if (std.math.abs(d) < 1.0) {
            dp = d;
        } else {
            dp = if (d < 0.0) -1.0 else 1.0;
        }
        pred.grad[i] += upstream * dp;
    }
}

// Old constructors removed: use Tensor.init / Tensor.initFromArray in Tensor.zig

pub const TensorError = error{ OutOfMemory, InvalidArgument };
