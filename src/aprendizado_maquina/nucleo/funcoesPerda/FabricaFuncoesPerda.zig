const std = @import("std");
const computacao = @import("../../../computacao/ComputacaoContexto.zig");
const tensorMod = @import("../tensor/Tensor.zig");
const tensorImpl = @import("../tensor/TensorImplementacao.zig");
const cpu = @import("../tensor/TensorCPU.zig");
const simd = @import("../tensor/TensorCPUSIMD.zig");

pub const FabricaFuncoesPerda = struct {
    pub fn MSE(_ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, pred: *tensorMod.Tensor, target: *tensorMod.Tensor) !*tensorMod.Tensor {
        if (pred.size != target.size) return error.InvalidArgument;
        const val = @import("./CPU/MSE.zig").loss_mse(pred.impl_ptr.data, target.impl_ptr.data);
        var out = try tensorMod.Tensor.init(_ctx, allocator, &[_]usize{1});
        out.set(0, val);
        // attach userdata for backward
        const ud = try allocator.create(tensorImpl.AnyUserData);
        ud.* = .{ .mse = .{ .pred = pred.impl_ptr, .target = target.impl_ptr, .n = pred.size } };
        out.impl_ptr.user = ud;
        switch (pred.tipo) {
            .CPU => out.grad_fn = &cpu.cpu_mse_backward,
            .CPUSIMD => out.grad_fn = &simd.simd_mse_backward,
            else => out.grad_fn = &cpu.cpu_mse_backward,
        }
        return out;
    }

    pub fn BCE(_ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, pred: *tensorMod.Tensor, target: *tensorMod.Tensor) !*tensorMod.Tensor {
        if (pred.size != target.size) return error.InvalidArgument;
        // compute scalar loss using CPU implementation for now
        var sum: f64 = 0.0;
        const n = pred.size;
        for (0..n) |i| {
            const p = pred.get(i);
            const t = target.get(i);
            const eps = 1e-12;
            const pc = if (p < eps) eps else if (p > 1.0 - eps) 1.0 - eps else p;
            sum += -(t * std.math.log(f64, std.math.E, pc) + (1.0 - t) * std.math.log(f64, std.math.E, 1.0 - pc));
        }
        var denom: f64 = 0.0;
        for (0..n) |_| denom += 1.0;
        const val = sum / denom;
        var out = try tensorMod.Tensor.init(_ctx, allocator, &[_]usize{1});
        out.set(0, val);
        const ud = try allocator.create(tensorImpl.AnyUserData);
        ud.* = .{ .bce = .{ .pred = pred.impl_ptr, .target = target.impl_ptr, .n = pred.size } };
        out.impl_ptr.user = ud;
        switch (pred.tipo) {
            .CPU => out.grad_fn = &cpu.cpu_bce_backward,
            .CPUSIMD => out.grad_fn = &simd.simd_bce_backward,
            else => out.grad_fn = &cpu.cpu_bce_backward,
        }
        return out;
    }

    pub fn SmoothL1(_ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, pred: *tensorMod.Tensor, target: *tensorMod.Tensor) !*tensorMod.Tensor {
        if (pred.size != target.size) return error.InvalidArgument;
        var sum: f64 = 0.0;
        const n = pred.size;
        for (0..n) |i| {
            const d = pred.get(i) - target.get(i);
            if (std.math.abs(d) < 1.0) {
                sum += 0.5 * d * d;
            } else {
                sum += std.math.abs(d) - 0.5;
            }
        }
        var denom: f64 = 0.0;
        for (0..n) |_| denom += 1.0;
        const val = sum / denom;
        var out = try tensorMod.Tensor.init(_ctx, allocator, &[_]usize{1});
        out.set(0, val);
        const ud = try allocator.create(tensorImpl.AnyUserData);
        ud.* = .{ .smoothl1 = .{ .pred = pred.impl_ptr, .target = target.impl_ptr, .n = pred.size } };
        out.impl_ptr.user = ud;
        switch (pred.tipo) {
            .CPU => out.grad_fn = &cpu.cpu_smoothl1_backward,
            .CPUSIMD => out.grad_fn = &simd.simd_smoothl1_backward,
            else => out.grad_fn = &cpu.cpu_smoothl1_backward,
        }
        return out;
    }

    pub fn Focal(_ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, pred: *tensorMod.Tensor, target: *tensorMod.Tensor, alpha: f64, gamma: f64) !*tensorMod.Tensor {
        if (pred.size != target.size) return error.InvalidArgument;
        var sum: f64 = 0.0;
        var denom: f64 = 0.0;
        const n = pred.size;
        for (0..n) |_| denom += 1.0;
        for (0..n) |i| {
            const p = pred.get(i);
            const t = target.get(i);
            const pt = if (t == 1.0) p else (1.0 - p);
            const logpt = if (pt <= 0.0) -1e12 else std.math.log(f64, 2.718281828459045, pt);
            sum += -alpha * std.math.pow(f64, 1.0 - pt, gamma) * logpt;
        }
        const val = sum / denom;
        var out = try tensorMod.Tensor.init(_ctx, allocator, &[_]usize{1});
        out.set(0, val);
        const ud = try allocator.create(tensorImpl.AnyUserData);
        ud.* = .{ .focal = .{ .pred = pred.impl_ptr, .target = target.impl_ptr, .n = n, .alpha = alpha, .gamma = gamma } };
        out.impl_ptr.user = ud;
        switch (pred.tipo) {
            .CPU => out.grad_fn = &cpu.cpu_focal_backward,
            .CPUSIMD => out.grad_fn = &simd.simd_focal_backward,
            else => out.grad_fn = &cpu.cpu_focal_backward,
        }
        return out;
    }
};
