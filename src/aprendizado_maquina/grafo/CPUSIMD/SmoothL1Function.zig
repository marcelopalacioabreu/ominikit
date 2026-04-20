const std = @import("std");
const tensorImpl = @import("../../nucleo/tensor/TensorImplementacao.zig");
const Backend = tensorImpl.BackendInstance;
pub const SmoothL1UserData = @import("../../nucleo/tensor/TensorImplementacao.zig").SmoothL1UserData;

pub fn simd_smoothl1_backward(user: ?*u8, allocator: *std.mem.Allocator, grad: []const f64) void {
    if (user == null) return;
    const ud = @ptrCast(*SmoothL1UserData, user.*);
    const pred = ud.pred.*;
    const target = ud.target.*;
    const n = ud.n;

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
