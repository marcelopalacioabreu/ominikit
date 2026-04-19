const std = @import("std");
const tensorImpl = @import("TensorImplementacao.zig");
const computacao = @import("computacao");

// Use shared BackendInstance from TensorImplementacao
const Backend = tensorImpl.BackendInstance;

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
    allocator.destroy(impl_ptr);
}

const simd_vtable = tensorImpl.TensorImplementacao{
    .get = &simd_get,
    .set = &simd_set,
    .toArray = &simd_toArray,
    .destroy = &simd_destroy,
};

pub const VTABLE = simd_vtable;

pub fn create_impl(ctx: *computacao.ComputacaoContextoModule.ComputacaoContexto, allocator: *std.mem.Allocator, total: usize) !*Backend {
    _ = ctx;
    var impl = try allocator.create(Backend);
    impl.data = try allocator.alloc(f64, total);
    impl.count = total;
    for (0..total) |i| impl.data[i] = 0.0;
    return impl;
}

// Old constructors removed: use Tensor.init / Tensor.initFromArray in Tensor.zig

pub const TensorError = error{ OutOfMemory, InvalidArgument };
