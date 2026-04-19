const std = @import("std");
const tensorImpl = @import("TensorImplementacao.zig");
const computacao = @import("computacao");

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
    allocator.destroy(impl_ptr);
}

const cpu_vtable = tensorImpl.TensorImplementacao{
    .get = &cpu_get,
    .set = &cpu_set,
    .toArray = &cpu_toArray,
    .destroy = &cpu_destroy,
};

pub const VTABLE = cpu_vtable;

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
