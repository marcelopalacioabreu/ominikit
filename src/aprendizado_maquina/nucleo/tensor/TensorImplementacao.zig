const std = @import("std");

pub const BackendInstance = struct {
    data: []f64,
    count: usize,
};

pub const TensorImplementacao = struct {
    get: *const fn (impl_ptr: *BackendInstance, i: usize) f64,
    set: *const fn (impl_ptr: *BackendInstance, i: usize, v: f64) void,
    toArray: *const fn (impl_ptr: *BackendInstance, allocator: *std.mem.Allocator) anyerror![]f64,
    destroy: *const fn (allocator: *std.mem.Allocator, impl_ptr: *BackendInstance) void,
};

pub const TensorError = error{OutOfMemory};
