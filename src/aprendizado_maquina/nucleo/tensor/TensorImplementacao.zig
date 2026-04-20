const std = @import("std");

pub const TensorImplementacao = struct {
    get: *const fn (impl_ptr: *BackendInstance, i: usize) f64,
    set: *const fn (impl_ptr: *BackendInstance, i: usize, v: f64) void,
    toArray: *const fn (impl_ptr: *BackendInstance, allocator: *std.mem.Allocator) anyerror![]f64,
    destroy: *const fn (allocator: *std.mem.Allocator, impl_ptr: *BackendInstance) void,
};

pub const BackendInstance = struct {
    vtable: *const TensorImplementacao,
    data: []f64,
    grad: []f64,
    count: usize,
    user: ?*AnyUserData,
};

pub const MatMulUserData = struct {
    a: *BackendInstance,
    b: *BackendInstance,
    m: usize,
    n: usize,
    p: usize,
};

pub const ConvUserData = struct {
    input: *BackendInstance,
    kernel: *BackendInstance,
    hin: usize,
    win: usize,
    kh: usize,
    kw: usize,
};

pub const BatchNormUserData = struct {
    input: *BackendInstance,
    out: *BackendInstance,
    denom: f64,
    n: usize,
};

pub const MSEUserData = struct {
    pred: *BackendInstance,
    target: *BackendInstance,
    n: usize,
};

pub const BCEUserData = struct {
    pred: *BackendInstance,
    target: *BackendInstance,
    n: usize,
};

pub const SmoothL1UserData = struct {
    pred: *BackendInstance,
    target: *BackendInstance,
    n: usize,
};

pub const FocalUserData = struct {
    pred: *BackendInstance,
    target: *BackendInstance,
    n: usize,
    alpha: f64,
    gamma: f64,
};

pub const AnyUserData = union(enum) {
    matmul: MatMulUserData,
    conv: ConvUserData,
    batchnorm: BatchNormUserData,
    mse: MSEUserData,
    focal: FocalUserData,
    bce: BCEUserData,
    smoothl1: SmoothL1UserData,
};

pub const TensorError = error{OutOfMemory};
