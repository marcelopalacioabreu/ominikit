const std = @import("std");
const tensor = @import("../nucleo/tensor/Tensor.zig").Tensor;

pub fn normalizar(allocator: *std.mem.Allocator, t: *tensor, epsilon: f64) !*tensor {
    // Wrapper that calls Tensor.batchnorm
    return try t.batchnorm(allocator, epsilon);
}
pub fn normalizar_inplace(data: []f64) void {
    // placeholder normalization
}
