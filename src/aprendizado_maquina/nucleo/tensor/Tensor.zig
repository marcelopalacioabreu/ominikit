const std = @import("std");
const tensorImpl = @import("TensorImplementacao.zig");
const cpu = @import("TensorCPU.zig");
const cpusimd = @import("TensorCPUSIMD.zig");
const computacao = @import("computacao");

pub const Tensor = struct {
    tipo: computacao.ComputacaoContextoModule.TipoComputacao,
    impl_ptr: *tensorImpl.BackendInstance,
    shape: []usize,
    size: usize,
    requires_grad: bool,

    pub fn init(ctx: *computacao.ComputacaoContextoModule.ComputacaoContexto, allocator: *std.mem.Allocator, shape_in: []const usize) !*Tensor {
        var total: usize = 1;
        for (shape_in) |d| total *= if (d == 0) 1 else d;

        // allocate shape storage
        var shape_buf = try allocator.alloc(usize, shape_in.len);
        for (0..shape_in.len) |i| shape_buf[i] = shape_in[i];

        // create implementation via backend
        var impl_ptr: *tensorImpl.BackendInstance = undefined;
        var vtype: computacao.ComputacaoContextoModule.TipoComputacao = undefined;
        switch (ctx.tipo) {
            .CPU => {
                impl_ptr = try cpu.create_impl(ctx, allocator, total);
                vtype = .CPU;
            },
            .CPUSIMD => {
                impl_ptr = try cpusimd.create_impl(ctx, allocator, total);
                vtype = .CPUSIMD;
            },
            else => {
                impl_ptr = try cpu.create_impl(ctx, allocator, total);
                vtype = .CPU;
            },
        }

        var obj = try allocator.create(Tensor);
        obj.tipo = vtype;
        obj.impl_ptr = impl_ptr;
        obj.shape = shape_buf[0..shape_in.len];
        obj.size = total;
        obj.requires_grad = false;
        return obj;
    }

    pub fn fromArray(ctx: *computacao.ComputacaoContextoModule.ComputacaoContexto, allocator: *std.mem.Allocator, shape_in: []const usize, data: []const f64) !*Tensor {
        const obj = try Tensor.init(ctx, allocator, shape_in);
        if (data.len != obj.size) return error.InvalidArgument;
        for (0..data.len) |i| {
            switch (obj.tipo) {
                .CPU => cpu.cpu_set(obj.impl_ptr, i, data[i]),
                .CPUSIMD => cpusimd.simd_set(obj.impl_ptr, i, data[i]),
                else => cpu.cpu_set(obj.impl_ptr, i, data[i]),
            }
        }
        return obj;
    }

    pub fn get(self: *Tensor, i: usize) f64 {
        return switch (self.tipo) {
            .CPU => cpu.cpu_get(self.impl_ptr, i),
            .CPUSIMD => cpusimd.simd_get(self.impl_ptr, i),
            else => cpu.cpu_get(self.impl_ptr, i),
        };
    }

    pub fn set(self: *Tensor, i: usize, v: f64) void {
        switch (self.tipo) {
            .CPU => cpu.cpu_set(self.impl_ptr, i, v),
            .CPUSIMD => cpusimd.simd_set(self.impl_ptr, i, v),
            else => cpu.cpu_set(self.impl_ptr, i, v),
        }
    }

    pub fn toArray(self: *Tensor, allocator: *std.mem.Allocator) anyerror![]f64 {
        return switch (self.tipo) {
            .CPU => cpu.cpu_toArray(self.impl_ptr, allocator),
            .CPUSIMD => cpusimd.simd_toArray(self.impl_ptr, allocator),
            else => cpu.cpu_toArray(self.impl_ptr, allocator),
        };
    }

    pub fn destroy(self: *Tensor, allocator: *std.mem.Allocator) void {
        switch (self.tipo) {
            .CPU => cpu.cpu_destroy(allocator, self.impl_ptr),
            .CPUSIMD => cpusimd.simd_destroy(allocator, self.impl_ptr),
            else => cpu.cpu_destroy(allocator, self.impl_ptr),
        }
        allocator.free(self.shape);
        allocator.destroy(self);
    }
};
