const std = @import("std");
const tensorImpl = @import("TensorImplementacao.zig");
const cpu = @import("TensorCPU.zig");
const cpusimd = @import("TensorCPUSIMD.zig");
const computacao = @import("../../../computacao/mod.zig");
const ComputacaoContextoModule = computacao.ComputacaoContextoModule;
const tipo_mod = computacao.TipoModule;

pub const Tensor = struct {
    tipo: tipo_mod.TipoComputacao,
    impl_ptr: *tensorImpl.BackendInstance,
    shape: []usize,
    size: usize,
    requires_grad: bool,
    grad_fn: ?*const fn (impl_ptr: *tensorImpl.BackendInstance, allocator: *std.mem.Allocator, grad: []const f64) void,

    pub fn init(ctx: *ComputacaoContextoModule.ComputacaoContexto, allocator: *std.mem.Allocator, shape_in: []const usize) !*Tensor {
        var total: usize = 1;
        for (shape_in) |d| total *= if (d == 0) 1 else d;

        // allocate shape storage
        var shape_buf = try allocator.alloc(usize, shape_in.len);
        for (0..shape_in.len) |i| shape_buf[i] = shape_in[i];

        // create implementation via backend
        var impl_ptr: *tensorImpl.BackendInstance = undefined;
        var vtype: tipo_mod.TipoComputacao = undefined;
        switch (ctx.tipo) {
            .CPU => {
                impl_ptr = try cpu.create_impl(allocator, total);
                vtype = .CPU;
            },
            .CPUSIMD => {
                impl_ptr = try cpusimd.create_impl(allocator, total);
                vtype = .CPUSIMD;
            },
            else => {
                impl_ptr = try cpu.create_impl(allocator, total);
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

    pub fn fromArray(ctx: *ComputacaoContextoModule.ComputacaoContexto, allocator: *std.mem.Allocator, shape_in: []const usize, data: []const f64) !*Tensor {
        const obj = try Tensor.init(ctx, allocator, shape_in);
        if (data.len != obj.size) return error.InvalidArgument;
        for (0..data.len) |i| {
            (obj.impl_ptr.vtable.*.set)(obj.impl_ptr, i, data[i]);
        }
        return obj;
    }

    pub fn get(self: *Tensor, i: usize) f64 {
        return (self.impl_ptr.vtable.*.get)(self.impl_ptr, i);
    }

    pub fn set(self: *Tensor, i: usize, v: f64) void {
        (self.impl_ptr.vtable.*.set)(self.impl_ptr, i, v);
    }

    pub fn toArray(self: *Tensor, allocator: *std.mem.Allocator) anyerror![]f64 {
        return (self.impl_ptr.vtable.*.toArray)(self.impl_ptr, allocator);
    }

    pub fn backward(self: *Tensor, allocator: *std.mem.Allocator, grad: []const f64) void {
        if (self.grad_fn) |fnptr| fnptr(self.impl_ptr, allocator, grad);
    }

    pub fn destroy(self: *Tensor, allocator: *std.mem.Allocator) void {
        (self.impl_ptr.vtable.*.destroy)(allocator, self.impl_ptr);
        allocator.free(self.shape);
        allocator.destroy(self);
    }

    pub fn add(self: *Tensor, allocator: *std.mem.Allocator, other: *Tensor) !*Tensor {
        if (!std.mem.eql(usize, self.shape, other.shape)) return error.IncompatibleShapes;
        const res = try Tensor.init_with_type(self.tipo, allocator, self.shape);
        for (0..self.size) |i| {
            res.set(i, self.get(i) + other.get(i));
        }
        return res;
    }

    pub fn sub(self: *Tensor, allocator: *std.mem.Allocator, other: *Tensor) !*Tensor {
        if (!std.mem.eql(usize, self.shape, other.shape)) return error.IncompatibleShapes;
        const res = try Tensor.init_with_type(self.tipo, allocator, self.shape);
        for (0..self.size) |i| {
            res.set(i, self.get(i) - other.get(i));
        }
        return res;
    }

    pub fn mulScalar(self: *Tensor, allocator: *std.mem.Allocator, scalar: f64) !*Tensor {
        const res = try Tensor.init_with_type(self.tipo, allocator, self.shape);
        for (0..self.size) |i| {
            res.set(i, self.get(i) * scalar);
        }
        return res;
    }

    pub fn matMul(self: *Tensor, allocator: *std.mem.Allocator, other: *Tensor) !*Tensor {
        if (self.shape.len != 2 or other.shape.len != 2) return error.Not2D;
        const m = self.shape[0];
        const n = self.shape[1];
        const n2 = other.shape[0];
        const p = other.shape[1];
        if (n != n2) return error.IncompatibleInnerDims;

        var result: *Tensor = undefined;
        switch (self.tipo) {
            .CPU => {
                const res_shape = [_]usize{ m, p };
                result = try Tensor.init_with_type(self.tipo, allocator, &res_shape);
                for (0..m) |i| {
                    for (0..p) |j| {
                        var sum: f64 = 0.0;
                        for (0..n) |k| {
                            sum += self.get(i * n + k) * other.get(k * p + j);
                        }
                        result.set(i * p + j, sum);
                    }
                }
                // attach CPU matmul backward userdata and callback
                const ud = try allocator.create(tensorImpl.AnyUserData);
                ud.* = .{ .matmul = .{ .a = self.impl_ptr, .b = other.impl_ptr, .m = m, .n = n, .p = p } };
                result.impl_ptr.user = ud;
                result.grad_fn = &cpu.cpu_matmul_backward;
            },
            .CPUSIMD => {
                const out_impl = try cpusimd.simd_matMul(allocator, self.impl_ptr, other.impl_ptr, m, n, p);
                var obj = try allocator.create(Tensor);
                obj.tipo = .CPUSIMD;
                obj.impl_ptr = out_impl;
                var shape_buf = try allocator.alloc(usize, 2);
                shape_buf[0] = m;
                shape_buf[1] = p;
                obj.shape = shape_buf[0..2];
                obj.size = m * p;
                obj.requires_grad = false;
                // allocate and attach matmul user-data for backward
                const ud = try allocator.create(tensorImpl.AnyUserData);
                ud.* = .{ .matmul = .{ .a = self.impl_ptr, .b = other.impl_ptr, .m = m, .n = n, .p = p } };
                out_impl.user = ud;
                obj.grad_fn = &cpusimd.simd_matmul_backward;
                result = obj;
            },
            else => {
                const res_shape = [_]usize{ m, p };
                result = try Tensor.init_with_type(self.tipo, allocator, &res_shape);
                for (0..m) |i| {
                    for (0..p) |j| {
                        var sum: f64 = 0.0;
                        for (0..n) |k| {
                            sum += self.get(i * n + k) * other.get(k * p + j);
                        }
                        result.set(i * p + j, sum);
                    }
                }
            },
        }
        return result;
    }

    pub fn batchnorm(self: *Tensor, allocator: *std.mem.Allocator, epsilon: f64) !*Tensor {
        var result: *Tensor = undefined;
        switch (self.tipo) {
            .CPU => {
                // CPU fallback: perform in-place-like normalization into new tensor
                const n = self.size;
                var data = try allocator.alloc(f64, n);
                var mean: f64 = 0.0;
                for (0..n) |i| mean += self.get(i);
                mean /= @as(f64, n);
                var varacc: f64 = 0.0;
                for (0..n) |i| {
                    const d = self.get(i) - mean;
                    varacc += d * d;
                }
                varacc /= @as(f64, n);
                const denom = std.math.sqrt(varacc + epsilon);
                for (0..n) |i| data[i] = (self.get(i) - mean) / denom;

                const res = try Tensor.init_with_type(self.tipo, allocator, self.shape);
                for (0..n) |i| res.set(i, data[i]);
                allocator.free(data);
                // attach CPU batchnorm userdata and backward
                const ud = try allocator.create(tensorImpl.AnyUserData);
                ud.* = .{ .batchnorm = .{ .input = self.impl_ptr, .out = res.impl_ptr, .denom = denom, .n = n } };
                res.impl_ptr.user = ud;
                res.grad_fn = &cpu.cpu_batchnorm_backward;
                result = res;
            },
            .CPUSIMD => {
                const out_impl = try cpusimd.simd_batchnorm(allocator, self.impl_ptr, epsilon);
                var obj = try allocator.create(Tensor);
                obj.tipo = .CPUSIMD;
                obj.impl_ptr = out_impl;
                var shape_buf = try allocator.alloc(usize, self.shape.len);
                for (0..self.shape.len) |i| shape_buf[i] = self.shape[i];
                obj.shape = shape_buf[0..self.shape.len];
                obj.size = self.size;
                obj.requires_grad = false;
                // compute denom and attach userdata
                // recompute denom to store in user data
                var mean: f64 = 0.0;
                for (0..self.size) |i| mean += self.get(i);
                mean /= @as(f64, self.size);
                var varacc: f64 = 0.0;
                for (0..self.size) |i| {
                    const d = self.get(i) - mean;
                    varacc += d * d;
                }
                varacc /= @as(f64, self.size);
                const denom = std.math.sqrt(varacc + epsilon);

                const ud = try allocator.create(tensorImpl.AnyUserData);
                ud.* = .{ .batchnorm = .{ .input = self.impl_ptr, .out = out_impl, .denom = denom, .n = self.size } };
                out_impl.user = ud;
                obj.grad_fn = &cpusimd.simd_batchnorm_backward;
                result = obj;
            },
            else => return error.InvalidArgument,
        }
        return result;
    }

    pub fn conv(self: *Tensor, allocator: *std.mem.Allocator, kernel: *Tensor) !*Tensor {
        if (self.shape.len != 2 or kernel.shape.len != 2) return error.Not2D;
        const hin = self.shape[0];
        const win = self.shape[1];
        const kh = kernel.shape[0];
        const kw = kernel.shape[1];
        if (hin < kh or win < kw) return error.InvalidArgument;

        var result: *Tensor = undefined;
        switch (self.tipo) {
            .CPU => {
                const hout = hin - kh + 1;
                const wout = win - kw + 1;
                const res_shape = [_]usize{ hout, wout };
                result = try Tensor.init_with_type(self.tipo, allocator, &res_shape);
                for (0..hout) |i| {
                    for (0..wout) |j| {
                        var sum: f64 = 0.0;
                        for (0..kh) |ii| {
                            for (0..kw) |jj| {
                                const in_r = i + ii;
                                const in_c = j + jj;
                                sum += self.get(in_r * win + in_c) * kernel.get(ii * kw + jj);
                            }
                        }
                        result.set(i * wout + j, sum);
                    }
                }
                // attach CPU conv backward userdata and callback
                const cud = try allocator.create(tensorImpl.AnyUserData);
                cud.* = .{ .conv = .{ .input = self.impl_ptr, .kernel = kernel.impl_ptr, .hin = hin, .win = win, .kh = kh, .kw = kw } };
                result.impl_ptr.user = cud;
                result.grad_fn = &cpu.cpu_conv_backward;
            },
            .CPUSIMD => {
                const out_impl = try cpusimd.simd_conv(allocator, self.impl_ptr, hin, win, kernel.impl_ptr, kh, kw);
                var obj = try allocator.create(Tensor);
                obj.tipo = .CPUSIMD;
                obj.impl_ptr = out_impl;
                var shape_buf = try allocator.alloc(usize, 2);
                shape_buf[0] = hin - kh + 1;
                shape_buf[1] = win - kw + 1;
                obj.shape = shape_buf[0..2];
                obj.size = (hin - kh + 1) * (win - kw + 1);
                obj.requires_grad = false;
                const ud = try allocator.create(tensorImpl.AnyUserData);
                ud.* = .{ .conv = .{ .input = self.impl_ptr, .kernel = kernel.impl_ptr, .hin = hin, .win = win, .kh = kh, .kw = kw } };
                out_impl.user = ud;
                obj.grad_fn = &cpusimd.simd_conv_backward;
                result = obj;
            },
            else => {
                const hout = hin - kh + 1;
                const wout = win - kw + 1;
                const res_shape = [_]usize{ hout, wout };
                result = try Tensor.init_with_type(self.tipo, allocator, &res_shape);
                for (0..hout) |i| {
                    for (0..wout) |j| {
                        var sum: f64 = 0.0;
                        for (0..kh) |ii| {
                            for (0..kw) |jj| {
                                const in_r = i + ii;
                                const in_c = j + jj;
                                sum += self.get(in_r * win + in_c) * kernel.get(ii * kw + jj);
                            }
                        }
                        result.set(i * wout + j, sum);
                    }
                }
            },
        }
        return result;
    }

    pub fn transpose(self: *Tensor, allocator: *std.mem.Allocator) !*Tensor {
        if (self.shape.len != 2) return error.Not2D;
        const m = self.shape[0];
        const n = self.shape[1];
        const res_shape = [_]usize{ n, m };
        const res = try Tensor.init_with_type(self.tipo, allocator, &res_shape);
        for (0..m) |i| {
            for (0..n) |j| {
                res.set(j * m + i, self.get(i * n + j));
            }
        }
        return res;
    }

    fn init_with_type(tipo: tipo_mod.TipoComputacao, allocator: *std.mem.Allocator, shape_in: []const usize) !*Tensor {
        var total: usize = 1;
        for (shape_in) |d| total *= if (d == 0) 1 else d;
        var shape_buf = try allocator.alloc(usize, shape_in.len);
        for (0..shape_in.len) |i| shape_buf[i] = shape_in[i];

        var impl_ptr: *tensorImpl.BackendInstance = undefined;
        switch (tipo) {
            .CPU => impl_ptr = try cpu.create_impl(allocator, total),
            .CPUSIMD => impl_ptr = try cpusimd.create_impl(allocator, total),
            else => impl_ptr = try cpu.create_impl(allocator, total),
        }

        var obj = try allocator.create(Tensor);
        obj.tipo = tipo;
        obj.impl_ptr = impl_ptr;
        obj.shape = shape_buf;
        obj.size = total;
        obj.requires_grad = false;
        return obj;
    }
};
