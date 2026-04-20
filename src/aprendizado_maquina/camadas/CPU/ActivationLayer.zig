const std = @import("std");
const computacao = @import("C:/PROJETOS/OMINIKIT/src/computacao/ComputacaoContexto.zig");
const tensor = @import("C:/PROJETOS/OMINIKIT/src/aprendizado_maquina/nucleo/tensor/Tensor.zig").Tensor;

pub const ActivationLayer = struct {
    pub fn init(allocator: *std.mem.Allocator) !*ActivationLayer {
        const obj = try allocator.create(ActivationLayer);
        return obj;
    }

    pub fn relu(_self: *ActivationLayer, allocator: *std.mem.Allocator, input: *tensor) !*tensor {
        _ = _self;
        const out = try tensor.init_with_type(input.tipo, allocator, input.shape);
        for (0..input.size) |i| {
            const v = input.get(i);
            if (v > 0.0) out.set(i, v) else out.set(i, 0.0);
        }
        return out;
    }
};
