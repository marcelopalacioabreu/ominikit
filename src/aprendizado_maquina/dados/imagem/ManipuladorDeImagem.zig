const std = @import("std");
const tensor = @import("../../nucleo/tensor/Tensor.zig").Tensor;
const computacao = @import("../../../computacao/mod.zig");

pub fn transformarEmTensor(ctx: *computacao.ComputacaoContexto, allocator: *std.mem.Allocator, width: usize, height: usize) !*tensor {
    // Minimal stub: return a zero-initialized grayscale tensor of shape [height,width]
    const ft = @import("../../nucleo/tensor/FabricaTensor.zig").FabricaTensor.init(ctx, allocator);
    const t = try ft.criar(&[_]usize{ height, width });
    for (0..t.size) |i| t.set(i, 0.0);
    return t;
}
const std = @import("std");

pub fn carregar_bmp(path: []const u8) anyerror!void {
    // placeholder: load bitmap and convert to tensor in future
}
