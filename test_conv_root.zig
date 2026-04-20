const std = @import("std");

const computacao = @import("src/computacao/ComputacaoContexto.zig");
const FabricaTensor = @import("src/aprendizado_maquina/nucleo/tensor/FabricaTensor.zig").FabricaTensor;

test "conv and backward CPU (root test)" {
    var allocator = std.heap.page_allocator;
    var ctx = computacao.ComputacaoCPUContexto();
    var ft = FabricaTensor.init(&ctx, &allocator);

    const input = try ft.criar(&[_]usize{ 4, 4 });
    defer input.destroy(&allocator);
    const kernel = try ft.criar(&[_]usize{ 2, 2 });
    defer kernel.destroy(&allocator);

    for (0..input.size) |i| input.set(i, 1.0);
    for (0..kernel.size) |i| kernel.set(i, 1.0);

    const out = try input.conv(&allocator, kernel);
    defer out.destroy(&allocator);

    const grad = [_]f64{1.0};
    out.backward(&allocator, &grad);

    var any_nonzero: bool = false;
    for (0..input.size) |i| {
        if (input.impl_ptr.grad[i] != 0.0) any_nonzero = true;
    }
    try std.testing.expect(any_nonzero);
}

test "conv and backward CPUSIMD (root test)" {
    var allocator = std.heap.page_allocator;
    var ctx = computacao.ComputacaoCPUSIMDContexto();
    var ft = FabricaTensor.init(&ctx, &allocator);

    const input = try ft.criar(&[_]usize{ 4, 4 });
    defer input.destroy(&allocator);
    const kernel = try ft.criar(&[_]usize{ 2, 2 });
    defer kernel.destroy(&allocator);

    for (0..input.size) |i| input.set(i, 1.0);
    for (0..kernel.size) |i| kernel.set(i, 1.0);

    const out = try input.conv(&allocator, kernel);
    defer out.destroy(&allocator);

    const grad = [_]f64{1.0};
    out.backward(&allocator, &grad);

    var any_nonzero: bool = false;
    for (0..input.size) |i| {
        if (input.impl_ptr.grad[i] != 0.0) any_nonzero = true;
    }
    try std.testing.expect(any_nonzero);
}
