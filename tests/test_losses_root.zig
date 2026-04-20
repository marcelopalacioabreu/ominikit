const std = @import("std");
const computacao = @import("src/computacao/ComputacaoContexto.zig");
const FabricaFuncoesPerda = @import("src/aprendizado_maquina/nucleo/funcoesPerda/FabricaFuncoesPerda.zig").FabricaFuncoesPerda;
const FabricaTensor = @import("src/aprendizado_maquina/nucleo/tensor/FabricaTensor.zig").FabricaTensor;

test "MSE loss backward CPU and CPUSIMD" {
    var alloc = std.heap.page_allocator;
    var ctx = computacao.ComputacaoCPUContexto();
    var ft = FabricaTensor.init(&ctx, &alloc);
    const a = try ft.fromArray(&[_]usize{4}, &[_]f64{ 1.0, 2.0, 3.0, 4.0 });
    defer a.destroy(&alloc);
    const b = try ft.fromArray(&[_]usize{4}, &[_]f64{ 0.0, 0.0, 0.0, 0.0 });
    defer b.destroy(&alloc);
    const out = try FabricaFuncoesPerda.MSE(&ctx, &alloc, a, b);
    defer out.destroy(&alloc);
    const grad = [_]f64{1.0};
    out.backward(&alloc, &grad);
    var any_nonzero: bool = false;
    for (0..a.size) |i| if (a.impl_ptr.grad[i] != 0.0) any_nonzero = true;
    try std.testing.expect(any_nonzero);

    // CPUSIMD
    var ctx2 = computacao.ComputacaoCPUSIMDContexto();
    var ft2 = FabricaTensor.init(&ctx2, &alloc);
    const a2 = try ft2.fromArray(&[_]usize{4}, &[_]f64{ 1.0, 2.0, 3.0, 4.0 });
    defer a2.destroy(&alloc);
    const b2 = try ft2.fromArray(&[_]usize{4}, &[_]f64{ 0.0, 0.0, 0.0, 0.0 });
    defer b2.destroy(&alloc);
    const out2 = try FabricaFuncoesPerda.MSE(&ctx2, &alloc, a2, b2);
    defer out2.destroy(&alloc);
    out2.backward(&alloc, &grad);
    var any_nonzero2: bool = false;
    for (0..a2.size) |i| if (a2.impl_ptr.grad[i] != 0.0) any_nonzero2 = true;
    try std.testing.expect(any_nonzero2);
}
