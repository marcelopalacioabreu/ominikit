const std = @import("std");

pub fn build(b: *std.Build) void {
    const optimize = b.standardOptimizeOption(.{});

    // module do pacote `computacao` (usa o código em ../../src/computacao)
    const computacao_mod = b.createModule(.{
        .root_source_file = b.path("../../src/computacao/mod.zig"),
        .optimize = optimize,
        .target = b.graph.host,
    });

    // module do pacote `tensor` (usa o código em ../../src/aprendizado_maquina/nucleo/tensor/Tensor.zig)
    const tensor_mod = b.createModule(.{
        .root_source_file = b.path("../../src/aprendizado_maquina/nucleo/tensor/Tensor.zig"),
        .optimize = optimize,
        .target = b.graph.host,
    });

    // allow tensor module to import computacao module
    tensor_mod.addImport("computacao", computacao_mod);

    const app_mod = b.createModule(.{
        .root_source_file = b.path("exemplo_0001/main.zig"),
        .optimize = optimize,
        .target = b.graph.host,
    });

    // tornar o módulo importável como @import("computacao/...")
    app_mod.addImport("computacao", computacao_mod);
    app_mod.addImport("tensor", tensor_mod);

    const exe = b.addExecutable(.{
        .name = "exemplo_0001",
        .root_module = app_mod,
    });

    b.default_step.dependOn(&exe.step);
}
