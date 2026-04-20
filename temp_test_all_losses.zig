const std = @import("std");
const computacao = @import("src/computacao/ComputacaoContexto.zig");
const FabricaFuncoesPerda = @import("src/aprendizado_maquina/nucleo/funcoesPerda/FabricaFuncoesPerda.zig").FabricaFuncoesPerda;
const FabricaTensor = @import("src/aprendizado_maquina/nucleo/tensor/FabricaTensor.zig").FabricaTensor;

// Import the testes file so tests are executed under project root
const _ = @import("testes/test_losses_root.zig");
