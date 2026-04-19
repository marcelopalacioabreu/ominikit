const std = @import("std");
const computacao = @import("computacao");
const tensor = @import("tensor");

pub fn main() anyerror!void {
    var allocator_buf = std.heap.page_allocator;
    const allocator: *std.mem.Allocator = &allocator_buf;

    var ctx = computacao.ComputacaoContextoModule.ComputacaoCPUContexto();

    const size: usize = 16;
    var shape = [_]usize{size};

    var t = try tensor.Tensor.init(&ctx, allocator, shape[0..]);
    defer t.destroy(allocator);

    // set some values
    t.set(0, 3.1415);
    t.set(1, 2.718);

    // read back
    const v0 = t.get(0);
    const v1 = t.get(1);
    std.debug.print("Tensor[0]={}, Tensor[1]={}\n", .{ v0, v1 });

    // toArray and print first 4 values
    const arr = try t.toArray(allocator);
    defer allocator.free(arr);
    std.debug.print("first four: {}, {}, {}, {}\n", .{ arr[0], arr[1], arr[2], arr[3] });
}
