const std = @import("std");
const tensor = @import("../nucleo/tensor/Tensor.zig").Tensor;

pub fn toJson(allocator: *std.mem.Allocator, t: *tensor) anyerror![]u8 {
    const arr = try t.toArray(allocator);
    var writer = std.json.Value.initRoot(allocator);
    // Create a JSON array of numbers
    var a = try writer.rootArray();
    for (arr) |v| try a.appendNumber(v);
    const bytes = try writer.toStringAlloc(allocator, .{});
    return bytes;
}
pub fn serializar(data: []const f64) []u8 {
    return []u8{}; // placeholder
}
