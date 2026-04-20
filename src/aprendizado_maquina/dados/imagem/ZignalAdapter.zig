const std = @import("std");

// Adapter that uses vendored zignal (vendor/zignal) to load images from
// in-memory buffers. This avoids any C stdio or non-portable stdlib IO.

const Manipulador = @import("./ManipuladorDeImagem.zig");

pub fn loadAsGray(allocator: *std.mem.Allocator, path: []const u8) !Manipulador.GrayImage {
    _ = allocator;
    _ = path;
    return error.FileNotFound;
}

pub fn loadAsGrayFromBytes(allocator: *std.mem.Allocator, data: []const u8) !Manipulador.GrayImage {
    const zimg = @import("../../../../vendor/zignal/src/image.zig");
    const zcolor = @import("../../../../vendor/zignal/src/color.zig");
    const Rgba = zcolor.Rgba(u8);
    const Img = zimg.Image(Rgba);

    var img = try Img.loadFromBytes(allocator.*, data);
    defer img.deinit(allocator.*);

    const rows = @as(usize, img.rows);
    const cols = @as(usize, img.cols);
    const px_count = rows * cols;

    var out = try allocator.alloc(u8, px_count);

    // Convert RGB(A) -> grayscale using Rec. 709 luma coefficients
    for (0..px_count) |i| {
        const p = img.data[i];
        // Integer luma approximation (Rec.709 scaled by 10000) to avoid float casts
        const r_u = @as(u32, p.r);
        const g_u = @as(u32, p.g);
        const b_u = @as(u32, p.b);
        const gray_i = (@as(u32, 2126) * r_u + @as(u32, 7152) * g_u + @as(u32, 722) * b_u + 5000) / 10000;
        const gval: u8 = @intCast(gray_i);
        out[i] = gval;
    }

    return Manipulador.GrayImage{ .buf = out, .width = cols, .height = rows };
}
