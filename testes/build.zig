const std = @import("std");

pub fn build(b: *std.Build) void {
    const test_conv = b.addExecutable("test_conv", "test_conv_root.zig");
    const test_losses = b.addExecutable("test_losses", "test_losses_root.zig");
    b.installArtifact(test_conv);
    b.installArtifact(test_losses);
}
