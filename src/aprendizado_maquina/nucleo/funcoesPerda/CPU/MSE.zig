pub fn loss_mse(a: []const f64, b: []const f64) f64 {
    var s: f64 = 0.0;
    for (0..a.len) |i| s += (a[i] - b[i]) * (a[i] - b[i]);
    var denom: f64 = 0.0;
    for (0..a.len) |_| denom += 1.0;
    return s / denom;
}
