def objective(features):
    rudy = term("rudy_p95")
    density = term("cell_density_p95")
    score = rudy + 0.35 * density
    return score, {"rudy_hotspot": rudy, "density_hotspot": density}
