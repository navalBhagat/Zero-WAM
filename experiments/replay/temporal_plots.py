"""Reference-target trajectories and compact robot/target contact sheets."""


def figures(output, rows, selector, reference, observations):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    x = [r["chunk"] for r in rows]
    for key, label in [
        ("matched_frame", "Matched frame"),
        ("frame_target", "Frame lookahead"),
        ("distance_target", "DINO-distance lookahead"),
    ]:
        axes[0].plot(x, [r[key] for r in rows], marker="o", label=label)
    axes[0].set(
        ylabel="Human reference frame",
        title="Same observations and alignment; different future targets",
    )
    axes[0].legend()
    axes[1].plot(range(len(selector.coordinate)), selector.coordinate)
    axes[1].set(
        xlabel="Human reference frame",
        ylabel="Cumulative cosine distance",
        title=f"Distance threshold = {selector.distance:.6g}; smoothing = {selector.smoothing}",
    )
    fig.savefig(output / "targets.png", dpi=170)
    plt.close(fig)
    # At most six evenly spaced observed chunks, including the endpoints.
    chosen = sorted(
        {
            round(i * (len(rows) - 1) / max(1, min(6, len(rows)) - 1))
            for i in range(min(6, len(rows)))
        }
    )
    sheet = Image.new("RGB", (960, 205 * len(chosen)), "white")
    draw = ImageDraw.Draw(sheet)
    for line, index in enumerate(chosen):
        row = rows[index]
        y = line * 205
        images = [observations[index]] + [
            Image.fromarray(
                (reference[row[key]].permute(1, 2, 0).clamp(0, 1).numpy() * 255).astype(
                    "uint8"
                )
            )
            for key in ("frame_target", "distance_target")
        ]
        labels = [
            f"Robot chunk {row['chunk']}",
            f"+frames target {row['frame_target']}",
            f"Distance target {row['distance_target']}",
        ]
        for column, (img, label) in enumerate(zip(images, labels)):
            img = img.copy()
            img.thumbnail((310, 175))
            sheet.paste(img, (column * 320, y + 25))
            draw.text((column * 320 + 5, y + 5), label, fill="black")
    sheet.save(output / "comparison.png")
