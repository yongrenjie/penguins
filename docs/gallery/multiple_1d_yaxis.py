import penguins as pg


dss = pg.read('.', [106, 107, 108])

# Define the three spectral regions we want to plot
regions = [(5.05, 5.25),   # desired signal
           (4.60, 4.80),   # water
           (4.46, 4.66)]   # desired signal

# Set up subplot grid
fig, axs = pg.subplots(3, 3, figsize=(6, 3.3))

# Plot the spectra row by row
for i, (r, axs_row) in enumerate(zip(regions, axs)):
    for ds, ax in zip(dss, axs_row):
        ds.stage(ax=ax, bounds=r)
        # i == 2 represents the bottom row, which we want the xlabel for
        pg.mkplot(ax, xlabel=("$^1$H (ppm)" if i == 2 else ""))


# Synchronise y-axes for each row.
reference_ylim = axs[0][0].get_ylim()

for i, axs_row in enumerate(axs):
    # For the middle row (i = 1), the displayed spectra contain the water peak
    # which is most intense, so we want to scale it down to fit in the same
    # height.
    if i == 1:
        # Figure out what y-limits to use for this row
        ymin = min(ax.get_ylim()[0] for ax in axs_row)
        ymax = max(ax.get_ylim()[1] for ax in axs_row)
        # Multiplying ymin by 2 is mostly for aesthetics, it gives us some
        # wiggle room below of the spectrum.
        ymin = 2 * ymin
        # Calculate magnification factor relative to the other rows
        magnification = ((ymax - ymin)
                         / (reference_ylim[1] - reference_ylim[0]))
        for ax in axs_row:
            ax.set_ylim((ymin, ymax))
            ax.text(x=regions[1][0], y=0.5,
                    s=f"/{magnification:.0f}", color="black",
                    transform=ax.get_xaxis_transform(),
                    horizontalalignment="right", fontsize=10)

    # For the top and bottom rows, we simply set the ylims to be equal to the
    # top-left axes.
    else:
        for ax in axs_row:
            ax.set_ylim(reference_ylim)

pg.label_axes([ax_row[0] for ax_row in axs],
              fstr="({})", fontweight="semibold")

pg.show()
