import penguins as pg
fig, axs = pg.subplots(2, 2)
# Set up the lists.
# 15N HMQC; 13C HSQC; COSY; NOESY
spectra = [pg.read("tests/data/noah", i, 1) for i in range(1, 5)]
levels = [7e3, 2.3e4, 8.5e5, 8.9e4]
titles = [r"$^{15}$N HMQC", r"$^{13}$C HSQC", "COSY", "NOESY"]
clr = ("blue", "red")  # we use the same colours for all spectra
# Iterate over the lists.
for spec, ax, lvl, title, char in zip(spectra, axs.flat, levels, titles, "abcd"):
    # Staging proceeds as normal
    spec.stage(ax=ax, levels=lvl, colors=clr)
    # When constructing the plot, you need to pass the correct axis instance
    pg.mkplot(ax=ax,
              title=title,
              autolabel="nucl")
    # Add a label in the top left corner of each spectrum.
    ax.text(x=0.02, y=0.97, s=f"({char})", transform=ax.transAxes,
            fontweight="semibold", verticalalignment="top")
    # Display as usual (outside the loop)
    pg.show()

