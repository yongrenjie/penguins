fig, axs = pg.subplots2d(2, 2)
for ax in axs.flat:
    # Inside this loop, `ax` refers to an individual Axes.
    # It is also the name of the keyword parameter.
    data_2d.stage(ax=ax, levels=3e5)
    pg.mkplot(ax=ax)