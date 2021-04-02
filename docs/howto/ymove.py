import numpy as np
import matplotlib.pyplot as plt
import penguins as pg
fig, axs = pg.subplots2d(2, 2)
ds = pg.read(".", 2)
styles = ["none", "topright", "midright", "topspin"]

# Stage and construct -- this will probably be familiar
for ax, style in zip(axs.flat, styles):
    ds.stage(ax, levels=5e5, f1_bounds="0.3..7", f2_bounds="0.3..7")
    pg.mkplot(ax, title=style)

# Apply the styles.
for ax, style in zip(axs.flat[1:], styles[1:]):
    pg.ymove(ax, style)

# Always call cleanup_axes() after ymove()!
pg.cleanup_axes()

# This is not necessary in a real plot and is only included to make
# it clear which plot is which.
plt.subplots_adjust(hspace=0.3, wspace=0.3)

pg.show()
