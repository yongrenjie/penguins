import penguins as pg

data_2d = pg.read(".", 2, 1)
data_2d.stage(levels=2e5)
fig, ax = pg.mkplot()

ax.text(x=0.5, y=0.5, s="A peak")

ax.text(x=0.5, y=0.5, s="Middle", transform=ax.transAxes

pg.show()
