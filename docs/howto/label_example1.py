import penguins as pg

fig, axs = pg.subplots(1, 4, figsize=(4, 2))

pg.label_axes(axs, fstr="({})", fontweight="semibold")
