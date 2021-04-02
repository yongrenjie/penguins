import penguins as pg
hmqc, hsqc, cosy, noesy = (pg.read("penguins-testdata", expno)
                           for expno in range(22001, 22005))

fig, axs = pg.subplots2d(2, 2)
hmqc.stage(axs[0, 0], levels=7e3, f1_bounds="110..130", f2_bounds="7..9.5")
hsqc.stage(axs[0, 1], levels=4e4, f1_bounds="12..65", f2_bounds="0.5..5")
cosy.stage(axs[1, 0], levels=8e5)
noesy.stage(axs[1, 1], levels=1e5)

titles = [r"$^{15}$N HMQC", r"$^{13}$C HSQC", "COSY", "NOESY"]
for ax, title in zip(axs.flat, titles):
    pg.mkplot(ax, title=title)
    pg.ymove(ax, pos="topright")

pg.label_axes(axs, fstr="({})", fontweight="semibold", fontsize=12)
pg.cleanup_axes()
pg.savefig("./docs/images/readme_example.png", dpi=500)
