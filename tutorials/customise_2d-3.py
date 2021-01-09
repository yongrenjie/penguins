# contour levels reduced so that the negative ones are more visible
data_2d.stage(levels=1e5, colors=("yellowgreen", "purple"))
pg.mkplot()
pg.show()