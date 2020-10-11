import penguins as pg
prot = pg.read("tests/data/rot1", 1)
# Staging
prot.stage(bounds="2..7",   # plot between 2 and 7 ppm
           label=r"My proton spectrum")
# Construct
pg.mkplot()
# Display
pg.show()
