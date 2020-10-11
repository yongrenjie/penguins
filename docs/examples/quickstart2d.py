import penguins as pg
hsqc = pg.read("tests/data/pt2", 4)
# Staging
hsqc.stage(f1_bounds="11..140",
           f2_bounds=(0.5, 8.5),
           levels=(2.5e4, None, None))
# Construct plot.
pg.mkplot(autolabel="nucl")
# Display
pg.show()
