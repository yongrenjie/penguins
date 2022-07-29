import penguins as pg

ds = pg.read('.', 101)

# Create the main plot (showing the entire spectrum)
fig, ax = pg.subplots(figsize=(6, 4))
ds.stage(ax)
pg.mkplot(ax)

# Create a new Axes to house the inset
inset_ax = pg.mkinset(ax, (0.05, 0.4), (0.55, 0.4))

# Stage the dataset again on the new Axes, with different bounds
ds.stage(ax=inset_ax, bounds=(6.25, 7.6))
# Construct plot and suppress the default x-label
pg.mkplot(inset_ax, xlabel="")

pg.show()
