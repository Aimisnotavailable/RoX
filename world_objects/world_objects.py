# Redefine grid system to be localized and be rendered and built only once
# Make the entire world to be an offgrid format
# Allowing world objects to have their own rotation without affecting the entire world
# So in short
# Voxels are built within a local grid system but placed in a free offgrid world, need to find a way to reimplement raycasting to adjust for this option
# This will make huge objects to be interactable and at the same time fast enough to not be drawn within the world space for every draw call
# Oh boi this is a lot I hope this works