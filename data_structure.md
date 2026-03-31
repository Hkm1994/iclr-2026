t	                (10,)	               ~0.409 → 0.418	                 Time window (very short transient)
pos	                (100000, 3)	           x ∈ [-0.4, 2.09]	                 3D spatial points (mesh)
idcs_airfoil        (20000,)	                                             indices	subset of mesh → airfoil surface
pressure	        (10, 100000)	       -3666 → 1251	                     pressure field
velocity_in	        (5, 100000, 3)	        -49 → 79	                     past velocity
velocity_out	    (5, 100000, 3)	        -49 → 78	                     future velocity



t (10,) — Time Vector

This is the set of discrete time steps for the simulation snapshot.

It represents a short transient window of the flow.
Each value corresponds to a specific instant in time.
From an ML perspective, this defines the temporal axis over which the system evolves.

👉 You can think of this as the “clock” of the system.


pos (100000, 3) — Spatial Coordinates (Mesh)

This defines the geometry of the simulation domain.

Each row is a point in 3D space: (x, y, z).
These points form the computational mesh used in CFD.
Even for 2D flow, a dummy third dimension may exist.

👉 For ML: this is your spatial grid / node features (positions).
👉 Everything (pressure, velocity) is defined at these points.

<u>ids_airfoil (20000,) — Airfoil Surface Indices</u>

This is a subset of indices pointing to mesh points that lie on the airfoil surface.

These points define the boundary of the object in the flow.
Critical for enforcing boundary conditions (e.g., no-slip velocity).

👉 For ML:

This tells you which nodes belong to the object
Useful for:
Masking
Loss functions
Geometry-aware models



