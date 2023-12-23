import numpy as np
import sklearn.metrics.pairwise as sp
from dolfinx import mesh, fem, io

from ufl import (
    Identity,
    sym,
    grad,
    tr,
    dx,
    sqrt,
    inner,
    Measure,
    TrialFunction,
    TestFunction,
    Form,
)

from petsc4py.PETSc import ScalarType

from dolfinx.fem import Function, FunctionSpace, VectorFunctionSpace

from mpi4py import MPI
import petsc4py

# Aluminium replica
rho = 2700
g = 9.81
E = 69e09   # Young module
nu = 0.334  # poisson_ratio

mu = E / (2 * (1 + nu))
lmbda = E * nu / ((1 + nu) * (1 / 2 * nu))
# lmbda = 0.6
# mu = 0.4


def epsilon(u):
    """Return an expression for the deformation given a displacement field u"""
    return sym(grad(u))


def sigma(u: Function) -> Function:
    """Return an expression for the stress given a displacement field u"""
    return lmbda * tr(epsilon(u)) * Identity(len(u)) + 2 * mu * epsilon(u)


def elastic_strain_energy_func(u: Function) -> Function:
    """Return the energy psi given a displacement field u"""
    return lmbda / 2 * (tr(epsilon(u)) ** 2) + mu * tr(epsilon(u)) ** 2


def main(nelx: int, nely: int, volfrac: float, penal: float, rmin: float) -> None:

    # ---------------  Generate the mesh -----------------------------##
    msh = mesh.create_rectangle(
        comm=MPI.COMM_WORLD,
        points=((0.0, 0.0), (180, 90)),
        n=(nelx, nely),
    )

    # ---------------  Instanciate the function spaces  ----------------   ####

    D: FunctionSpace = FunctionSpace(msh, ("DG", 0))
    V: FunctionSpace = VectorFunctionSpace(msh, ("CG", 1))
    # or - element = ufl.VectorElement("CG", msh.ufl_cell(), 1)
    # or - V: FunctionSpace = FunctionSpace(msh, element)

    # The second argument on th above is the tuple containing the type of finite element
    # and the element degree. The type of element here is “CG”, which implies the
    # standard Lagrange family of elements

    u = TrialFunction(V)
    v = TestFunction(V)

    # ---------------  Boundary conditions  ---------------------------- ####

    tdim = msh.topology.dim # -> 2

    facets_clamped = mesh.locate_entities_boundary(
        mesh=msh, dim=tdim - 1, marker=lambda x: np.isclose(x[0], 0.0)
    )

    facet_sigma = mesh.locate_entities_boundary(
        mesh=msh, dim=tdim - 1, marker=lambda x: np.isclose(x[0], 180.0)
    )

    bc_clamped = fem.dirichletbc(
        value=np.array([0, 0], dtype=ScalarType),
        dofs=fem.locate_dofs_topological(V, entity_dim=1, entities=facets_clamped),
        V=V,
    )

    bc_sigma = fem.dirichletbc(
        value=np.array([0, 0], dtype=ScalarType),
        dofs=fem.locate_dofs_topological(V, entity_dim=1, entities=facet_sigma),
        V=V,
    )

    T = fem.Constant(msh, ScalarType((0, 0)))
    ds = Measure("ds", msh)

    f = fem.Constant(msh, ScalarType((0, -rho * g)))

    # ---------------  Create initial condition for the density
    density: Function = fem.Function(D)
    density.x.array[:] = volfrac

    a: Form = inner(density**penal * sigma(u), grad(v)) * dx
    L: Form = inner(f, v) * dx + inner(T, v) * ds

    # ---------------   Prepare distance matrices for filter -------------------
    num_cells_local = msh.topology.index_map(tdim).size_local
    midpoint_vector = mesh.compute_midpoints(
        msh, tdim, np.arange(num_cells_local, dtype=np.int32)
    )
    distance_mat = rmin - sp.euclidean_distances(midpoint_vector, midpoint_vector)
    distance_mat[distance_mat < 0] = 0
    distance_sum = distance_mat.sum(1)

    solver = fem.petsc.LinearProblem(a, L, bcs=[bc_clamped, bc_sigma])

    loop = 1
    objective = Function(D)
    elastic_strain_energy = Function(D)

    while loop < 200:
        # finite element analysis resolution -------------------------------------------
        uh: Function = solver.solve()

        # # OBJECTIVE FUNCTION AND SENSITIVITY ANALYSIS ------------------
        objective_expression = fem.Expression(
            ufl_expression=density**penal * elastic_strain_energy_func(uh),
            X=D.element.interpolation_points(),
        )
        objective.interpolate(objective_expression)

        strain_energy_expression = fem.Expression(
            ufl_expression=elastic_strain_energy_func(uh),
            X=D.element.interpolation_points(),
        )
        elastic_strain_energy.interpolate(strain_energy_expression)

        sensitivity: petsc4py.PETSc.Vec = (
            -penal
            * (density.vector[:]) ** (penal - 1)
            * elastic_strain_energy.vector[:]
        )

        # Filtering/modification of sensitivities ----------------------
        sensitivity = np.divide(
            np.matmul(distance_mat, np.multiply(density.vector[:], sensitivity)),
            np.multiply(density.vector[:], distance_sum),
        )

        # Design variable (density) update by the optimality criteria method 
        l1, l2 = 0, 100000
        move = 0.2

        looop = 1
        # fmt: off
        while l2 - l1 > 1e-4:
            looop += 1
            l_mid = 0.5 * (l2 + l1)
            density_new = \
                np.maximum(0.001, 
                           np.maximum(density.vector[:] - move,
                                      np.minimum(1.0, 
                                                 np.minimum(density.vector[:] + move,
                                                            density.vector[:] * np.sqrt(-sensitivity / l_mid)
                        )
                    )
                )
            )

            if sum(density_new) - volfrac * len(density.vector[:]) > 0: 
                l1 = l_mid
                l2 = l2
            else:
                l1 = l1
                l2 = l_mid

        # fmt: on
        density.vector[:] = density_new

        print(
            "iteration: ",
            loop,
            "objective: ",
            loop,
            sum(objective.vector[:]),
            "Volume fraction: {2:.3f}",
            sum(density.x.array[:]) / len(density.x.array[:]),
        )
        loop += 1

    ## ----------   Von Mises stress computation   -----------------  ##

    deviatoric_stress = sigma(uh) - 1 / 3 * tr(sigma(uh)) * Identity(len(uh))
    von_mises: Form = sqrt(3.0 / 2 * inner(deviatoric_stress, deviatoric_stress))
    V_von_mises = FunctionSpace(msh, ("DG", 0))

    stress_expr = fem.Expression(von_mises, V_von_mises.element.interpolation_points())
    stresses = Function(V_von_mises)
    stresses.interpolate(stress_expr)

    ## ------  IO for visualization with Paraview  --------------  ##
    # Read it with the xdmf3 reader on Paraview - the xdmf reader will not work
    with io.XDMFFile(msh.comm, "case.xdmf", "w") as xdmf:
        xdmf.write_mesh(msh)
        uh.name = "Deformation"
        xdmf.write_function(uh)
        stresses.name = "Von Mises stress"
        xdmf.write_function(stresses)
        density.name = "Density"
        xdmf.write_function(density)


if __name__ == "__main__":
    main(nelx=180, nely=60, volfrac=0.5, penal=3.0, rmin=2.0)

# Credit to Abhinav Guptaa, Rajib Chowdhurya, Anupam Chakrabartia, Timon Rabczukb
# for their work on A 55-line code for large-scale parallel topology optimization
# in 2D and 3D