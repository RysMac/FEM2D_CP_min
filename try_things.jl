# TO DO LIST:
    # 1. reduce the problem to 2D plane strain
    # 2. change the computational problem to 2D compression/shearing
    # 3. change BC so the final deformation is realized in few steps
    # 4. solve it by minimization not by solving set of linear equations
    # 5. extend it do plasticity with 1 than 2 than 6 slip systems
    # 6. try to do periodic boundary condition

using Ferrite, Tensors
using ForwardDiff
using TimerOutputs, ProgressMeter, IterativeSolvers
using ForwardDiff: Chunk, GradientConfig, HessianConfig
using SparseArrays

# parameters
struct NeoHooke
    μ::Float64
    λ::Float64
end

function ψe(C, mp::NeoHooke)
    # parameters
    μ   = mp.μ
    λ   = mp.λ
    # functions    
    Ic  = tr(C)
    J   = sqrt(det(C))
    return μ / 2 * (Ic -3) - μ * log(J) + λ / 2 * log(J)^2
end

function element_potential(ue::AbstractVector{T}, cv, mp::NeoHooke) where T
    energy = zero(T)
    for qp=1:getnquadpoints(cv)
        ∇u      = function_gradient(cv, qp, ue)
        F       = one(∇u) + ∇u
        C       = tdot(F) # F' ⋅ F
        energy  += ψe(C, mp)
    end
    return energy
end

# Create struct which poseses all element data e.g. ke, re, potential etc...
struct ThreadCache{CV, T, DIM, F <: Function, GC <: GradientConfig, HC <: HessianConfig}
    cvP                 ::CV
    element_indices     ::Vector{Int}
    element_dofs        ::Vector{T}
    element_gradient    ::Vector{T}
    element_hessian     ::Matrix{T}
    element_coords      ::Vector{Vec{DIM, T}}
    element_potential   ::F
    gradconf            ::GC
    hessconf            ::HC
end

function ThreadCache(dpc::Int, nodespercell, cvP::CellValues{DIM, T}, modelparams, elpotential) where {DIM, T}
    element_indices     = zeros(Int, dpc)
    element_dofs        = zeros(dpc)
    element_gradient    = zeros(dpc)
    element_hessian     = zeros(dpc, dpc)
    element_coords      = zeros(Vec{DIM, T}, nodespercell)
    potfunc             = x -> elpotential(x, cvP, modelparams)
    gradconf            = GradientConfig(potfunc, zeros(dpc), Chunk{3}())
    hessconf            = HessianConfig(potfunc, zeros(dpc), Chunk{3}())
    return ThreadCache(cvP, element_indices, element_dofs, element_gradient, element_hessian, element_coords, potfunc, gradconf, hessconf)
end


mutable struct Model{T, DH <: DofHandler, CH <: ConstraintHandler, TC <: ThreadCache}
    dofs          ::Vector{T}
    dofhandler    ::DH
    boundaryconds ::CH
    threadindices ::Vector{Int} # cells iterator
    threadcaches  ::TC  # cache with all element information
end

function assemble_global!(K::SparseMatrixCSC, r::Vector{T}, model::Model{T}) where T
    dh = model.dofhandler
    u = model.dofs
    # start_assemble resets K and r
    assembler = start_assemble(K, r)
    cache = model.threadcaches # only one cache - space for all element data like Ke, re etc...
    eldofs = cache.element_dofs
    # loop over all cells
    @timeit "assemble" for cell_i in model.threadindices
        nodeids = dh.grid.cells[cell_i].nodes
        #
        for j=1:length(cache.element_coords)
            cache.element_coords[j] = dh.grid.nodes[nodeids[j]].x
        end
        reinit!(cache.cvP, cache.element_coords)    
        celldofs!(cache.element_indices, dh, cell_i) # Store the degrees of freedom that belong to cell i in global_dofs
        #
        for j=1:length(cache.element_dofs)
            eldofs[j] = u[cache.element_indices[j]]
        end  
        @timeit "gradient" ForwardDiff.gradient!(cache.element_gradient, cache.element_potential, eldofs, cache.gradconf)
        @timeit "hessian" ForwardDiff.hessian!(cache.element_hessian, cache.element_potential, eldofs, cache.hessconf)
        assemble!(assembler, cache.element_indices, cache.element_gradient, cache.element_hessian)
    end
end;

function solve()
    reset_timer!()

    # Generate a grid
    N = 30
    L = 1.0
    left = zero(Vec{3})
    right = L * ones(Vec{3})
    grid = generate_grid(Tetrahedron, (N, N, N), left, right)

    # Material parameters
    E = 10.0
    ν = 0.3
    μ = E / (2(1 + ν))
    λ = (E * ν) / ((1 + ν) * (1 - 2ν))
    mp = NeoHooke(μ, λ)

    # Finite element base
    ip = Lagrange{3, RefTetrahedron, 1}()
    qr = QuadratureRule{3, RefTetrahedron}(1)
    cv = CellVectorValues(qr, ip)

    # DofHandler
    dh = DofHandler(grid)
    add!(dh, :u, 3) # Add a displacement field
    close!(dh)

    function rotation(X, t)
        θ = pi / 3 # 60°
        x, y, z = X
        return t * Vec{3}((
            0.0,
            L/2 - y + (y-L/2)*cos(θ) - (z-L/2)*sin(θ),
            L/2 - z + (y-L/2)*sin(θ) + (z-L/2)*cos(θ)
        ))
    end

    dbcs = ConstraintHandler(dh)
    # Add a homogeneous boundary condition on the "clamped" edge
    dbc = Dirichlet(:u, getfaceset(grid, "right"), (x,t) -> [0.0, 0.0, 0.0], [1, 2, 3])
    add!(dbcs, dbc)
    dbc = Dirichlet(:u, getfaceset(grid, "left"), (x,t) -> rotation(x, t), [1, 2, 3])
    add!(dbcs, dbc)
    close!(dbcs)
    t = 0.05
    Ferrite.update!(dbcs, t)

    # Pre-allocation of vectors for the solution and Newton increments
    _ndofs = ndofs(dh)
    un = zeros(_ndofs) # previous solution vector
    u  = zeros(_ndofs)
    Δu = zeros(_ndofs)
    ΔΔu = zeros(_ndofs)
    apply!(un, dbcs)

    # Create sparse matrix and residual vector
    K = create_sparsity_pattern(dh)
    g = zeros(_ndofs)

    # Make one cache 
    dpc = ndofs_per_cell(dh)
    cpc = length(grid.cells[1].nodes)
    caches = ThreadCache(dpc, cpc, copy(cv), mp, element_potential)
    
    threadindices = [i for i in 1:length(grid.cells)]
    model = Model(u, dh, dbcs, threadindices, caches) 
    
    newton_itr = -1
    NEWTON_TOL = 1e-10
    NEWTON_MAXITER = 30
    prog = ProgressMeter.ProgressThresh(NEWTON_TOL, "Solving:")

    while true; newton_itr += 1
        println("Newton iteration = ", newton_itr)
        # Construct the current guess
        u .= un .+ Δu
        model.dofs .= u
        # Compute residual and tangent for current guess
        assemble_global!(K, g, model)
        # Apply boundary conditions
        apply_zero!(K, g, dbcs)
        # Compute the residual norm and compare with tolerance
        normg = norm(g)
        println("normg = ", normg)
        ProgressMeter.update!(prog, normg; showvalues = [(:iter, newton_itr)])
        if normg < NEWTON_TOL
            break
        elseif newton_itr > NEWTON_MAXITER
            error("Reached maximum Newton iterations, aborting")
        end

        # Compute increment using conjugate gradients
        @timeit "linear solve" IterativeSolvers.cg!(ΔΔu, K, g; maxiter=1000)

        apply_zero!(ΔΔu, dbcs)
        Δu .-= ΔΔu
    end

    # Save the solution
    @timeit "export" begin
        vtk_grid("hyperelasticity_my", dh) do vtkfile
            vtk_point_data(vtkfile, dh, u)
        end
    end

    print_timer(title = "Analysis with $(getncells(grid)) elements", linechars = :ascii)
    return u
end

u_my = solve();


# steps which can be done on this :
    # 1. Compute tangent matrix differentiating wrt global dofs - done
    # 2. Add scratch struct - done 
    # 3. Solve it by minimization
    # 3. Make it parallel
    # 4. Generate C code for tangent stiffness 
    # 5. Go to plasticity...