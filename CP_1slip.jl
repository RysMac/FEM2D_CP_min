# TO DO LIST:
    # 1. reduce the problem to 2D small strain plane strain - done
    # 2. change the computational problem to 2D compression/shearing - done
    # 3. change BC so the final deformation is realized in few steps - done
    # 4. solve it by minimization not by solving set of linear equations - done
    # 5. extend it do plasticity with 1 than 2 than 6 slip systems - 1 slip system done
    # 6. try to do periodic boundary condition

using Ferrite, Tensors
using ForwardDiff
using TimerOutputs, ProgressMeter, IterativeSolvers
using ForwardDiff: Chunk, GradientConfig, HessianConfig
using SparseArrays
using Optim, LineSearches

mutable struct MaterialState{T, S <: Vector{T}}
    # store converged values
    accp    ::T
    f_yield ::T
    τc      ::T
    σ       ::S
    ϵ       ::S
end

function MaterialState(τ0::Float64)
    return MaterialState(
                0.0,
                -τ0, #f0 = -τ0 ???
                τ0,
                zeros(6),
                zeros(6)
                )
end

struct HookeConst
    c11::Float64
    c12::Float64
    c44::Float64
    gab::Float64
end

struct param_modules
    D # elastic tangent matrix
    P # vector with slip systems
end

function CubicMatrix(param::HookeConst)
    c11 = param.c11
    c12 = param.c12
    c44 = param.c44
    z = zero(Float64)
    return    [ Vec{}(c11, c12, c12, z, z, z),
                Vec{}(c12, c11, c12, z, z, z), 
                Vec{}(c12, c12, c11, z, z, z),
                Vec{}(z, z, z, 2c44, z, z),
                Vec{}(z, z, z, z, 2c44, z),
                Vec{}(z, z, z, z, z, 2c44) ]
end

# tutaj trzeba wszystko pozamieniać na Tensor lub Vector bo ϵ wchodzi jako SymmetricTensor więc musimy pracować na tych samych typach
function ψe(ϵ, p, Dᵉ, Pv)
    # parameters
    gab = 0.1
    # the energy is wrong 
    # wszystkie moduly trzeba stad wywalic do struc parameters
    ϵv  = vec([ϵ[1, 1], ϵ[2, 2], 0.0, sqrt(2)/2*(ϵ[1, 2] + ϵ[2, 1]), 0.0, 0.0])
    Dᵉϵv = vec([Dᵉ[i] ⋅ ϵv for i in 1:6])
    DᵉPv = vec([Dᵉ[i] ⋅ Pv for i in 1:6])
    return - (-0.01 + Dᵉϵv ⋅ Pv ) * p + 0.5 * ϵv ⋅ Dᵉϵv + 0.5 * (Pv ⋅ DᵉPv + gab * p) * p^2
    #return 0.5(ϵv' * Dᵉ) * ϵv + 0.5 * gab * p^2
end

function element_potential(dofe::AbstractVector{T}, cvu, cvp, Dᵉ, Pv) where T
    energy = zero(T)
    # muszą wchodzic dwa elementy cvu cvp dla u i pl
    nu = getnbasefunctions(cvu)
    np = getnbasefunctions(cvp)
    ue = dofe[1:nu]
    pe = dofe[nu + 1:end] 
    for qp=1:getnquadpoints(cvu)
        # tutaj trzeba rozdzielić ue i pe
        # tutaj trzeba dać ue-uen oraz pe-pen żeby mieć energię przysrostową!!! 
        ∇u      = function_gradient(cvu, qp, ue)
        p       = function_value(cvp, qp, pe)
        energy  += ψe(∇u, p, Dᵉ, Pv) * getdetJdV(cvu, qp)
    end
    return energy
end;

# Create struct which poseses all element data e.g. ke, re, potential etc...
struct ThreadCache{CVu, CVp, T, DIM, F <: Function, GC <: GradientConfig, HC <: HessianConfig}
    cvPu                ::CVu
    cvPp                ::CVp
    element_indices     ::Vector{Int}
    element_dofs        ::Vector{T}
    element_gradient    ::Vector{T}
    element_hessian     ::Matrix{T}
    element_coords      ::Vector{Vec{DIM, T}}
    element_potential   ::F
    gradconf            ::GC
    hessconf            ::HC
end

function ThreadCache(dpc::Int, nodespercell, cvPu::CellValues{DIM, T}, cvPp::CellValues{DIM, T}, Dᵉ, Pv, elpotential) where {DIM, T}
    element_indices     = zeros(Int, dpc)
    element_dofs        = zeros(dpc)
    element_gradient    = zeros(dpc)
    element_hessian     = zeros(dpc, dpc)
    element_coords      = zeros(Vec{DIM, T}, nodespercell)
    potfunc             = x -> elpotential(x, cvPu, cvPp, Dᵉ, Pv)
    gradconf            = GradientConfig(potfunc, zeros(dpc), Chunk{3}())
    hessconf            = HessianConfig(potfunc, zeros(dpc), Chunk{3}())
    return ThreadCache(cvPu, cvPp, element_indices, element_dofs, element_gradient, element_hessian, element_coords, potfunc, gradconf, hessconf)
end


mutable struct Model{T, DH <: DofHandler, CH <: ConstraintHandler, TC <: ThreadCache}
    dofs          ::Vector{T}
    dofhandler    ::DH
    boundaryconds ::CH
    threadindices ::Vector{Int} # cells iterator now it is 1:ncells
    threadcaches  ::TC  # cache with all element information
constr_lower  ::Vector{T}
end

function hessian_global!(K::SparseMatrixCSC, dofvector::Vector{T}, model::Model{T}) where T
    dh = model.dofhandler
    # start_assemble resets K and r
    assembler = start_assemble(K)
    cache = model.threadcaches # only one cache - space for all element data like Ke, re etc...
    eldofs = cache.element_dofs
    # loop over all cells
    @timeit "assemble hessian" for cell_i in model.threadindices
        nodeids = dh.grid.cells[cell_i].nodes
        #
        for j=1:length(cache.element_coords)
            cache.element_coords[j] = dh.grid.nodes[nodeids[j]].x
        end
        reinit!(cache.cvPu, cache.element_coords)    
        reinit!(cache.cvPp, cache.element_coords)    
        celldofs!(cache.element_indices, dh, cell_i) # Store the degrees of freedom that belong to cell i in global_dofs
        #
        for j=1:length(cache.element_dofs)
            eldofs[j] = dofvector[cache.element_indices[j]]
        end  
        #@timeit "gradient" ForwardDiff.gradient!(cache.element_gradient, cache.element_potential, eldofs, cache.gradconf)
        ForwardDiff.hessian!(cache.element_hessian, cache.element_potential, eldofs, cache.hessconf)
        assemble!(assembler, cache.element_indices, cache.element_hessian)
    end
end;

function gradient_global!(r::Vector{T}, dofvector::Vector{T}, model::Model{T}) where T
    fill!(r, 0.0)
    dh = model.dofhandler
    # start_assemble resets K and r
    # assembler = start_assemble(K, r)
    cache = model.threadcaches # only one cache - space for all element data like Ke, re etc...
    eldofs = cache.element_dofs
    # loop over all cells
    @timeit "assemble gradient" for cell_i in model.threadindices
        nodeids = dh.grid.cells[cell_i].nodes
        #
        for j=1:length(cache.element_coords)
            cache.element_coords[j] = dh.grid.nodes[nodeids[j]].x
        end
        reinit!(cache.cvPu, cache.element_coords)    
        reinit!(cache.cvPp, cache.element_coords)    
        celldofs!(cache.element_indices, dh, cell_i) # Store the degrees of freedom that belong to cell i in global_dofs
        #
        for j=1:length(cache.element_dofs)
            eldofs[j] = dofvector[cache.element_indices[j]]
        end  
        ForwardDiff.gradient!(cache.element_gradient, cache.element_potential, eldofs, cache.gradconf)
        #@timeit "hessian" ForwardDiff.hessian!(cache.element_hessian, cache.element_potential, eldofs, cache.hessconf)
        assemble!(r, cache.element_indices, cache.element_gradient)
    end
end;

function energy_global(dofvector::Vector{T}, model::Model{T}) where T
    dh = model.dofhandler
    # start_assemble resets K and r
    # assembler = start_assemble(K, r)
    cache = model.threadcaches # only one cache - space for all element data like Ke, re etc...
    eldofs = cache.element_dofs
    total_energy = 0.0
    # loop over all cells
    @timeit "assemble energy" for cell_i in model.threadindices
        nodeids = dh.grid.cells[cell_i].nodes # indexy nodow / numeracja nodow
        #
        for j=1:length(cache.element_coords)
            cache.element_coords[j] = dh.grid.nodes[nodeids[j]].x # zapisanie w tablicy coordinates of nodes w celu identyfikacji elementu
        end
        reinit!(cache.cvPu, cache.element_coords) 
        reinit!(cache.cvPp, cache.element_coords) # bedzie chyba klopot przy uzyciu innych elementow dla dwoch pol      
        celldofs!(cache.element_indices, dh, cell_i) # Store the degrees of freedom that belong to cell i in global_dofs i.e. dofvector array
        #
        for j=1:length(cache.element_dofs)
            eldofs[j] = dofvector[cache.element_indices[j]]
        end  
        total_energy += cache.element_potential(eldofs)
    end
    return total_energy
end;


function ElasticModel()
    # Generate a grid
    N       = 10
    L       = 1.0
    left    = zero(Vec{2})
    right   = L * ones(Vec{2})
    #grid    = generate_grid(QuadraticQuadrilateral, (N, N), left, right)
    grid    = generate_grid(Quadrilateral, (N, N), left, right)

    function importTestGrid()
        addfaceset!(grid, "myBottom", x -> norm(x[2]) ≈ 0.0);
        addfaceset!(grid, "myUp", x -> norm(x[2]) ≈ L);
        addvertexset!(grid, "cornerdown", (x) -> x[1] ≈ L/2 && x[2] ≈ 0.0)
        addvertexset!(grid, "cornerup", (x) -> x[1] ≈ L/2 && x[2] ≈ L)
    end;

    # Material parameters
    c11 = 170.
    c12 = 124.
    c44 = 75.
    gab = 0.
    mp  = HookeConst(c11, c12, c44, gab)
    Dᵉ  = CubicMatrix(mp)
    # Finite element base
    #linear    = Lagrange{2,RefTetrahedron,1}()
    #quadratic = Lagrange{2,RefTetrahedron,2}()
    ipu = Lagrange{2, RefCube, 1}() # field interpolation fo u
    ipp = Lagrange{2, RefCube, 1}() # field interpolation fo p
    gip = Lagrange{2, RefCube, 1}() # geometric interpolation
    qr  = QuadratureRule{2, RefCube}(2) # Gauss points
    cvu = CellVectorValues(qr, ipu, gip)
    cvp = CellScalarValues(qr, ipp, gip)

    # DofHandler
    dh = DofHandler(grid)
    add!(dh, :u, 2, ipu) # add displacement field
    add!(dh, :p, 1, ipp) # add plastic field
    close!(dh)

    ue_range = dof_range(dh, :u)
    pe_range = dof_range(dh, :p)


    # nodeids = grid.cells[1].nodes
    #     #
    # element_coords = zeros(Vec{2, Float64}, 4)
    # for j=1:length(element_coords)
    #     element_coords[j] = grid.nodes[nodeids[j]].x
    # end
    # reinit!(cvu, element_coords)    
    # reinit!(cvp, element_coords)    
    
    # ue = zeros(8)
    # pe = zeros(4)
    # ∇u = function_symmetric_gradient(cvu, 1, ue)
    # p  = function_value(cvp, 1, pe)


    # @code_warntype ψe(∇u, p, Dᵉ, Pv)
    
    # using BenchmarkTools
    R1 = 1/sqrt(2.)
    R = Tensor{2, 3, Float64}([R1 -R1 0.
                               0. 0. -1.
                               R1 R1 0.])
    m = R ⋅ Vec{3,Float64}([1., -1., 1.]/sqrt(3.))
    s = R ⋅ Vec{3,Float64}([1., -1., -2.]/sqrt(6))
    P = 0.5(m⊗s + s⊗m)
    #Pvoigt = tovoigt(SymmetricTensor{2,3}(P); offdiagscale = sqrt(2))
    Pv =  vec([P[1, 1], P[2, 2], 0.0, sqrt(2)*P[1, 2], 0.0, 0.0])

    #@benchmark ψe(∇u, p, Dᵉ, Pv)

    _ndofs  = ndofs(dh)
    lc      = zeros(_ndofs)
    for i in 1:length(grid.cells)
        lc[celldofs(dh, i)[ue_range]] .= -Inf
    end

    dbcs = ConstraintHandler(dh)
    addvertexset!(grid, "cornerdown", (x) -> x[1] ≈ 0.0 && x[2] ≈ 0.0)
    addvertexset!(grid, "cornertop", (x) -> x[1] ≈ 0.0 && x[2] ≈ 1.0)
    # Add a homogeneous boundary condition on the "clamped" edge
    dbc = Dirichlet(:u, getfaceset(grid, "bottom"), (x,t) -> [0.0], [2])
    add!(dbcs, dbc)
    dbc = Dirichlet(:u, getfaceset(grid, "top"), (x,t) -> -t*[1.], [2])
    add!(dbcs, dbc)
    dbc = Dirichlet(:u, getfaceset(grid, "left"), (x,t) -> [0.0], [1])
    add!(dbcs, dbc)
    #add!(dbcs, Dirichlet(:u, getvertexset(grid, "cornerdown"), (x,t) -> [0], [1]))
    #add!(dbcs, Dirichlet(:u, getvertexset(grid, "cornertop"), (x,t) -> [0], [1]))
    close!(dbcs)

    # Make one cache 
    dpc     = ndofs_per_cell(dh)
    cpc     = length(grid.cells[1].nodes)
    caches  = ThreadCache(dpc, cpc, copy(cvu), copy(cvp), Dᵉ, Pv, element_potential)

    threadindices = [i for i in 1:length(grid.cells)]
    return Model(zeros(ndofs(dh)), dh, dbcs, threadindices, caches, lc), param_modules(Dᵉ, Pv)
end; 

function postprocessing(dofvector::Vector{T}, model::Model{T}, states, modules::param_modules ) where T
    Dᵉ = modules.D
    Pv = modules.P
    dh = model.dofhandler
    cache = model.threadcaches # only one cache - space for all element data like Ke, re etc...
    eldofs = cache.element_dofs
    # loop over all cells
    @timeit "assemble energy" for cell_i in model.threadindices
        nodeids = dh.grid.cells[cell_i].nodes # indexy nodow / numeracja nodow
        #
        for j=1:length(cache.element_coords)
            cache.element_coords[j] = dh.grid.nodes[nodeids[j]].x # zapisanie w tablicy coordinates of nodes w celu identyfikacji elementu
        end
        reinit!(cache.cvPu, cache.element_coords) 
        reinit!(cache.cvPp, cache.element_coords) # bedzie chyba klopot przy uzyciu innych elementow dla dwoch pol      
        celldofs!(cache.element_indices, dh, cell_i) # Store the degrees of freedom that belong to cell i in global_dofs i.e. dofvector array
        #
        for j=1:length(cache.element_dofs)
            eldofs[j] = dofvector[cache.element_indices[j]]
        end  
        cell_state = states[:, cell_i]
        element_values(eldofs, cache.cvPu, cache.cvPp, Dᵉ, Pv, cell_state)
    end
end;

function element_values(dofe::AbstractVector{T}, cvu, cvp, Dᵉ, Pv, cell_state) where T
    # muszą wchodzic dwa elementy cvu cvp dla u i pl
    nu = getnbasefunctions(cvu)
    np = getnbasefunctions(cvp)
    ue = dofe[1:nu]
    pe = dofe[nu + 1:end] 
    for qp=1:getnquadpoints(cvu)
        pn       = cell_state[qp].accp
        fn_yield = cell_state[qp].f_yield
        τcn      = cell_state[qp].τc
        σn       = cell_state[qp].σ
        ϵn       = cell_state[qp].ϵ
        # 
        Δ∇u = function_gradient(cvu, qp, ue)
        Δp  = function_value(cvp, qp, pe)
        # calculate/export ϵ, accp, s, f_yield...
        Δϵv     = vec([Δ∇u[1, 1], Δ∇u[2, 2], 0.0, sqrt(2)/2*(Δ∇u[1, 2] + Δ∇u[2, 1]), 0.0, 0.0])
        Δσ      = [Dᵉ[i] ⋅ Δϵv for i in 1:6]
        σ       = σn .+ Δσ
        Δτc     = 0.1 * Δp 
        τc      = τcn + Δτc
        f_yield = σ ⋅ Pv - τc 
        #
        cell_state[qp].accp     = pn + Δp
        cell_state[qp].f_yield  = f_yield
        cell_state[qp].τc       = τc
        cell_state[qp].σ        = σ
        cell_state[qp].ϵ        = Δϵv
    end
end;

#function solve()
#    reset_timer!()
    NEWTON_TOL      = 1e-10
    NEWTON_MAXITER  = 30
    Tf              = 0.05
    Δt              = 0.0025
    counter         = 0 # step counter
    
    model, modules   = ElasticModel()
    
    # Create states history
    nqp = getnquadpoints(model.threadcaches.cvPu)
    τ0 = 0.01
    states = [MaterialState(τ0) for _ in 1:nqp, _ in 1:getncells(model.dofhandler.grid)]
    #states_old = [MaterialState(τ0) for _ in 1:nqp, _ in 1:getncells(model.dofhandler.grid)]
    
    _ndofs  = length(model.dofs)
    u      = zeros(_ndofs)

    u_range = dof_range(model.dofhandler, :u)
    function set_initial_p!(x)
        for i in 1:length(model.dofhandler.grid.cells)
            x[celldofs(model.dofhandler, i)[length(u_range) + 1:end]] .= 1e-30
        end
    end
    #model.dofs .= u
    u       = zeros(_ndofs)
    un      = zeros(_ndofs)
    Δu      = zeros(_ndofs)
    # Create sparse matrix and residual vector
    K = create_sparsity_pattern(model.dofhandler)
    g = zeros(_ndofs)
      
    function grad!(storage, x)
        gradient_global!(storage, x, model)
        #∇F!(storage, x, model)
        apply_zero!(storage, model.boundaryconds)
    end
    function hess!(storage, x)
        hessian_global!(storage, x, model)
        #∇²F!(storage, x, model)
        apply!(storage, model.boundaryconds)
    end

    f(x) = energy_global(x, model)
    
    od = TwiceDifferentiable(f, grad!, hess!, model.dofs, 0.0, g, K)

    lx = model.constr_lower #[-Inf for _ in 1:_ndofs]
    ux = [Inf for _ in 1:_ndofs]
    #ux = [Inf for _ in 1:length(lx)]
    odc = TwiceDifferentiableConstraints(lx, ux)
    #res(x) = optimize(od, odc, x, Fminbox(GradientDescent()), Optim.Options(show_trace=false, show_every=1, g_tol=1e-16, iterations=50))
    res(x) = optimize(f, grad!, lx, ux, x, Fminbox(LBFGS()), Optim.Options(show_trace=false, show_every=1, g_tol=1e-16, iterations=50))
    # res(x) = optimize(od, odc, x, IPNewton( linesearch = Optim.backtrack_constrained_grad,
    #                                                        #μ0 = 10,
    #                                                        show_linesearch = false),
    #                                        Optim.Options(allow_f_increases = true, successive_f_tol = 2)
    #                                                        )
    pvd = paraview_collection("small_strain_elasticity_2D.pvd");
    #for t ∈ 0.0:Δt:Tf
                #Perform Newton iterations
        t = 0.001
        Ferrite.update!(model.boundaryconds, t)
        apply!(u, model.boundaryconds)
        Δu .= u - un
        set_initial_p!(Δu)
        model.dofs .= Δu
        newton_itr = -1
        prog = ProgressMeter.ProgressThresh(NEWTON_TOL, "Solving @ time $t of $Tf;")
        #fill!(Δu, 0.0)
        counter += 1
        @timeit "minimization time" min = res(model.dofs) 
        #@timeit "minimization time" min = optimize(od, model.dofs, Newton(), Optim.Options(show_trace=false, show_every=1, g_tol=1e-20))
        model.dofs .= min.minimizer
        println(min)
        un .+= model.dofs
        u .= un

        states = [MaterialState(τ0) for _ in 1:nqp, _ in 1:getncells(model.dofhandler.grid)]

        postprocessing(model.dofs, model, states, modules )

        [(states[:,i])[i].f_yield for i in 1:4]
        
        [(states[:,1])[i].τc for i in 1:4]
        eps = [(states[:,i])[i].ϵ for i in 1:4]
        # evaluate current states of f_yield, s, e at each gauss point

        s = [modules.D[i] ⋅ eps[1] for i in 1:6]
        
        s ⋅ modules.P
        1
        # while true; newton_itr += 1
        #     println("Newton iteration = ", newton_itr)
        #     # compute tangent, residuals and energy
        #     grad!(g, model.dofs)
        #     hess!(K, model.dofs)
        #     total_energy = f(model.dofs)
        #     println("total energy = ", total_energy)
        #     # Apply boundary conditions
        #     # apply_zero!(K, g, model.boundaryconds)
        #     # Compute the residual norm and compare with tolerance
        #     normg = norm(g)
        #     println("for t = ", t, "  normg = ", normg)
        #     ProgressMeter.update!(prog, normg; showvalues = [(:iter, newton_itr)])
        #     if normg < NEWTON_TOL
        #         break
        #     elseif newton_itr > NEWTON_MAXITER
        #         error("Reached maximum Newton iterations, aborting")
        #     end

        #     # Compute increment using conjugate gradients
        #     @timeit "linear solve" IterativeSolvers.cg!(Δu, K, g; maxiter=1000)

        #     apply_zero!(Δu, model.boundaryconds)
        #     u .-= Δu
        #     fill!(Δu, 0.0)
        #     model.dofs .= u
        #     println("Is TRUE ? = ", norm(u .- res.minimizer))
        # end
        # Save the solution fields
        vtk_grid("small_strain_elasticity_2D_$counter.vtu", model.dofhandler) do vtkfile
            vtk_point_data(vtkfile, model.dofhandler, u)
            vtk_save(vtkfile)
            pvd[t] = vtkfile
        end
    #end

    print_timer(title = "Analysis with $(getncells(model.dofhandler.grid)) elements", linechars = :ascii)
    return u, model.dofs
end

u_my, dofs = solve();


maximum(u_my)
maximum(dofs)

# steps which can be done on this :
    # 1. Compute tangent matrix differentiating wrt global dofs - done
    # 2. Add scratch struct - done 
    # 3. Solve it by minimization - done
    # 3. Make it parallel
    # 4. Generate C code for tangent stiffness !!! this should help much !!! 
    # 5. Go to plasticity...

    range = 1:8

    length(range)
    maximum(range)