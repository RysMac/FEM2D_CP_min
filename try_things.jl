# TO DO LIST:
    # 1. reduce the problem to 2D small strain plane strain - done
    # 2. change the computational problem to 2D compression/shearing - done
    # 3. change BC so the final deformation is realized in few steps - done
    # 4. solve it by minimization not by solving set of linear equations - done
    # 5. extend it do plasticity with 1 than 2 than 6 slip systems
    # 6. try to do periodic boundary condition

using Ferrite, Tensors
using ForwardDiff
using TimerOutputs, ProgressMeter, IterativeSolvers
using ForwardDiff: Chunk, GradientConfig, HessianConfig
using SparseArrays
using Optim, LineSearches
using NLSolversBase

function extend_mx!(mx)
    mx = hcat(mx, [0.0; 0.0])
    mx = vcat(mx, [0.0 0.0 0.0])
    return mx
end;

function tens2vect(t::Matrix)
    return [t[1, 1], t[2, 2], t[3, 3], sqrt(2)*t[1, 2], sqrt(2)*t[1, 3], sqrt(2)*t[2, 3]]
end;

function vect2tens(v::Vector)
    return [
        v[1]  1 / sqrt(2) * v[4]  1 / sqrt(2) * v[5]
        1 / sqrt(2) * v[4] v[2] 1 / sqrt(2) * v[6]
        1 / sqrt(2) * v[5] 1 / sqrt(2) * v[6] v[3]
            ]
end;

struct HookeConst
    c11::Float64
    c12::Float64
    c44::Float64
    gab::Float64
end

function CubicMatrix(c11, c12, c44)
    return    [ c11 c12 c12 0 0 0
                c12 c11 c12 0 0 0
                c12 c12 c11 0 0 0
                0 0 0 2c44 0 0
                0 0 0 0 2c44 0
                0 0 0 0 0 2c44 ]
end;

function ψe(ϵ, p, mp::HookeConst) where T
    # parameters
    c11 = mp.c11
    c12 = mp.c12
    c44 = mp.c44
    gab = mp.gab
    Dᵉ  = CubicMatrix(c11, c12, c44)
    ϵ   = extend_mx!(ϵ) # this is quite dirty code
    ϵv  = tens2vect(ϵ)
    # simple fake energy having p = function_values(cvp, qp, pe)
    return 0.5(ϵv' * Dᵉ) * ϵv + 0.5 * gab * p^2
end;

function element_potential(dofe::AbstractVector{T}, cvu, cvp, mp::HookeConst) where T
    energy = zero(T)
    # muszą wchodzic dwa elementy cvu cvp dla u i pl
    nu = getnbasefunctions(cvu)
    np = getnbasefunctions(cvp)
    ue = dofe[1:nu]
    pe = dofe[nu + 1:end] 
    for qp=1:getnquadpoints(cvu)
        # tutaj trzeba rozdzielić ue i ple 
        ∇u      = function_symmetric_gradient(cvu, qp, ue)
        p       = function_value(cvp, qp, pe)
        energy  += ψe(∇u, p, mp) * getdetJdV(cvu, qp)
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

function ThreadCache(dpc::Int, nodespercell, cvPu::CellValues{DIM, T}, cvPp::CellValues{DIM, T}, modelparams, elpotential) where {DIM, T}
    element_indices     = zeros(Int, dpc)
    element_dofs        = zeros(dpc)
    element_gradient    = zeros(dpc)
    element_hessian     = zeros(dpc, dpc)
    element_coords      = zeros(Vec{DIM, T}, nodespercell)
    potfunc             = x -> elpotential(x, cvPu, cvPp, modelparams)
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
        #@timeit "gradient" ForwardDiff.gradient!(cache.element_gradient, cache.element_potential, eldofs, cache.gradconf)
        #@timeit "hessian" ForwardDiff.hessian!(cache.element_hessian, cache.element_potential, eldofs, cache.hessconf)
        #assemble!(assembler, cache.element_indices, cache.element_gradient, cache.element_hessian)
    end
    return total_energy
end;


function ElasticModel()
    # Generate a grid
    N       = 10
    L       = 1.0
    left    = zero(Vec{2})
    right   = L * ones(Vec{2})
    grid    = generate_grid(Quadrilateral, (N, N), left, right)

    # Material parameters
    c11 = 170.
    c12 = 124.
    c44 = 75.
    gab = 1000.
    mp  = HookeConst(c11, c12, c44, gab)

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
    
    
    _ndofs  = ndofs(dh)
    lc      = zeros(_ndofs)
    for i in 1:length(grid.cells)
        lc[celldofs(dh, i)[1:8]] .= -Inf
    end

    #pe_r = @view (pe, (3, :)) 
    dbcs = ConstraintHandler(dh)
    # Add a homogeneous boundary condition on the "clamped" edge
    dbc = Dirichlet(:u, getfaceset(grid, "right"), (x,t) -> [0.0, 0.0], [1, 2])
    add!(dbcs, dbc)
    dbc = Dirichlet(:u, getfaceset(grid, "left"), (x,t) -> t*[1], [1])
    add!(dbcs, dbc)
    close!(dbcs)

    # Make one cache 
    dpc     = ndofs_per_cell(dh)
    cpc     = length(grid.cells[1].nodes)
    caches  = ThreadCache(dpc, cpc, copy(cvu), copy(cvp), mp, element_potential)
      
    threadindices = [i for i in 1:length(grid.cells)]
    return Model(zeros(ndofs(dh)), dh, dbcs, threadindices, caches, lc)
end; 

function solve()
    reset_timer!()
    NEWTON_TOL      = 1e-10
    NEWTON_MAXITER  = 30
    Tf              = 0.5
    Δt              = 0.025
    counter         = 0 # step counter
    # Create model
    model   = ElasticModel()
    _ndofs  = length(model.dofs)
    u       = zeros(_ndofs) .+ 1e-2
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
    res(x) = optimize(od, odc, x, IPNewton(), Optim.Options(show_trace=true, show_every=1, g_tol=1e-16))
    pvd = paraview_collection("small_strain_elasticity_2D.pvd");
    #for t ∈ 0.0:Δt:Δt
        #Perform Newton iterations
        t = 0.1
        Ferrite.update!(model.boundaryconds, t)
        apply!(u, model.boundaryconds)
        model.dofs .= u
        newton_itr = -1
        prog = ProgressMeter.ProgressThresh(NEWTON_TOL, "Solving @ time $t of $Tf;")
        fill!(Δu, 0.0)
        counter += 1
        @timeit "minimization time" min = res(model.dofs) 
        #@timeit "minimization time" min = optimize(od, model.dofs, Newton(), Optim.Options(show_trace=false, show_every=1, g_tol=1e-20))
        println(min)
        model.dofs .= min.minimizer
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
            vtk_point_data(vtkfile, model.dofhandler, model.dofs)
            vtk_save(vtkfile)
            pvd[t] = vtkfile
        end
    #end

    print_timer(title = "Analysis with $(getncells(model.dofhandler.grid)) elements", linechars = :ascii)
    return u
end

u_my = solve();

u_my
# steps which can be done on this :
    # 1. Compute tangent matrix differentiating wrt global dofs - done
    # 2. Add scratch struct - done 
    # 3. Solve it by minimization - done
    # 3. Make it parallel
    # 4. Generate C code for tangent stiffness !!! this should help much !!!
    # 5. Go to plasticity...