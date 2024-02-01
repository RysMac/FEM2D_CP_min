# TO DO LIST:
    # 1. reduce the problem to 2D small strain plane strain - done
    # 2. change the computational problem to 2D compression/shearing - done
    # 3. change BC so the final deformation is realized in few steps - done
    # 4. solve it by minimization not by solving set of linear equations - done
    # 5. extend it do plasticity with 1 than 2 than 6 slip systems - 1 slip system done
    # 6. try to do periodic boundary condition

include("gqtpar(1).jl")

using Ferrite, Tensors
using ForwardDiff
using TimerOutputs, ProgressMeter, IterativeSolvers
using ForwardDiff: Chunk, GradientConfig, HessianConfig
using SparseArrays
using Optim, LineSearches
using LinearAlgebra.BLAS
using LinearAlgebra.LAPACK
import LinearAlgebra: cross, Diagonal, UpperTriangular
using .MoreSorensen    

struct HookeConst
    c11::Float64
    c12::Float64
    c44::Float64
    gab::Float64
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
    return - (-0.01 + Dᵉϵv ⋅ Pv) * p + 0.5 * ϵv ⋅ Dᵉϵv + 0.5 * (Pv ⋅ DᵉPv + gab * p) * p^2
    #return 0.5(ϵv' * Dᵉ) * ϵv + 0.5 * gab * p^2
end

#ψ(ϵ, p, Dᵉ, Pv, λ, penpar) = ψe(ϵ, p, Dᵉ, Pv) + penalty(p, λ, penpar)
    
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
    #grid    = generate_grid(QuadraticQuadrilateral, (N, N), left, right)
    grid    = generate_grid(Quadrilateral, (N, N), left, right)

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
    Pv =  vec([P[1, 1], P[2, 2], 0.0, P[1, 2], 0.0, 0.0])
    
    #@benchmark ψe(∇u, p, Dᵉ, Pv)

    _ndofs  = ndofs(dh)
    lc      = zeros(_ndofs)
    for i in 1:length(grid.cells)
        lc[celldofs(dh, i)[ue_range]] .= -Inf
    end

    #pe_r = @view (pe, (3, :)) 
    dbcs = ConstraintHandler(dh)
    # Add a homogeneous boundary condition on the "clamped" edge
    dbc = Dirichlet(:u, getfaceset(grid, "bottom"), (x,t) -> [0.0, 0.0], [1, 2])
    add!(dbcs, dbc)
    dbc = Dirichlet(:u, getfaceset(grid, "top"), (x,t) -> -t*[0, 1], [1, 2])
    add!(dbcs, dbc)
    close!(dbcs)

    # Make one cache 
    dpc     = ndofs_per_cell(dh)
    cpc     = length(grid.cells[1].nodes)
    caches  = ThreadCache(dpc, cpc, copy(cvu), copy(cvp), Dᵉ, Pv, element_potential)
        
    threadindices = [i for i in 1:length(grid.cells)]
    return Model(zeros(ndofs(dh)), dh, dbcs, threadindices, caches, lc)
end; 
    
#function solve()
    reset_timer!()
    NEWTON_TOL      = 1e-10
    NEWTON_MAXITER  = 30
    Tf              = 0.05
    Δt              = 0.0025
    counter         = 0 # step counter
    # Create model
    model   = ElasticModel();
    _ndofs  = length(model.dofs);
    u      = zeros(_ndofs);

    # u_range = dof_range(model.dofhandler, :u)
    # function set_initial_p!(x)
    #     for i in 1:length(model.dofhandler.grid.cells)
    #         x[celldofs(model.dofhandler, i)[length(u_range) + 1:end]] .= 1e-3
    #     end
    # end

    function pl_dofs_all()
        pl_dofs = Int64[]
        for cell in CellIterator(model.dofhandler)
            global_dofs = celldofs(cell)
            append!(pl_dofs, global_dofs[dof_range(model.dofhandler, :p)])
        end
        return collect(Set(pl_dofs))
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
        return storage
    end
    function hess!(storage, x)
        hessian_global!(storage, x, model)
        #∇²F!(storage, x, model)
        apply!(storage, model.boundaryconds)
        return storage
    end

    f(x) = energy_global(x, model)
    od = TwiceDifferentiable(f, grad!, hess!, model.dofs, 0.0, g, K);

    lx = model.constr_lower #[-Inf for _ in 1:_ndofs]
    ux = [Inf for _ in 1:_ndofs]
    #ux = [Inf for _ in 1:length(lx)]
    odc = TwiceDifferentiableConstraints(lx, ux);

    # Trust region minimization !!!!!!!

    pl_dofs_set = pl_dofs_all()
    grad_error = 1.e-14
    pen_par = 1.2;
    inner_error = 1.e-6
    delta0 = 1.;

    Ferrite.update!(model.boundaryconds, 0.1)
    apply!(u, model.boundaryconds)
    model.dofs .= u
    la = zeros(length(u))

    function func_penalty(x, λ, penpar, pl_dofs_set)
        penalties = [(λ[i] + penpar * x[i]) < 0. ? λ[i] * x[i] + 0.5 * penpar * x[i]^2 : -1. /( 2. * penpar ) * λ[i]^2 for i in pl_dofs_set]
        return sum(penalties)
    end

    function grad_penalty(x, λ, penpar, pl_dofs_set)
        return [(λ[i] + penpar * x[i]) < 0. ? λ[i] + penpar * x[i] : 0. for i in pl_dofs_set]
    end

    function grad_penalty(x::Vector{T}, λ::Vector{T}, penpar::T, pl_dofs_set::Vector{Int64}) where T
        g = zeros(T, length(x))
        for i in pl_dofs_set
            if λ[i] + penpar * x[i] < 0. 
                g[i] = λ[i] + penpar * x[i] 
            end
        end
        return g
    end

    function hess_penalty(x::Vector{T}, λ::Vector{T}, penpar::T, pl_dofs_set::Vector{Int64}) where T
        h = zeros(T, length(x))
        for i in pl_dofs_set
            if λ[i] + penpar * x[i] < 0. 
                h[i] = penpar 
            else
                h[i] = 0.
            end
        end
        return Diagonal(h)
    end
    #println(la)
    
    @time solution = trust_region(u, pl_dofs_set, la, pen_par, delta0, grad_error, inner_error);
    
    vtk_grid("CP_1slip_TR", model.dofhandler) do vtkfile
        vtk_point_data(vtkfile, model.dofhandler, solution) # displacement field
        #vtk_cell_data(vtkfile, mises_values, "von Mises [Pa]")
        #vtk_cell_data(vtkfile, κ_values, "Drag stress [Pa]")
    end

function trust_region(  x::Vector{T},
                        pl_dofs_set::Vector{Int64},
                        la::Vector{T},
                        penalty::T,
                        delta::T,
                        grad_error::T,
                        inner_error::T
                        ) where T <: Real

    sol = copy(x)
    sol_try = zeros(Float64, length(sol))
    λ = copy(la)
    w1 = 1 / (penalty^0.1)
    w2 = 1 / penalty
    

    grad_vals = grad!(g, sol) + grad_penalty(sol, λ, penalty, pl_dofs_set)
    hess_vals = hess!(K, sol) + hess_penalty(sol, λ, penalty, pl_dofs_set)
    func_vals = f(sol)        + func_penalty(sol, λ, penalty, pl_dofs_set)
    #println(hess_vals)
    #return hess_vals
    for i in 1:10000
        #println("iteration = ", i , "  delta = ", delta)
        #sol_inner = subproblem(hess_vals, grad_vals, delta)
        err = inner_error
        #in_matrix = UpperTriangular(Matrix(hess_vals))
        info, sol_inner, iter = gqtpar(Matrix(hess_vals), 'U', grad_vals, delta, err, err, 100, 0.0)[[1, 3, 5]]
        println("inner info = ", info, "  inner  iter = ", iter)
        
        sol_try = sol .+ sol_inner
        model.dofs .= sol_try
        f_try = f(sol_try) + func_penalty(sol_try, λ, penalty, pl_dofs_set)
        licznik = (f_try - func_vals)
        mianownik = grad_vals' * sol_inner + 0.5 * sol_inner' * hess_vals * sol_inner

        if abs(licznik) < 10^-16 && abs(licznik - mianownik) < 10^-16
            rho = 1
        else
            rho = licznik/mianownik 
        end

        #println("rho = ", rho, " licznik = ", licznik, " mianownik = ", mianownik)

        if rho > 0.25
            sol = sol_try
            model.dofs .= sol
            #println("accepted solution = ", sol)

            # for dof_i in pl_dofs_set

            #     λ_i = λ[dof_i]
            #     penalty_i = penalty_vec[dof_i]
            #     sol_i = sol[dof_i]

            if constraints_val(sol, pl_dofs_set) ≤ w1

                penalty = penalty #penalty0 ?
                λ = lgr_mult(λ, penalty, sol, pl_dofs_set)
                w1 = w1 / (penalty^0.4)
                w2 = w2 / penalty    

            else

                penalty = penalty * 2
                w1 = 1 / penalty^0.1
                w2 = 1 / penalty
            end
            # end


            grad_vals = grad!(g, sol) + grad_penalty(sol, λ, penalty, pl_dofs_set)
            hess_vals = hess!(K, sol) + hess_penalty(sol, λ, penalty, pl_dofs_set)
            
            ispos = LAPACK.potrf!('U', Matrix(hess_vals))[2]
            println(i, " grad_vals = ", norm(grad_vals), "  delta = ", delta, "  positive def. = ", ispos)
            func_vals = f(sol) + func_penalty(sol, λ, penalty, pl_dofs_set)
            #do_vtk(sol)
        end

        if rho ≤ 0.4
            delta = delta/2.
        end

        if rho > 0.9 && abs(delta - norm(sol_inner)) ≤ 10^-8 && delta < 2000.
            delta = 2. * delta
        end

        #println("solution = ", sol)
        if norm(grad_vals) < grad_error || i == 100
            println("gradient = ",norm(grad_vals), "  outer iterations = ", i, " delta = ", delta, " penalty = ", penalty)
            #println("solution = ", sol, "delta = ", delta)
        return sol
        end
    end
    return sol
end

function constraints_val(x, pl_dofs_set)
    # this can be written better
    result = 0.0
    total = 0.0
    for i in pl_dofs_set
        if x[i] <= 0.
            result = abs(x[i])
        else
            result = 0.
        end
        total += result
    end
    return total
end

function lgr_mult(λ, penalty, variables, pl_dofs_set)

    for i in pl_dofs_set
        if λ[i] + penalty * (variables[i]) ≤ 0.
            # moze tu minus powinien byc ?
            λ[i] += penalty * (variables[i])
        else
            λ[i] = 0.
        end
    end
    return λ
end