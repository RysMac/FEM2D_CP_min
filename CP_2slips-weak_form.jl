# tasks:
    # 1. change it to 2D - done
    # 2. make it small strain anisotropic
    # 3. make it proper plastic problem
    # 4. 
    include("gqtpar(1).jl")

    using Ferrite, Tensors, ProgressMeter
    using BlockArrays, SparseArrays
    using BenchmarkTools
    using LinearAlgebra.BLAS
    using LinearAlgebra.LAPACK
    import LinearAlgebra: cross, Diagonal, UpperTriangular
    using .MoreSorensen
    using BenchmarkTools
    
    function CubicMatrix(c11::T, c12::T, c44::T) where T
        z = zero(T)
        return    [ Vec{}(c11, c12, c12, z, z, z),
                    Vec{}(c12, c11, c12, z, z, z), 
                    Vec{}(c12, c12, c11, z, z, z),
                    Vec{}(z, z, z, 2c44, z, z),
                    Vec{}(z, z, z, z, 2c44, z),
                    Vec{}(z, z, z, z, z, 2c44) ]
    end
    
    struct modules{T}
        E   ::Vector{Vec{6, T}}
        P   ::Vector{T}
        EP  ::Vector{T}
    end
    
    function modules(c11, c12, c44)
        #
        Dᵉ = CubicMatrix(c11, c12, c44)
        R1 = 1/sqrt(2.)
        R  = Tensor{2, 3, Float64}([R1 -R1 0.
                                   0. 0. -1.
                                   R1 R1 0.])
        m  = R ⋅ Vec{3,Float64}([1., -1., 1.]/sqrt(3))
        s  = R ⋅ Vec{3,Float64}([1., -1., -2.]/sqrt(6))
        #s = Vec{3, Int64}([1, 0, 0])
        #m = Vec{3, Int64}([0, 1, 0])
        P  = 1/2*(m⊗s + s⊗m)
        #Pvoigt = tovoigt(SymmetricTensor{2,3}(P); offdiagscale = sqrt(2))
        Pv   = vec([P[1, 1], P[2, 2], 0.0, sqrt(2)*P[1, 2], 0.0, 0.0])
        DᵉPv = vec([Dᵉ[i] ⋅ Pv for i in 1:6])
        #
        return modules(Dᵉ, Pv, DᵉPv)
    end
    
    
    
    function importTestGrid()
        grid = generate_grid( Quadrilateral, (20, 20), zero(Vec{2}), ones(Vec{2}));
        addfaceset!(grid, "myBottom", x -> norm(x[2]) ≈ 0.0);
        addfaceset!(grid, "myTop", x -> norm(x[2]) ≈ 1.0);
        addfaceset!(grid, "myRight", x -> norm(x[1]) ≈ 1.0);
        addfaceset!(grid, "myLeft", x -> norm(x[1]) ≈ 0.0);
        return grid
    end;
    
    function create_values(interpolation_u, interpolation_p)
        # quadrature rules
        qr      = QuadratureRule{2,RefCube}(2)
        face_qr = QuadratureRule{1,RefCube}(2)
    
        # geometric interpolation
        interpolation_geom = Lagrange{2,RefCube,1}()
    
        # cell and facevalues for u
        cellvalues_u = CellVectorValues(qr, interpolation_u, interpolation_geom)
        facevalues_u = FaceVectorValues(face_qr, interpolation_u, interpolation_geom)
    
        # cellvalues for p
        cellvalues_p = CellScalarValues(qr, interpolation_p, interpolation_geom)
    
        return cellvalues_u, cellvalues_p, facevalues_u
    end;
    
    function Ψ(ϵ, p, mp::modules)
        Dᵉ = mp.E
        Pv = mp.P
        DᵉPv = mp.EP
        gab = 0.1
        ϵv  = vec([ϵ[1, 1], ϵ[2, 2], 0.0, sqrt(2)/2*(ϵ[1, 2] + ϵ[2, 1]), 0.0, 0.0]) # czy tu mozna dac tylko e12 lub e21 ??
        Dᵉϵv = vec([Dᵉ[i] ⋅ ϵv for i in 1:6])
        #return 0.5 * ϵv ⋅ Dᵉϵv + 0.5 * (gab) * p^2
        return - (-0.01 + Dᵉϵv ⋅ Pv ) * p + 0.5 * ϵv ⋅ Dᵉϵv + 0.5 * (Pv ⋅ DᵉPv + gab) * p^2
    end;
    
    function constitutive_driver(ϵ, p, mp::modules)
        # Compute all derivatives in one function call
        ∂²Ψ∂F², ∂Ψ∂F = Tensors.hessian(y -> Ψ(y, p, mp), ϵ, :all)
        ∂²Ψ∂p², ∂Ψ∂p = Tensors.hessian(y -> Ψ(ϵ, y, mp), p, :all)
        ∂²Ψ∂F∂p = Tensors.gradient(q -> Tensors.gradient(y -> Ψ(y, q, mp), ϵ), p)
        return ∂Ψ∂F, ∂²Ψ∂F², ∂Ψ∂p, ∂²Ψ∂p², ∂²Ψ∂F∂p
    end;
    
    function create_dofhandler(grid, ipu, ipp)
        dh = DofHandler(grid)
        add!(dh, :u, 2, ipu) # displacement dim = 3
        add!(dh, :p, 1, ipp) # plastic slip dim = 1
        close!(dh)
        return dh
    end;
    
    function create_bc(dh)
        dbc = ConstraintHandler(dh)
        add!(dbc, Dirichlet(:u, getfaceset(dh.grid, "myBottom"), (x,t) -> zero(Vec{2}), [1 ,2]))
        add!(dbc, Dirichlet(:u, getfaceset(dh.grid, "myTop"), (x,t) -> [0, -t], [1 ,2]))
        #add!(dbc, Dirichlet(:u, getfaceset(dh.grid, "myLeft"), (x,t) -> zero(Vec{1}), [1]))
        #add!(dbc, Dirichlet(:u, getfaceset(dh.grid, "myBack"), (x,t) -> zero(Vec{1}), [3]))
        #add!(dbc, Dirichlet(:u, getfaceset(dh.grid, "myRight"), (x,t) -> t*ones(Vec{1}), [1]))
        close!(dbc)
        Ferrite.update!(dbc, 0.0)
        return dbc
    end;
    
    function calculate_element_energy(cell, cellvalues_u, cellvalues_p, mp, ue, pe)
        reinit!(cellvalues_u, cell)
        reinit!(cellvalues_p, cell)
        energy::Float64=0.0;
        # Dᵉ = mp.E
        # Pv = mp.P
        # DᵉPv = mp.EP
        for qp in 1:getnquadpoints(cellvalues_u)
            dΩ = getdetJdV(cellvalues_u, qp)
            ∇u = symmetric(function_gradient(cellvalues_u, qp, ue))
            p = function_value(cellvalues_p, qp, pe)
            energy += Ψ(∇u, p, mp) * dΩ
        end
        return energy
    end;
    
    function calculate_total_energy(w, dh::DofHandler, cellvalues_u, cellvalues_p, mp)
        total_energy::Float64 = 0.0;
        for cell in CellIterator(dh)
            global_dofs = celldofs(cell)
            nu = getnbasefunctions(cellvalues_u)
            global_dofsu = global_dofs[1:nu]; # first nu dofs are displacement
            global_dofsp = global_dofs[nu + 1:end]; # last np dofs are pressure
            #@assert size(global_dofs, 1) == nu + np # sanity check
            ue = w[global_dofsu] # displacement dofs for the current cell
            pe = w[global_dofsp] # pressure dofs for the current cell
            δenergy = calculate_element_energy(cell, cellvalues_u, cellvalues_p, mp, ue, pe)
            total_energy += δenergy;
        end
        return total_energy
    end;
    
    function assemble_element!(Ke, fe, cell, cellvalues_u, cellvalues_p, mp, ue, pe)
        # Dᵉ      = mp.E
        # Pv      = mp.P
        # DᵉPv    = mp.EP
        # Reinitialize cell values, and reset output arrays
        ublock, pblock = 1, 2
        reinit!(cellvalues_u, cell)
        reinit!(cellvalues_p, cell)
        fill!(Ke, 0.0)
        fill!(fe, 0.0)
    
        n_basefuncs_u = getnbasefunctions(cellvalues_u)
        n_basefuncs_p = getnbasefunctions(cellvalues_p)
    
        for qp in 1:getnquadpoints(cellvalues_u)
            dΩ = getdetJdV(cellvalues_u, qp)
            # Compute small strain tensor
            ∇u = symmetric(function_gradient(cellvalues_u, qp, ue))
            p = function_value(cellvalues_p, qp, pe)
    
            # Compute first Piola-Kirchhoff stress and tangent modulus
            ∂Ψ∂F, ∂²Ψ∂F², ∂Ψ∂p, ∂²Ψ∂p², ∂²Ψ∂F∂p = constitutive_driver(∇u, p, mp)
    
            # Loop over the `u`-test functions to calculate the `u`-`u` and `u`-`p` blocks
            for i in 1:n_basefuncs_u
                # gradient of the test function
                ∇δui = symmetric(shape_gradient(cellvalues_u, qp, i))
                # Add contribution to the residual from this test function
                fe[BlockIndex((ublock), (i))] += ( ∇δui ⊡ ∂Ψ∂F) * dΩ
    
                ∇δui∂S∂F = ∇δui ⊡ ∂²Ψ∂F²
                for j in 1:n_basefuncs_u
                    ∇δuj = symmetric(shape_gradient(cellvalues_u, qp, j))
    
                    # Add contribution to the tangent
                    Ke[BlockIndex((ublock, ublock), (i, j))] += ( ∇δui∂S∂F ⊡ ∇δuj ) * dΩ
                end
                # Loop over the `p`-test functions
                for j in 1:n_basefuncs_p
                    δp = shape_value(cellvalues_p, qp, j)
                    # Add contribution to the tangent
                    Ke[BlockIndex((ublock, pblock), (i, j))] += ( ∂²Ψ∂F∂p ⊡ ∇δui ) * δp * dΩ
                end
            end
            # Loop over the `p`-test functions to calculate the `p-`u` and `p`-`p` blocks
            for i in 1:n_basefuncs_p
                δp = shape_value(cellvalues_p, qp, i)
                fe[BlockIndex((pblock), (i))] += ( δp * ∂Ψ∂p) * dΩ
    
                for j in 1:n_basefuncs_u
                    ∇δuj = symmetric(shape_gradient(cellvalues_u, qp, j))
                    Ke[BlockIndex((pblock, ublock), (i, j))] += ∇δuj ⊡ ∂²Ψ∂F∂p * δp * dΩ
                end
                for j in 1:n_basefuncs_p
                    δp = shape_value(cellvalues_p, qp, j)
                    Ke[BlockIndex((pblock, pblock), (i, j))] += δp * ∂²Ψ∂p² * δp * dΩ
                end
            end
        end
    end;
    
    function assemble_global!(K::SparseMatrixCSC, f, cellvalues_u::CellVectorValues{dim},
                             cellvalues_p::CellScalarValues{dim}, dh::DofHandler, mp::modules, w) where {dim}
        nu = getnbasefunctions(cellvalues_u)
        np = getnbasefunctions(cellvalues_p)
    
        # start_assemble resets K and f
        fe = PseudoBlockArray(zeros(nu + np), [nu, np]) # local force vector
        ke = PseudoBlockArray(zeros(nu + np, nu + np), [nu, np], [nu, np]) # local stiffness matrix
    
        assembler = start_assemble(K, f)
        # Loop over all cells in the grid
        for cell in CellIterator(dh)
            global_dofs = celldofs(cell)
            global_dofsu = global_dofs[1:nu]; # first nu dofs are displacement
            global_dofsp = global_dofs[nu + 1:end]; # last np dofs are pressure
            @assert size(global_dofs, 1) == nu + np # sanity check
            ue = w[global_dofsu] # displacement dofs for the current cell
            pe = w[global_dofsp] # pressure dofs for the current cell
            assemble_element!(ke, fe, cell, cellvalues_u, cellvalues_p, mp, ue, pe)
            assemble!(assembler, global_dofs, fe, ke)
        end
    end;
    
    #function solve(interpolation_u, interpolation_p)
    
        # import the mesh
        grid = importTestGrid()
    
        # Material parameters
        c11 = 170.
        c12 = 124.
        c44 = 75.
        mp = modules(c11, c12, c44)
    
        # Create the DofHandler and CellValues
        interpolation_u = Lagrange{2, RefCube, 1}()
        interpolation_p = Lagrange{2, RefCube, 1}()
        dh = create_dofhandler(grid, interpolation_u, interpolation_p)
        cellvalues_u, cellvalues_p, facevalues_u = create_values(interpolation_u, interpolation_p)
    
        # Create the DirichletBCs
        dbc = create_bc(dh)
    
        # Pre-allocation of vectors for the solution and Newton increments
        _ndofs = ndofs(dh)
        w  = zeros(_ndofs)
        ΔΔw = zeros(_ndofs)
        apply!(w, dbc)
    
        # find plastic dofs:
        function pl_dofs_all()
            pl_dofs = Int64[]
            for cell in CellIterator(dh)
                global_dofs = celldofs(cell)
                append!(pl_dofs, global_dofs[dof_range(dh, :p)])
            end
            return collect(Set(pl_dofs))
        end
        
        pl_dofs_set = pl_dofs_all()
    
        # Create the sparse matrix and residual vector
        K = create_sparsity_pattern(dh)
        f = zeros(_ndofs)
    
        @benchmark assemble_global!(K, f, cellvalues_u, cellvalues_p, dh, mp, w)
    
        # We run the simulation parameterized by a time like parameter. `Tf` denotes the final value
        # of this parameter, and Δt denotes its increment in each step
        Tf = 0.02;
        Δt = 0.01;
        NEWTON_TOL = 1e-8
    
        pvd = paraview_collection("hyperelasticity_incomp_mixed.pvd");
        #for t ∈ 0.0:Δt:Tf
        t = 0.1
            # Perform Newton iterations
            Ferrite.update!(dbc, t)
            apply!(w, dbc)
            newton_itr = -1
            prog = ProgressMeter.ProgressThresh(NEWTON_TOL, "Solving @ time $t of $Tf;")
            fill!(ΔΔw, 0.0);
            
            f_energy(x) = calculate_total_energy(x, dh, cellvalues_u, cellvalues_p, mp)
    
            function gradhess(w)
                assemble_global!(K, f, cellvalues_u, cellvalues_p, dh, mp, w)
                apply_zero!(K, f, dbc)
                return K, f
            end        
            
            grad_error = 1.e-13
            pen_par = 1.2;
            inner_error = 1.e-8
            delta0 = 1.;
            la = zeros(length(w))
            
            @time results = trust_region(w, pl_dofs_set, la, pen_par, delta0, grad_error, inner_error);
    
            function func_penalty(x, λ, penpar, pl_dofs_set)
                penalties = [(λ[i] + penpar * x[i]) < 0. ? λ[i] * x[i] + 0.5 * penpar * x[i]^2 : -1. /( 2. * penpar ) * λ[i]^2 for i in pl_dofs_set]
                return sum(penalties)
            end
        
            # function grad_penalty(x, λ, penpar, pl_dofs_set)
            #     return [(λ[i] + penpar * x[i]) < 0. ? λ[i] + penpar * x[i] : 0. for i in pl_dofs_set]
            # end
        
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
    
            
    
            #while true; newton_itr += 1
                @time assemble_global!(K, f, cellvalues_u, cellvalues_p, dh, mp, w)
    
                norm_res = norm(f[Ferrite.free_dofs(dbc)])
                apply_zero!(K, f, dbc)
                # Only display output at specific load steps
                if t%(5*Δt) == 0
                    ProgressMeter.update!(prog, norm_res; showvalues = [(:iter, newton_itr)])
                end
                if norm_res < NEWTON_TOL
                    break
                elseif newton_itr > 30
                    error("Reached maximum Newton iterations, aborting")
                end
                # Compute the incremental `dof`-vector (both displacement and pressure)
                ΔΔw .= K\f;
    
                apply_zero!(ΔΔw, dbc)
                w .-= ΔΔw
            end;
    
            # Save the solution fields
            vtk_grid("hyperelasticity_incomp_mixed_$t.vtu", dh) do vtkfile
                vtk_point_data(vtkfile, dh, results)
                vtk_save(vtkfile)
                pvd[t] = vtkfile
            end
        end;
        vtk_save(pvd);
        vol_def = calculate_volume_deformed_mesh(w, dh, cellvalues_u);
        print("Deformed volume is $vol_def")
        return vol_def;
    end;
    
    
    quadratic = Lagrange{2, RefCube, 2}()
    linear = Lagrange{2, RefCube, 1}()
    vol_def = solve(quadratic, linear)
    
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
        
        hess, grad= gradhess(sol)
        grad_vals = grad .+ grad_penalty(sol, λ, penalty, pl_dofs_set)
        hess_vals = hess .+ hess_penalty(sol, λ, penalty, pl_dofs_set)
        func_vals = f_energy(sol) + func_penalty(sol, λ, penalty, pl_dofs_set)
        
        #println(grad_vals)
        #return hess_vals
        for i in 1:10000
            #println("iteration = ", i , "  delta = ", delta)
            #sol_inner = subproblem(hess_vals, grad_vals, delta)
            err = inner_error
            #in_matrix = UpperTriangular(Matrix(hess_vals))
            info, sol_inner, iter = gqtpar(Matrix(hess_vals), 'U', grad_vals, delta, err, err, 100, 0.0)[[1, 3, 5]]
            println("inner info = ", info, "  inner  iter = ", iter)
            
            sol_try = sol .+ sol_inner
            f_try = f_energy(sol_try) + func_penalty(sol_try, λ, penalty, pl_dofs_set)
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
                #println("accepted solution = ", sol)
    
                # for dof_i in pl_dofs_set
    
                #     λ_i = λ[dof_i]
                #     penalty_i = penalty_vec[dof_i]
                #     sol_i = sol[dof_i]
    
                if constraints_val(sol, pl_dofs_set) ≤ w1
    
                    penalty = penalty #penalty0 ?
                    λ = lgr_mult(λ, penalty, sol, pl_dofs_set)
                    w1 = w1 / (penalty^0.8)
                    w2 = w2 / penalty    
    
                else
    
                    penalty = penalty * 2
                    w1 = 1 / penalty^0.1
                    w2 = 1 / penalty
                end
                # end
    
                hess, grad = gradhess(sol)
                grad_vals = grad .+ grad_penalty(sol, λ, penalty, pl_dofs_set)
                hess_vals = hess .+ hess_penalty(sol, λ, penalty, pl_dofs_set)
                
                ispos = LAPACK.potrf!('U', Matrix(hess_vals))[2]
                println(i, " grad_vals = ", norm(grad_vals), "  delta = ", delta, "  positive def. = ", ispos)
                func_vals = f_energy(sol) + func_penalty(sol, λ, penalty, pl_dofs_set)
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