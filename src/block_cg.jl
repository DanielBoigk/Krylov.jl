# An implementation of the block conjugate gradient method for the solution of the Hermitian
# positive (semi)definite linear system AX = B, with an optional projection onto the
# complement of a known null space of A.
#
# The block iteration is the one of O'Leary (1980), in Galerkin form with search blocks that
# are orthonormalized by a rank-revealing eigendecomposition of their Gram matrix (SVQB), so
# that (nearly) linearly dependent right-hand sides and search directions are dropped instead
# of breaking down. Converged columns are removed from the block (deflation).
#
# For singular A with known null space V (e.g. the stiffness matrix of a pure Neumann problem,
# V = constants), every residual and every preconditioned residual is projected onto V⊥ =
# range(A), so the iteration never leaves V⊥; the component of the solution in V is fixed at
# the end by Wᴴx = 0.
#
# Daniel Boigk -- 2026.

export block_cg, block_cg!

"""
    (X, stats) = block_cg(A, B::AbstractMatrix{FC};
                          nullspace=nothing, grounding=nothing,
                          M=I, ldiv::Bool=false,
                          atol::T=√eps(T), rtol::T=√eps(T), itmax::Int=0,
                          rank_tol::T=√eps(T), recompute_every::Int=50,
                          timemax::Float64=Inf, verbose::Int=0, history::Bool=false,
                          callback=workspace->false, iostream::IO=kstdout)

`T` is an `AbstractFloat` such as `Float32`, `Float64` or `BigFloat`.
`FC` is `T` or `Complex{T}`.

    (X, stats) = block_cg(A, B, X0::AbstractMatrix; kwargs...)

Block-CG can be warm-started from an initial guess `X0` where `kwargs` are the same keyword arguments as above.

Solve the Hermitian positive definite linear system AX = B of size n with p right-hand sides using the block conjugate gradient method.

If `A` is only positive semidefinite, with a known null space spanned by the columns of `nullspace`, Block-CG solves the projected system `AX = ΠB`, where `Π` is the orthogonal projector onto the range of `A`, the orthogonal complement of the null space.
Every residual and every preconditioned residual is projected, so round-off and preconditioners that do not preserve the range cannot pollute the solution, and inconsistent right-hand sides are handled by solving the consistent projected system (`stats.inconsistent` is then set).
The component of `X` in the null space is fixed at the end by `Wᴴ X = 0`, where `W = grounding` (default: the null space itself, which gives the solution orthogonal to the null space).

Every search block is orthonormalized by a rank-revealing eigendecomposition of its Gram matrix, so dependent right-hand sides and directions are dropped instead of causing a breakdown.
Columns that have converged are removed from the block.
The residual of column `j` is compared with `atol + rtol * ‖Π bⱼ‖`, and the method stops when all columns are below their tolerance.

#### Interface

To easily switch between block Krylov methods, use the generic interface [`krylov_solve`](@ref) with `method = :block_cg`.

For an in-place variant that reuses memory across solves, see [`block_cg!`](@ref).

#### Input arguments

* `A`: a linear operator that models a Hermitian positive (semi)definite matrix of dimension `n`;
* `B`: a matrix of size `n × p`.

#### Optional argument

* `X0`: a matrix of size `n × p` that represents an initial guess of the solution `X`.

#### Keyword arguments

* `nullspace`: a basis of the null space of `A`, an `n × k` matrix or a vector (not necessarily orthonormal), or `nothing` for a positive definite `A`;
* `grounding`: an `n × k` matrix or a vector `W` that selects the solution `Wᴴ X = 0` among all solutions of the projected system, or `nothing` for `W = nullspace`;
* `M`: linear operator that models a Hermitian positive-definite matrix of size `n` used for centered preconditioning;
* `ldiv`: define whether the preconditioner uses `ldiv!` or `mul!`;
* `atol`: absolute stopping tolerance based on the residual norm of every column;
* `rtol`: relative stopping tolerance based on the residual norm of every column;
* `itmax`: the maximum number of iterations. If `itmax=0`, the default number of iterations is set to `2n` (deflation can reduce the block to a single column);
* `rank_tol`: relative eigenvalue threshold below which directions of a search block are considered linearly dependent and dropped;
* `recompute_every`: number of iterations after which the updated residual is replaced by the true residual `Π(B - AX)`, to limit the drift of the recurrence;
* `timemax`: the time limit in seconds;
* `verbose`: additional details can be displayed if verbose mode is enabled (verbose > 0). Information will be displayed every `verbose` iterations;
* `history`: collect additional statistics on the run such as residual norms;
* `callback`: function or functor called as `callback(workspace)` that returns `true` if the block-Krylov method should terminate, and `false` otherwise;
* `iostream`: stream to which output is logged.

#### Output arguments

* `X`: a dense matrix of size `n × p`;
* `stats`: statistics collected on the run in a [`SimpleStats`](@ref) structure.

#### Reference

* D. P. O'Leary, [*The block conjugate gradient algorithm and related methods*](https://doi.org/10.1016/0024-3795(80)90247-5), Linear Algebra and its Applications, 29, pp. 293--322, 1980.
"""
function block_cg end

"""
    workspace = block_cg!(workspace::BlockCgWorkspace, A, B; kwargs...)
    workspace = block_cg!(workspace::BlockCgWorkspace, A, B, X0; kwargs...)

In these calls, `kwargs` are keyword arguments of [`block_cg`](@ref), except `nullspace` and
`grounding`, which belong to the workspace.

See [`BlockCgWorkspace`](@ref) for instructions on how to create the `workspace`.

For a more generic interface, you can use [`krylov_workspace`](@ref) with `method = :block_cg` to allocate the workspace,
and [`krylov_solve!`](@ref) to run the block Krylov method in-place.
"""
function block_cg! end

"""
    kgram!(G, X, Y)

`G ← Xᴴ Y` for tall and skinny blocks `X` and `Y` (`n × r` and `n × s` with `n ≫ r, s`), the
Gram products of [`block_cg`](@ref). The generic method calls `mul!`; array backends can
specialise it where the default kernel is slow for such shapes.
"""
kgram!(G, X, Y) = mul!(G, X', Y)

# The first `r * c` entries of a column-major buffer, as an `r × c` matrix (no copy).
_bcg_block(X::AbstractMatrix, r::Integer, c::Integer) = size(X) == (r, c) ? X : reshape(view(vec(X), 1:(r * c)), r, c)

# a small block → a freshly allocated host matrix
_bcg_host(X::AbstractMatrix{FC}) where FC = copyto!(Matrix{FC}(undef, size(X)), X)

# X ← (I - V Vᴴ) X: orthogonal projection onto the range of A (V orthonormal)
function _bcg_project!(X, workspace)
  size(workspace.V, 2) == 0 && return X
  K = _bcg_block(workspace.K, size(workspace.K, 1), size(X, 2))
  kgram!(K, workspace.V, X)
  mul!(X, workspace.V, K, -one(eltype(X)), one(eltype(X)))
  return X
end

# X ← X - V (WᴴV)⁻¹ Wᴴ X: oblique projection onto {Wᴴx = 0} along the null space
function _bcg_ground!(X, workspace)
  size(workspace.V, 2) == 0 && return X
  K = _bcg_block(workspace.K, size(workspace.K, 1), size(X, 2))
  kgram!(K, workspace.W, X)
  mul!(X, workspace.F, K, -one(eltype(X)), one(eltype(X)))
  return X
end

# Euclidean column norms of X: one reduction on the device, p values to the host.
function _bcg_colnorms!(out::Vector{T}, X, workspace) where T
  sum!(abs2, workspace.colsums, X)
  copyto!(workspace.colsumsh, workspace.colsums)
  for j in eachindex(out)
    out[j] = sqrt(max(real(workspace.colsumsh[j]), zero(T)))
  end
  return out
end

# Replace the leading columns of P by an orthonormal basis of span(P) and return its dimension
# r (SVQB). The columns are scaled to unit norm first, so that only (near) linear dependence,
# not scale, decides which directions are dropped. Q is used as scratch.
function _bcg_orthonormalize!(workspace, rank_tol::T) where T
  P, Q, d = workspace.P, workspace.Q, workspace.d
  FC = eltype(P)
  p = size(P, 2)
  kgram!(workspace.G, P, P)
  Gh = _bcg_host(workspace.G)
  tiny = floatmin(T) / eps(T)
  for j = 1:p
    d[j] = real(Gh[j,j]) > tiny ? inv(sqrt(real(Gh[j,j]))) : zero(T)
  end
  for j = 1:p, i = 1:p
    Gh[i,j] *= d[i] * d[j]
  end
  E = eigen!(Hermitian(Gh))
  λmax = maximum(E.values; init = zero(T))
  λmax > 0 || return 0
  r = count(λ -> λ > rank_tol * λmax, E.values)
  r == 0 && return 0
  Th = zeros(FC, p, r)
  for (c, j) in enumerate(p:-1:p-r+1)  # the r largest eigenvalues
    s = inv(sqrt(E.values[j]))
    for i = 1:p
      Th[i,c] = d[i] * E.vectors[i,j] * s
    end
  end
  Tr = _bcg_block(workspace.C, p, r)
  copyto!(Tr, Th)
  Qr = _bcg_block(Q, size(Q, 1), r)
  mul!(Qr, P, Tr)
  copyto!(_bcg_block(P, size(P, 1), r), Qr)
  return r
end

# Solve the small Hermitian positive definite system G Y = C on the host (Cholesky, with an
# eigenvalue-based pseudo-inverse as fallback). The result overwrites C.
function _bcg_spd_solve!(G::Matrix{FC}, C::Matrix{FC}) where FC
  T = real(FC)
  F = cholesky!(Hermitian(copy(G)); check = false)
  if issuccess(F)
    ldiv!(F, C)
  else
    E = eigen(Hermitian(G))
    λmax = maximum(abs, E.values)
    λinv = [λ > sqrt(eps(T)) * λmax ? inv(λ) : zero(T) for λ in E.values]
    C .= E.vectors * (Diagonal(λinv) * (E.vectors' * C))
  end
  return C
end

# Z ← Π M⁻¹ R, with the columns of converged right-hand sides set to zero (deflation)
function _bcg_precondition!(workspace, M, MisI::Bool, ldiv::Bool)
  Z, R = workspace.Z, workspace.R
  if MisI
    copyto!(Z, R)
  else
    ldiv ? ldiv!(Z, M, R) : mul!(Z, M, R)
  end
  _bcg_project!(Z, workspace)
  if !all(workspace.active)
    FC = eltype(Z)
    for j in eachindex(workspace.active)
      workspace.maskh[1,j] = workspace.active[j] ? one(FC) : zero(FC)
    end
    copyto!(workspace.mask, workspace.maskh)
    Z .*= workspace.mask
  end
  return Z
end

def_args_block_cg = (:(A                    ),
                     :(B::AbstractMatrix{FC}))

def_optargs_block_cg = (:(X0::AbstractMatrix),)

def_kwargs_block_cg = (:(; M = I                        ),
                       :(; ldiv::Bool = false           ),
                       :(; atol::T = √eps(T)            ),
                       :(; rtol::T = √eps(T)            ),
                       :(; itmax::Int = 0               ),
                       :(; rank_tol::T = √eps(T)        ),
                       :(; recompute_every::Int = 50    ),
                       :(; timemax::Float64 = Inf       ),
                       :(; verbose::Int = 0             ),
                       :(; history::Bool = false        ),
                       :(; callback = workspace -> false),
                       :(; iostream::IO = kstdout       ))

def_kwargs_workspace_block_cg = (:(; nullspace = nothing),
                                 :(; grounding = nothing))

def_kwargs_block_cg = mapreduce(extract_parameters, vcat, def_kwargs_block_cg)
def_kwargs_workspace_block_cg = extract_parameters.(def_kwargs_workspace_block_cg)

args_block_cg = (:A, :B)
optargs_block_cg = (:X0,)
kwargs_block_cg = (:M, :ldiv, :atol, :rtol, :itmax, :rank_tol, :recompute_every, :timemax, :verbose, :history, :callback, :iostream)
kwargs_workspace_block_cg = (:nullspace, :grounding)

@eval begin
  function block_cg!(workspace :: BlockCgWorkspace{T,FC,SV,SM}, $(def_args_block_cg...); $(def_kwargs_block_cg...)) where {T <: AbstractFloat, FC <: FloatOrComplex{T}, SV <: AbstractVector{FC}, SM <: AbstractMatrix{FC}}

    # Timer
    start_time = time_ns()
    timemax_ns = 1e9 * timemax

    n, m = size(A)
    s, p = size(B)
    m == n || error("System must be square")
    n == s || error("Inconsistent problem size")
    p == workspace.p || error("The workspace was built for $(workspace.p) right-hand sides")
    (verbose > 0) && @printf(iostream, "BLOCK-CG: system of size %d with %d right-hand sides\n", n, p)

    # Check M = Iₙ
    MisI = (M === I)

    # Check type consistency
    eltype(A) == FC || @warn "eltype(A) ≠ $FC. This could lead to errors or additional allocations in operator-matrix products."
    ktypeof(B) == SM || error("ktypeof(B) must be equal to $SM")

    # Set up workspace.
    X, R, Q = workspace.X, workspace.R, workspace.Q
    bnorm, rnorm, tol, active = workspace.bnorm, workspace.rnorm, workspace.tol, workspace.active
    warm_start = workspace.warm_start
    stats = workspace.stats
    RNorms = stats.residuals
    reset!(stats)

    # Compatibility of the right-hand sides: ‖B‖² = ‖ΠB‖² + ‖(I - Π)B‖².
    _bcg_colnorms!(rnorm, B, workspace)
    nB² = sum(abs2, rnorm)
    copyto!(R, B)
    _bcg_project!(R, workspace)
    _bcg_colnorms!(bnorm, R, workspace)
    defect = nB² > 0 ? sqrt(max(nB² - sum(abs2, bnorm), zero(T)) / nB²) : zero(T)
    workspace.defect = defect
    for j = 1:p
      tol[j] = atol + rtol * bnorm[j]
    end

    # Initial guess X₀ ∈ range(A) and residual R₀ = Π(B - AX₀).
    if warm_start
      copyto!(X, workspace.ΔX)
      _bcg_project!(X, workspace)
      mul!(R, A, X)
      R .= B .- R
      _bcg_project!(R, workspace)
    else
      fill!(X, zero(FC))
    end
    _bcg_colnorms!(rnorm, R, workspace)
    for j = 1:p
      active[j] = rnorm[j] > tol[j]
    end
    RNorm = sqrt(sum(abs2, rnorm))  # ‖R₀‖_F
    history && push!(RNorms, RNorm)

    iter = 0
    itmax == 0 && (itmax = 2 * n)

    (verbose > 0) && @printf(iostream, "%5s  %7s  %5s  %5s\n", "k", "‖Rₖ‖", "rank", "timer")
    kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %5s  %.2fs\n", iter, RNorm, "-", start_time |> ktimer)

    # Stopping criterion
    status = "unknown"
    solved = !any(active)
    tired = iter ≥ itmax
    exhausted = false
    user_requested_exit = false
    overtimed = false

    if !solved
      _bcg_precondition!(workspace, M, MisI, ldiv)
      copyto!(workspace.P, workspace.Z)
    end

    while !(solved || tired || exhausted || user_requested_exit || overtimed)
      # Update iteration index.
      iter = iter + 1

      # Orthonormal basis of the search block.
      r = _bcg_orthonormalize!(workspace, rank_tol)
      if r == 0
        exhausted = true
        break
      end
      Pr = _bcg_block(workspace.P, n, r)
      Qr = _bcg_block(Q, n, r)
      mul!(Qr, A, Pr)  # Q ← AP

      # Gₖ = PᴴAP (r × r) and αₖ = Gₖ⁻¹ PᴴR (r × p), both solved on the host.
      Gd = _bcg_block(workspace.G, r, r)
      kgram!(Gd, Pr, Qr)
      Gh = _bcg_host(Gd)
      for j = 1:r, i = 1:j
        Gh[i,j] = (Gh[i,j] + conj(Gh[j,i])) / 2
        Gh[j,i] = conj(Gh[i,j])
      end
      Cd = _bcg_block(workspace.C, r, p)
      kgram!(Cd, Pr, R)
      Ch = _bcg_host(Cd)
      _bcg_spd_solve!(Gh, Ch)
      copyto!(Cd, Ch)

      # Xₖ = Xₖ₋₁ + Pαₖ and Rₖ = Rₖ₋₁ - APαₖ (or the true residual every `recompute_every` iterations).
      mul!(X, Pr, Cd, one(FC), one(FC))
      if iter % recompute_every == 0
        mul!(R, A, X)
        R .= B .- R
        _bcg_project!(R, workspace)
      else
        mul!(R, Qr, Cd, -one(FC), one(FC))
      end

      _bcg_colnorms!(rnorm, R, workspace)
      for j = 1:p
        active[j] = rnorm[j] > tol[j]
      end
      RNorm = sqrt(sum(abs2, rnorm))
      history && push!(RNorms, RNorm)

      # Update stopping criterion.
      user_requested_exit = callback(workspace) :: Bool
      solved = !any(active)
      tired = iter ≥ itmax
      timer = time_ns() - start_time
      overtimed = timer > timemax_ns
      kdisplay(iter, verbose) && @printf(iostream, "%5d  %7.1e  %5d  %.2fs\n", iter, RNorm, r, start_time |> ktimer)

      # Next search block: P ← Z + Pβₖ with βₖ = -Gₖ⁻¹ (AP)ᴴZ, A-conjugate to the current P.
      if !(solved || tired || user_requested_exit || overtimed)
        Z = _bcg_precondition!(workspace, M, MisI, ldiv)
        kgram!(Cd, Qr, Z)
        copyto!(Ch, Cd)
        _bcg_spd_solve!(Gh, Ch)
        Ch .*= -one(FC)
        copyto!(Cd, Ch)
        mul!(Z, Pr, Cd, one(FC), one(FC))
        @kswap!(workspace.P, workspace.Z)
      end
    end
    (verbose > 0) && @printf(iostream, "\n")

    # Fix the component in the null space: Wᴴ X = 0.
    _bcg_ground!(X, workspace)

    # Termination status
    tired               && (status = "maximum number of iterations exceeded")
    exhausted           && (status = "no search direction left (rank-deficient search block)")
    solved              && (status = "solution good enough given atol and rtol")
    overtimed           && (status = "time limit exceeded")
    user_requested_exit && (status = "user-requested exit")

    workspace.warm_start = false

    # Update stats
    stats.niter = iter
    stats.solved = solved
    stats.inconsistent = defect > √eps(T)
    stats.timer = start_time |> ktimer
    stats.status = status
    return workspace
  end
end
