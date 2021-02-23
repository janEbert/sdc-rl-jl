import LinearAlgebra
import Random

include("coll_gauss_radau_right.jl")

const NORM_P = Inf

abstract type AbstractPreconditioner end

struct RLPreconditioner <: AbstractPreconditioner end
struct LUPreconditioner <: AbstractPreconditioner end
struct MINPreconditioner <: AbstractPreconditioner end

function get_preconditioner(env, action, ::RLPreconditioner)
    LinearAlgebra.Diagonal(action)
end

function get_preconditioner(env, action, ::LUPreconditioner)
    Q_T = transpose(env.Q)
    factorization = LinearAlgebra.lu(Q_T)
    U = factoriation.U
    transpose(U)
end

function get_preconditioner(env, action, ::MINPreconditioner)
    if M == 5
        x = [
            0.2818591930905709,
            0.2011358490453793,
            0.06274536689514164,
            0.11790265267514095,
            0.1571629578515223,
        ]
    elseif M == 3
        x = [
            0.3203856825077055,
            0.1399680686269595,
            0.3716708461097372,
        ]
    else
        # if M is some other number, take zeros. This won't work
        # well, but does not raise an error
        return zeros(M, M)
    end
    LinearAlgebra.Diagonal(x)
end

function norm(x)
    LinearAlgebra.norm(x, NORM_P)
end

mutable struct SDCEnv{L <: Number}
    # Constant

    M::Int
    dt::Float64
    restol::Float64
    max_sequence_length::Int
    max_episode_length::Int
    num_steps::Int

    num_nodes::Int
    Q::Matrix{Float64}
    # Complex{Float64} is the same as NumPy's `complex128`
    initial_u::Vector{L}

    # Mutable

    preconditioner::AbstractPreconditioner

    lambda::L
    C::Matrix{L}
    initial_residual::Vector{L}

    last_residual_norm::Float64

    function SDCEnv{L}(
        M,
        dt,
        restol,
        max_sequence_length=8,
        max_episode_length=50,
        lambda_real_interval=(-100, 0),
        lambda_imag_interval=(0, 10),
        preconditioner=RLPreconditioner(),
    ) where {L <: Number}
        coll = CollGaussRadauRight(M, 0, 1)
        num_nodes = coll.num_nodes
        Q = coll.Qmat[begin + 1:end, begin + 1:end]
        initial_u = ones(L, num_nodes)
        num_steps = 0

        new(M, dt, restol, max_sequence_length, max_episode_length,
            num_steps, num_nodes, Q, initial_u, preconditioner)
    end
end

function get_preconditioner(env, action, preconditioner=env.preconditioner)
    get_preconditioner(env, action, preconditioner)
end

function set_preconditioner!(env, preconditioner)
    env.preconditioner = preconditioner
end

function generate_lambda(lambda_real_interval,
                         lambda_imag_interval, type)
    generate_lambda(Random.GLOBAL_RNG, lambda_real_interval,
                    lambda_imag_interval, type)
end

function generate_lambda(rng, lambda_real_interval,
                         lambda_imag_interval, ::Type{<:Complex})
    @assert lambda_imag_interval[1] <= lambda_imag_interval[2]
    lambda_real = generate_lambda(rng, lambda_real_interval, nothing, Real)
    lambda_imag = (Random.rand(rng) * (lambda_imag_interval[2] - lambda_imag_interval[1])
                   + lambda_imag_interval[1])
    lambda_real + lambda_imag * im
end

function generate_lambda(rng, lambda_real_interval, ::Any, ::Type{<:Real})
    @assert lambda_real_interval[1] <= lambda_real_interval[2]
    lambda_real = (Random.rand(rng) * (lambda_real_interval[2] - lambda_real_interval[1])
                   + lambda_real_interval[1])
    lambda_real
end

function reset!(env)
    reset!(Random.GLOBAL_RNG, env)
end

function reset!(rng, env::SDCEnv{L}) where {L}
    env.lambda = generate_lambda(rng, (-100, 0), (0, 10), L)
    init!(env)
end

function init!(env)
    env.C = LinearAlgebra.I(env.num_nodes) - env.lambda .* env.dt .* env.Q

    residual = env.initial_u - env.C * env.initial_u
    env.initial_residual = residual
    env.last_residual_norm = norm(residual)

    env.num_steps = 0

    (env.initial_u, residual)
end

function step!(env, u, residual, action)
    Q_delta = get_preconditioner(env, action)
    P_inverse = inv(LinearAlgebra.I(env.num_nodes) - env.lambda .* env.dt .* Q_delta)

    u += P_inverse * residual
    residual = env.initial_u - env.C * u
    env.last_residual_norm = norm(residual)

    env.num_steps += 1

    (u, residual)
end

function set_lambda!(env, lambda)
    env.lambda = lambda
    init!(env)
end
