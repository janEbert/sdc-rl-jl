import Random

import Flux

# Layers

function LSTMCell(in::Integer, out::Integer; initW, initb, init_state)
    cell = Flux.LSTMCell(initW(out * 4, in), initW(out * 4, out), initb(out * 4),
                         init_state(out), init_state(out))
    cell.b[Flux.gate(out, 2)] .= 1
    return cell
end

LSTM(a...; ka...) = Flux.Recur(LSTMCell(a...; ka...))

struct CLSTMCell{A,V}
    Wi::A
    Wh::A
    b::V
    h::V
    c::V
end

function CLSTMCell(in::Integer, out::Integer; initW, initb, init_state)
    cell = CLSTMCell(initW(out * 4, in), initW(out * 4, out), initb(out * 4),
                     init_state(out), init_state(out))
    cell.b[Flux.gate(out, 2)] .= 1
    return cell
end

function (m::CLSTMCell)((h, c), x)
    b, o = m.b, size(h, 1)
    g = m.Wi*x .+ m.Wh*h .+ b
    input = sigmoid.(gate(g, o, 1))
    forget = sigmoid.(gate(g, o, 2))
    cell = tanh.(gate(g, o, 3))
    output = sigmoid.(gate(g, o, 4))
    c = forget .* c .+ input .* cell
    h′ = output .* tanh.(c)
    return (h′, c), h′
end

hidden(m::CLSTMCell) = (m.h, m.c)

Flux.@functor CLSTMCell

function Base.show(io::IO, l::CLSTMCell)
    print(io, "CLSTMCell(", size(l.Wi, 2), ", ", size(l.Wi, 1)÷4, ")")
end

CLSTM(a...; ka...) = Flux.Recur(CLSTMCell(a...; ka...))

# Initialization

function glorot_uniform(rng::Random.AbstractRNG, type::Type{<:Number}, dims...)
    (rand(rng, type, dims...) .- 0.5) .* sqrt(24.0 / sum(Flux.nfan(dims...)))
end

function glorot_uniform(type::Type{<:Number}, dims...)
    glorot_uniform(Random.GLOBAL_RNG, type, dims...)
end

function glorot_uniform(type::Type{<:Number})
    (dims...) -> glorot_uniform(type, dims...)
end

function init_bias(type::Type{<:Number}, dims...)
    zeros(type, dims...)
end

function init_bias(type::Type{<:Number})
    (dims...) -> init_bias(type, dims...)
end

# Activation

function crelu(x)
    ifelse(real(x) >= 0 && imag(x) >= 0, x, zero(x))
end

function crelu2(x)
    Flux.relu(real(x)) + Flux.relu(imag(x)) * im
end

function sigmoid(x)
    1 / (1 + exp(-x))
end

function softplus(x)
    log(1 + exp(x))
end

function silu(x)
    x * sigmoid(x)
end
