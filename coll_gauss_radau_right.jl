import LinearAlgebra

import QuadGK

include("barycentric_interpolator.jl")

mutable struct CollGaussRadauRight
    order::Int
    num_nodes::Int
    tleft::Float64
    tright::Float64
    nodes::Vector{Float64}
    weights::Vector{Float64}
    Qmat::Matrix{Float64}
    Smat::Matrix{Float64}
    delta_m::Vector{Float64}
    right_is_node::Bool
    left_is_node::Bool

    function _get_nodes(self)
        M = self.num_nodes
        a = self.tleft
        b = self.tright

        if M == 1
            return [b]
        end

        alpha = 1.0
        beta = 0.0

        diag = zeros(M - 1)
        subdiag = zeros(M - 2)

        diag[begin] = (beta - alpha) / (2 + alpha + beta)

        for jj in 1:(M - 2)
            diag[jj + 1] = ((beta - alpha) * (alpha + beta)
                            / (2 * jj + 2 + alpha + beta)
                            / (2 * jj + alpha + beta))
            num = sqrt(4 * jj * (jj + alpha) * (jj + beta) * (jj + alpha + beta))
            denom = sqrt((2 * jj - 1 + alpha + beta)
                         * (2 * jj + alpha + beta) ^ 2
                         * (2 * jj + 1 + alpha + beta))
            subdiag[jj] = num / denom
        end

        mat = LinearAlgebra.diagm(-1 => subdiag, 0 => diag, 1 => subdiag)
        x = LinearAlgebra.eigvals!(mat)

        nodes = vcat(x, [1.0])
        nodes = (a * (1 .- nodes) + b * (1 .+ nodes)) / 2
        nodes
    end

    function _get_weights(self, a, b)
        circ_one = zeros(self.num_nodes)
        circ_one[begin] = 1.0
        tcks = []
        for i in 0:(self.num_nodes - 1)
            push!(tcks, BarycentricInterpolator(self.nodes, circshift(circ_one, i)))
        end

        weights = zeros(self.num_nodes)
        for (i, tck) in enumerate(tcks)
            weights[i] = first(first(QuadGK.quadgk(tck, a, b, rtol=1e-14)))
        end
        weights
    end

    function _gen_Qmat(self)
        M = self.num_nodes
        Q = zeros(M + 1, M + 1)

        for m in 1:M
            Q[m + 1, begin + 1:end] = _get_weights(self, self.tleft, self.nodes[m])
        end
        Q
    end

    function _gen_Smat(self)
        M = self.num_nodes
        Q = self.Qmat
        S = zeros(M + 1, M + 1)

        S[begin + 1, :] = Q[begin + 1, :]
        for m in 3:M + 1
            S[m, :] = Q[m, :] - Q[m - 1, :]
        end
        S
    end

    function _gen_deltas(self)
        M = self.num_nodes
        delta = zeros(M)
        delta[begin] = self.nodes[begin] - self.tleft
        for m in 2:M
            delta[m] = self.nodes[m] - self.nodes[m - 1]
        end
        delta
    end

    function CollGaussRadauRight(num_nodes, tleft, tright)
        @assert num_nodes > 0 "at least one quadrature node required, got $num_nodes"
        @assert tleft < tright "interval boundaries are corrupt, got $tleft and $tright"

        self = new()
        self.num_nodes = num_nodes
        self.tleft = tleft
        self.tright = tright
        self.order = 2 * num_nodes - 1
        self.nodes = _get_nodes(self)
        self.weights = _get_weights(self, tleft, tright)
        self.Qmat = _gen_Qmat(self)
        self.Smat = _gen_Smat(self)
        self.delta_m = _gen_deltas(self)
        self.left_is_node = false
        self.right_is_node = true
        # From NumPy to Julia ordering
        self.Qmat = collect(transpose(_gen_Qmat(self)))
        self.Smat = collect(transpose(_gen_Smat(self)))
        self
    end
end
