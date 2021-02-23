mutable struct BarycentricInterpolator
    y_axis::Int
    xi::Vector{Float64}
    yi::Matrix{Float64}
    n::Int
    wi::Vector{Float64}
    r::Int
    y_extra_shape::Tuple

    function __set_yi(self, yi, xi)
        axis = self.y_axis

        shape = size(yi)
        if shape == ()
            shape = (1,)
        end
        @assert(shape[axis] == length(xi),
                "x and y arrays must be equal in length along interpolation axis")

        self.y_axis = (axis % ndims(yi)) + 1
        self.y_extra_shape = tuple(size(yi)[begin:self.y_axis - 1]...,
                                   size(yi)[self.y_axis + 1:end]...)
    end

    function _reshape_yi(self, yi)
        # TODO not necessary?
        # yi = np.rollaxis(yi, self.y_axis)
        reshape(yi, first(size(yi)), :)
    end

    function _set_yi(self, yi)
        __set_yi(self, yi, self.xi)
        self.yi = _reshape_yi(self, yi)
        (self.n, self.r) = size(self.yi)
    end

    function BarycentricInterpolator(xi, yi)
        self = new()
        self.y_axis = 1
        self.xi = xi
        _set_yi(self, yi)
        self.n = length(self.xi)

        self.wi = zeros(self.n)
        self.wi[begin] = 1
        for j in 2:self.n
            self.wi[begin:j - 1] .*= self.xi[j] .- self.xi[begin:j - 1]
            self.wi[j] = prod(self.xi[begin:j - 1] .- self.xi[j])
        end
        self.wi .^= -1
        self
    end
end

function (self::BarycentricInterpolator)(x)
    (x, x_shape) = _prepare_x(self, x)
    y = _evaluate(self, x)
    return _finish_y(self, y, x_shape)
end

function _to_array(x::AbstractArray)
    x
end

function _to_array(x)
    [x]
end

function _prepare_x(self, x)
    x = _to_array(x)
    (reshape(x, :), size(x))
end

function _finish_y(self, y, x_shape)
    y = reshape(y, x_shape..., self.y_extra_shape...)
    if self.y_axis != 1 && x_shape != ()
        nx = length(x_shape)
        ny = length(self.y_extra_shape)
        s = vcat(nx + 1:nx + self.y_axis,
                 1:nx - 1, nx + self.y_axis:nx + ny - 1)
        y = LinearAlgebra.permutedims(y, s)
    end
    y
end

function _evaluate(self, x)
    if length(x) == 0
        p = zeros(0, self.r)
    else
        c = reshape(x, size(x)..., 1) .- self.xi
        z = c .== 0
        c[z] .= 1
        c = self.wi ./ c
        csum = sum(c, dims=1)
        p = LinearAlgebra.dot(c, self.yi) ./ reshape(csum, size(csum)...)

        # We sort towards row-major-ordered columns here
        r = sort(findall(z), by=x -> (x[1], x[2]))
        if length(eltype(r)) == 1
            if length(r) > 0
                p = self.yi[first(r)[1]]
            end
        else
            p[[y[2] for y in r], :] = self.yi[[y[2] for y in r]]
        end
    end
    p
end
