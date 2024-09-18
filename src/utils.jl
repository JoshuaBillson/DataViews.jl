"""
    ones_like(x::AbstractArray)

Construct an array of ones with the same size and element type as `x`.
"""
ones_like(x::AbstractArray{T}) where {T} = ones(T, size(x))

"""
    zeros_like(x::AbstractArray)

Construct an array of zeros with the same size and element type as `x`.
"""
zeros_like(x::AbstractArray{T}) where {T} = zeros(T, size(x))

"""
    putobs(x::AbstractArray)

Add an N+1 obervation dimension of size 1 to the tensor `x`.
"""
putobs(x::AbstractArray) = reshape(x, size(x)..., 1)

"""
    rmobs(x::AbstractArray)

Remove the observation dimension from the tensor `x`.
"""
function rmobs(x::AbstractArray{<:Any,N}) where {N}
    @assert size(x,N) == 1 "Cannot drop observation dimension with size > 1!"
    dropdims(x, dims=N)
end

"""
    normalize(x::AbstractArray, μ::AbstractVector, σ::AbstractVector; dim=1)

Normalize the input array with respect to the specified dimension so that the mean is 0
and the standard deviation is 1.

# Parameters
- `μ`: A `Vector` of means for each index in `dim`.
- `σ`: A `Vector` of standard deviations for each index in `dim`.
- `dim`: The dimension along which to normalize the input array.
"""
function normalize(x::AbstractArray{<:Real,N}, μ::AbstractVector, σ::AbstractVector; dim=1) where {N}
    @assert 1 <= dim <= N
    @assert length(μ) == length(σ) == size(x,dim)
    return (x .- _vec2array(μ, N, dim)) ./ _vec2array(σ, N, dim)
end

"""
    denormalize(x::AbstractArray, μ::AbstractVector, σ::AbstractVector; dim=1)

Denormalize the input array with respect to the specified dimension. Reverses the
effect of `normalize`.

# Parameters
- `μ`: A `Vector` of means for each index in `dim`.
- `σ`: A `Vector` of standard deviations for each index in `dim`.
- `dim`: The dimension along which to denormalize the input array.
"""
function denormalize(x::AbstractArray{<:Real,N}, μ::AbstractVector, σ::AbstractVector; dim=1) where {N}
    @assert 1 <= dim <= N
    @assert length(μ) == length(σ) == size(x,dim)
    return (x .* _vec2array(σ, N, dim)) .+ _vec2array(μ, N, dim)
end

function _vec2array(x::AbstractVector, ndims::Int, dim::Int)
    return reshape(x, ntuple(i -> i == dim ? Colon() : 1, ndims))
end

"""
    onehot(x::AbstractArray, labels; dim=1)

Converts the input array `x` into a one-hot encoded representation based on the given `labels`. 
One-hot encoding transforms categorical data into an `Array` where each category is represented 
as a vector with one `1` and all other positions set to `0`.

# Arguments
- `x`: The input array containing categorical values that need to be one-hot encoded.
- `labels`: The set of possible labels (or categories) that `x` can take. This can be a vector of unique class labels or categories.

# Keyword Arguments
- `dim`: The dimension along which the one-hot encoding will be applied.

# Examples
```julia
julia> x = [1, 2, 3, 3, 1];

julia> labels = [1, 2, 3];

julia> onehot(x, labels)
3×5 BitMatrix:
 1  0  0  0  1
 0  1  0  0  0
 0  0  1  1  0

julia> x = rand([0,1], 28, 28, 1, 4);

julia> onehot(x, [0,1], dim=3) |> size
(28, 28, 2, 4)
"""
onehot(x::AbstractVector, labels; kw...) = onehot(reshape(x, (1,:)), labels)
function onehot(x::AbstractArray{<:Any,N}, labels; dim=1) where {N}
    @assert 1 <= dim <= N - 1
    @assert size(x, dim) == 1
    return cat(map(label -> x .== label, labels)..., dims=dim)
end

"""
    stackobs(x...)

Stack the elements in `x` as if they were observations in a batch. If `x` is an `AbstractArray`, 
elements will be concatenated along the Nth dimension. Other data types will simply be placed
in a `Vector` in the same order as they were received. Special attention is paid to a collection of
`Tuples`, where each tuple represents a single observation, such as a feature/label pair. In this
case, the tuples will be unzipped and have their constituent elements stacked as usual.

# Example
```julia
julia> stackobs(1, 2, 3, 4, 5)
5-element Vector{Int64}:
 1
 2
 3
 4
 5

julia> stackobs((1, :a), (2, :b), (3, :c))
([1, 2, 3], [:a, :b, :c])

julia> stackobs([rand(256, 256, 3, 1) for _ in 1:10]...) |> size
(256, 256, 3, 10)

julia> xs = [rand(256, 256, 3, 1) for _ in 1:10];

julia> ys = [rand(256, 256, 1, 1) for _ in 1:10];

julia> data = collect(zip(xs, ys));

julia> stackobs(data...) .|> size
((256, 256, 3, 10), (256, 256, 1, 10))
```
"""
stackobs(x::AbstractVector) = x
stackobs(::Vector{<:AbstractArray}) = throw(ArgumentError("Cannot stack arrays with different types or dimensions!"))
function stackobs(xs::Vector{<:AbstractArray{T,N}}) where {T,N}
    if size(first(xs), N) == 1
        return cat(xs..., dims=N)
    else
        return cat(xs..., dims=N+1)
    end
end
function stackobs(xs::Vector{<:Tuple})
    n = length(first(xs))
    @assert all(length.(xs) .== n) "Cannot batch tuples with different lengths"
    return ntuple(i -> stackobs([x[i] for x in xs]), n)
end
function stackobs(xs::Vector{<:NamedTuple})
    all_keys = [sort(collect(keys(x))) for x in xs]
    ks = all_keys[1]
    @assert all(==(ks), all_keys) "Cannot batch named tuples with different keys"
    NamedTuple(k => stackobs([x[k] for x in xs]) for k in ks)
end

"""
    unzip(x::AbstractVector{<:Tuple})

The inverse of `zip`.

# Example
```julia
julia> zip([1, 2, 3], [:a, :b, :c]) |> collect |> unzip
([1, 2, 3], [:a, :b, :c])
```
"""
unzip(x) = map(f -> getfield.(x, f), fieldnames(eltype(x)))

_all_equal(f, xs) = map(f, xs) |> _all_equal
_all_equal(xs) = all(==(first(xs)), xs)

_flatten_tuple(x::Tuple) = reduce(_flatmerge, x)

_flatmerge(a, b::Tuple) = (a, reduce(_flatmerge, b)...)
_flatmerge(a::Tuple, b) = (reduce(_flatmerge, a)..., b)
_flatmerge(a::Tuple, b::Tuple) = (reduce(_flatmerge, a)..., reduce(_flatmerge, b)...)
_flatmerge(a, b) = (a, b)