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
stackobs(x::Vararg{Any}) = [x...]
stackobs(x::Vararg{AbstractArray{T,N}}) where {T,N} = cat(x..., dims=N)
stackobs(::Vararg{AbstractArray}) = throw(ArgumentError("Cannot stack arrays with different types or dimensions!"))
stackobs(x::Vararg{Tuple}) = @pipe collect(x) |> unzip |> map(stackobs, _)

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