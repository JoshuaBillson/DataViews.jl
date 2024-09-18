# Interface

"""
Super type of all iterators.
"""
abstract type AbstractIterator{D} end

data(x::Any) = x
data(x::AbstractIterator) = data(x.data)

"""
    numobs(data)

Return the total number of observations contained in `data`.

Returns `length(data)` if `data` does not have `numobs` defined.

Arrays are a special case where `numobs` returns the size of the last dimension.

Authors of custom data containers should implement `Base.length` for their type instead of `numobs`.
`numobs` should only be implemented for types where there is a difference between `numobs` and `Base.length`
(such as multi-dimensional arrays).
"""
numobs(x::Any) = Tables.istable(x) ? length(Tables.rows(x)) : length(x)
numobs(x::Tuple) = map(numobs, x) |> minimum
numobs(x::NamedTuple) = map(numobs, x) |> minimum
numobs(x::AbstractArray{T,N}) where {T,N} = size(x, N)

"""
    getobs(data, [idx])

Return the observations corresponding to the observation index `idx`.
Note that `idx` can be any type as long as `data` has defined
`getobs` for that type. If `idx` is not provided, then materialize
all observations in `data`.

Returns `data[idx]` if `data` does not have `getobs` defined.

Authors of custom data containers should implement
`Base.getindex` for their type instead of `getobs`.
`getobs` should only be implemented for types where there is a
difference between `getobs` and `Base.getindex`
(such as multi-dimensional arrays).

The returned observation(s) should be in the form intended to
be passed as-is to some learning algorithm. There is no strict
interface requirement on how this "actual data" must look like.
Every author behind some custom data container can make this
decision themselves.
The output should be consistent when `idx` is a scalar vs vector.
"""
getobs(x::Any) = getobs(x, 1:numobs(x))
getobs(x::AbstractVector, i::Integer) = x[i]
getobs(x::Tuple, i::Integer) = map(x -> getobs(x, i), x) |> _flatten_tuple
getobs(x::NamedTuple, i::Integer) = map(x -> getobs(x, i), x)
getobs(x::Any, i::Integer) = Tables.istable(x) ? Tables.subset(x, i, viewhint=false) : x[i]
getobs(x::AbstractArray{T,N}, i::Integer) where {T,N} = selectdim(x, N, i:i) |> collect
getobs(x::Any, i::AbstractVector) = stackobs(map(j -> getobs(x, j), i))

firstobs(x) = getobs(x, 1)
lastobs(x) = getobs(x, numobs(x))

stackobs(x::AbstractIterator) = x[:]

Base.collect(x::AbstractIterator) = x[:]

Base.getindex(x::AbstractIterator, ::Colon) = getindex(x, firstindex(x):lastindex(x))
Base.getindex(x::AbstractIterator, i::AbstractVector) = stackobs(map(j -> getobs(x, j), i))

Base.iterate(x::AbstractIterator, state=1) = state > numobs(x) ? nothing : (x[state], state+1)

Base.firstindex(x::AbstractIterator) = 1
Base.lastindex(x::AbstractIterator) = numobs(x)

Base.keys(x::AbstractIterator) = Base.OneTo(numobs(x))

# MappedView

"""
    MappedView(f, data)

An iterator which lazily applies `f` to each element in `data` when requested.
"""
struct MappedView{F<:Function,D} <: AbstractIterator{D}
    f::F
    data::D
end

Base.length(x::MappedView) = numobs(data(x))

Base.getindex(x::MappedView, i::Int) = getobs(x.data, i) |> x.f

# JoinedView

"""
    JoinedView(data...)

An object that iterates over each element in the iterators given by `data` as
if they were concatenated into a single list.
"""
struct JoinedView{D} <: AbstractIterator{D}
    data::D
    JoinedView(data...) = JoinedView(data)
    JoinedView(data::D) where {D <: Tuple} = new{D}(data)
end

data(x::JoinedView) = map(data, x.data)

Base.length(x::JoinedView) = map(numobs, x.data) |> sum

function Base.getindex(x::JoinedView, i::Int)
    (i > numobs(x) || i < 1) &&  throw(BoundsError(x, i)) 
    lengths = map(numobs, x.data) |> cumsum
    for (j, len) in enumerate(lengths)
        if i <= len
            offset = j == 1 ? 0 : lengths[j-1]
            return getobs(x.data[j], i-offset)
        end
    end
end

# ObsView

"""
    ObsView(data, indices)

Construct an iterator over the elements specified by `indices` in `data`.
"""
struct ObsView{D} <: AbstractIterator{D}
    data::D
    indices::Vector{Int}

    ObsView(data, indices::AbstractVector{Int}) = ObsView(data, collect(indices))
    function ObsView(data::D, indices::Vector{Int}) where {D}
        _check_indices(data, indices)
        new{D}(data, indices)
    end
end

function _check_indices(data, indices)
    i = findfirst(x -> x < 1 || x > numobs(data), indices)
    isnothing(i) || throw(ArgumentError("Index $(indices[i]) is out of bounds!"))
end

Base.length(x::ObsView) = length(x.indices)

Base.getindex(x::ObsView, i::Int) = getobs(x.data, x.indices[i])

# ZippedView

"""
    ZippedView(data...)

Construct an iterator that zips each element of the given subiterators into a `Tuple`.
"""
struct ZippedView{D} <: AbstractIterator{D}
    data::D

    ZippedView(data...) = ZippedView(data)
    ZippedView(data::D) where {D<:Tuple} = new{D}(data)
end

data(x::ZippedView) = map(data, x.data)

Base.length(x::ZippedView) = map(numobs, data(x)) |> minimum

Base.getindex(x::ZippedView, i::Int) = map(d -> getobs(d, i), data(x)) |> _flatten_tuple

# BatchedView

struct BatchedView{D} <: AbstractIterator{D}
    data::D
    batchsize::Int
    partial::Bool
end

function BatchedView(data; batchsize=1, partial=true)
    return BatchedView(data, batchsize, partial)
end

Base.length(x::BatchedView) = x.partial ? cld(numobs(x.data), x.batchsize) : fld(numobs(x.data), x.batchsize)

function Base.getindex(x::BatchedView, i::Int)
    if i <= length(x)
        start_index = (i - 1) * x.batchsize + 1
        end_index = min(start_index + x.batchsize - 1, numobs(x.data))
        return getobs(x.data, start_index:end_index)
    else
        throw(BoundsError(x, i))
    end
end

# CachedView

"""
    CachedView(data)

Construct an iterator that caches each element in memory on the first retrieval.
When an index is passed for the first time, the corresponding element will be saved
in a lookup table, which will be used for every subsequent retrieval. Useful for
reusing the result of expensive computations.
"""
struct CachedView{D,V} <: AbstractIterator{D}
    data::D
    cache::Dict{Int,V}
end

function CachedView(data)
    cache = Dict{Int,typeof(firstobs(data))}()
    return CachedView(data, cache)
end

Base.length(x::CachedView) = numobs(x.data)

function Base.getindex(x::CachedView, i::Int)
    if i in keys(x.cache)
        return x.cache[i]
    else
        result = getobs(x.data, i)
        x.cache[i] = result
        return result
    end
end

# Methods

"""
    obsview(data, indices::AbstractVector{<:Integer})

Construct a lazy view of `data` at the specified `indices`.
"""
obsview(data, indices::AbstractVector{<:Integer}) = ObsView(data, indices)

"""
    splitobs([rng=default_rng()], data::Int; kw...)
    splitobs([rng=default_rng()], data::AbstractVector{Int}; kw...)
    splitobs([rng=default_rng()], data; at=0.8, shuffle=true)

Return a set of indices that splits the given observations according to the given break points.

# Arguments
- `data`: Any type that implements either `Base.length()` or `numobs`. Alternatively, can be
either an `AbstractVector` of indices or an `Int` indicating the number of observations.
- `at`: The fractions at which to split `data`. 
- `shuffle`: If `true`, shuffles the indices before splitting. 

# Example
```julia
julia> splitobs(1:100, at=(0.7, 0.2), shuffle=false)
3-element Vector{Vector{Int64}}:
 [1, 2, 3, 4, 5, 6, 7, 8, 9, 10  …  61, 62, 63, 64, 65, 66, 67, 68, 69, 70]
 [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]
 [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]
```
"""
splitobs(data; kwargs...) = splitobs(Random.default_rng(), data; kwargs...)
splitobs(rng::Random.AbstractRNG, data; kwargs...) = map(x -> ObsView(data, x), splitobs(rng, 1:numobs(data); kwargs...))
splitobs(rng::Random.AbstractRNG, n::Int; kwargs...) = splitobs(rng, 1:n; kwargs...)
function splitobs(rng::Random.AbstractRNG, data::AbstractVector{Int}; at=0.8, shuffle=true)
    sum(at) > 1 && throw(ArgumentError("'at' cannot sum to more than 1!"))
    indices = shuffle ? Random.randperm(rng, length(data)) : collect(1:length(data))
    breakpoints = _breakpoints(length(data), at)
    starts = (1, (breakpoints .+ 1)...)
    ends = ((breakpoints)..., length(data))
    return [indices[s:e] for (s, e) in zip(starts, ends)]
end

_breakpoints(n::Int, at::Tuple) = round.(Int, cumsum(at) .* n)
_breakpoints(n::Int, at::Real) = _breakpoints(n, (at,))
_breakpoints(n::Int, at::AbstractVector) = _breakpoints(n, Tuple(at))

"""
    zipobs(data...)

Create a new iterator where the elements of each iterator in `data` are returned as a tuple.

# Example
```jldoctest
julia> z = zipobs(1:5, 41:45, [:a, :b, :c, :d, :e])
5-element ZippedView
  with first element:
  (Int64, Int64, Symbol)

julia> [x for x in z]
5-element Vector{Tuple{Int64, Int64, Symbol}}:
 (1, 41, :a)
 (2, 42, :b)
 (3, 43, :c)
 (4, 44, :d)
 (5, 45, :e)

julia> z[1:2:end]
([1, 3, 5], [41, 43, 45], [:a, :c, :e])
```
"""
zipobs(data...) = zipobs(tuple(data...))
zipobs(data::Tuple) = ZippedView(data)

"""
    repeatobs(data, n::Int)

Create a new view which iterates over every element in `data` `n` times.
"""
repeatobs(data, n::Int) = JoinedView([data for _ in 1:n]...)

"""
    takeobs(data, obs::AbstractVector{Int})

Take all observations from `data` whose index corresponds to `obs` while removing everything else.
"""
takeobs(data, obs::AbstractVector{Int}) = ObsView(data, obs)

"""
    dropobs(data, obs::AbstractVector{Int})

Remove all observations from `data` whose index corresponds to those in `obs`.
"""
dropobs(data, obs::AbstractVector{Int}) = takeobs(data, filter(x -> !(x in obs), eachindex(data)))

"""
    filterobs(f, data)

Remove all observations from `data` for which `f` returns `false`.
"""
filterobs(f, data) = takeobs(data, findall(map(f, data)))

"""
    mapobs(f, data)

Lazily apply the function `f` to each element in `data`.
"""
mapobs(f, data) = MappedView(f, data)

"""
    sampleobs([rng=default_rng()], data, n)

Randomly sample `n` elements from `data` without replacement. `rng` may be optionally
provided for reproducible results.
"""
sampleobs(data, n::Int) = sampleobs(Random.default_rng(), data, n)
function sampleobs(rng::Random.AbstractRNG, data, n::Int)
    if (n > numobs(data)) || (n < 0)
        throw(ArgumentError("n must be between 0 and $(numobs(data)) (received $n)."))
    end
    takeobs(data, Random.randperm(rng, numobs(data))[1:n])
end

"""
    shuffleobs([rng=default_rng()], data)

Randomly shuffle the elements of `data`. Provide `rng` for reproducible results.
"""
shuffleobs(data) = shuffleobs(Random.default_rng(), data)
shuffleobs(rng::Random.AbstractRNG, data) = takeobs(data, Random.randperm(rng, numobs(data)))

function normobs(data, μ::AbstractVector{<:Real}, σ::AbstractVector{<:Real}; dim=1)
    return mapobs(x -> normalize(x, μ, σ, dim=dim), data)
end

"""
    kfolds(data, k = 5)
    kfolds(data::Integer, k = 5)
    kfolds(data::AbstractVector{<:Integer}, k = 5)

Compute the train/validation splits for `k` repartitions of
`n` observations, and return them as a vector of `(train, validation)`
pairs. If `data` is an `Integer`, then a vector containing the indices `1:n`
will be materialized and partitioned. If `data` is an iterable, 
a lazy `ObsView` will be constructed for each fold.
"""
function kfolds(data::AbstractVector{<:Integer}, k::Integer = 5)
    folds = kfolds(length(data), k)
    return [(getindex(data, train), getindex(data, val)) for (train, val) in folds]
end
function kfolds(data, k::Integer = 5)
    folds = kfolds(length(data), k)
    return [(obsview(data, train), obsview(data, val)) for (train, val) in folds]
end
function kfolds(n::Integer, k::Integer = 5)
    2 <= k <= n || throw(ArgumentError("n must be positive and k must to be within 2:$(max(2,n))"))
    # Compute size of each fold, dividing remaining observations between folds
    sizes = fill(floor(Int, n/k), k)
    for i = 1:(n % k)
        sizes[i] = sizes[i] + 1
    end
    # Compute start offset for each fold
    offsets = cumsum(sizes) .- sizes .+ 1
    # Compute the validation indices using the offsets and sizes
    val_indices = map((o,s)->collect(o:o+s-1), offsets, sizes)
    # The train indices are then the indicies not in validation
    train_indices = map(idx->setdiff(1:n,idx), val_indices)
    # We return a tuple of arrays
    collect(zip(train_indices, val_indices))
end