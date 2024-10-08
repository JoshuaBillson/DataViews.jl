"""
    DataLoader(data; batchsize=1, partial=true, shuffle=false, parallel=true, rng=Random.default_rng())

An object that iterates over mini-batches of `data`, each mini-batch containing `batchsize` observations
(except possibly the last one).

Takes as input a single data array, or in general any `data` object that implements the `numobs` and `getobs`
methods.

The last dimension in each array is the observation dimension.
"""
struct DataLoader{D,R}
    data::D
    batchsize::Int
    partial::Bool
    shuffle::Bool
    parallel::Bool
    rng::R
end

function DataLoader(data; batchsize=1, partial=true, shuffle=false, parallel=true, rng=Random.default_rng())
    return DataLoader(data, batchsize, partial, shuffle, parallel, rng)
end

data(x::DataLoader) = data(x.data)

function Base.iterate(x::DataLoader)
    # Construct ObsView over data
    data = x.shuffle ? shuffleobs(x.rng, x.data) : ObsView(x.data, 1:numobs(x.data))

    # Partition Into Batches
    batches = BatchedView(data, batchsize=x.batchsize, partial=x.partial)

    # Return Observations
    obs, state = iterate(batches)
    return obs, (batches, state)
end

function Base.iterate(::DataLoader, (batches, state))
    ret = iterate(batches, state)
    isnothing(ret) && return
    obs, newstate = ret
    return obs, (batches, newstate)
end

function Base.length(x::DataLoader)
    return x.partial ? cld(numobs(x.data), x.batchsize) : fld(numobs(x.data), x.batchsize)
end