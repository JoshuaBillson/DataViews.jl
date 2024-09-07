module DataViews
using Random
using Pipe: @pipe

include("utils.jl")
export ones_like, zeros_like, putobs, rmobs, stackobs, unzip

include("views.jl")
export AbstractIterator, MappedView, JoinedView, ObsView, ZippedView, BatchedView
export numobs, getobs, data, obsview, splitobs, zipobs, repeatobs, takeobs, dropobs, filterobs, mapobs
export sampleobs, shuffleobs, kfolds

include("dataloader.jl")
export DataLoader

end
