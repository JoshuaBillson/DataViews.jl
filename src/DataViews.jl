module DataViews
using Random
import Tables
using Pipe: @pipe

include("utils.jl")
export ones_like, zeros_like, putobs, rmobs, normalize, denormalize, onehot, stackobs, unzip

include("views.jl")
export AbstractIterator, MappedView, JoinedView, ObsView, ZippedView, BatchedView, CachedView
export numobs, getobs, data, obsview, splitobs, zipobs, repeatobs, takeobs, dropobs, filterobs, mapobs
export sampleobs, shuffleobs, kfolds, normobs

include("dataloader.jl")
export DataLoader

include("show.jl")

end
