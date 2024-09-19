var documenterSearchIndex = {"docs":
[{"location":"","page":"Home","title":"Home","text":"CurrentModule = DataViews","category":"page"},{"location":"#DataViews","page":"Home","title":"DataViews","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"Documentation for DataViews.","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [DataViews]","category":"page"},{"location":"#DataViews.AbstractIterator","page":"Home","title":"DataViews.AbstractIterator","text":"Super type of all iterators.\n\n\n\n\n\n","category":"type"},{"location":"#DataViews.CachedView","page":"Home","title":"DataViews.CachedView","text":"CachedView(data)\n\nConstruct an iterator that caches each element in memory on the first retrieval. When an index is passed for the first time, the corresponding element will be saved in a lookup table, which will be used for every subsequent retrieval. Useful for reusing the result of expensive computations.\n\n\n\n\n\n","category":"type"},{"location":"#DataViews.DataLoader","page":"Home","title":"DataViews.DataLoader","text":"DataLoader(data; batchsize=1, partial=true, shuffle=false, parallel=true, rng=Random.default_rng())\n\nAn object that iterates over mini-batches of data, each mini-batch containing batchsize observations (except possibly the last one).\n\nTakes as input a single data array, or in general any data object that implements the numobs and getobs methods.\n\nThe last dimension in each array is the observation dimension.\n\n\n\n\n\n","category":"type"},{"location":"#DataViews.JoinedView","page":"Home","title":"DataViews.JoinedView","text":"JoinedView(data...)\n\nAn object that iterates over each element in the iterators given by data as if they were concatenated into a single list.\n\n\n\n\n\n","category":"type"},{"location":"#DataViews.MappedView","page":"Home","title":"DataViews.MappedView","text":"MappedView(f, data)\n\nAn iterator which lazily applies f to each element in data when requested.\n\n\n\n\n\n","category":"type"},{"location":"#DataViews.ObsView","page":"Home","title":"DataViews.ObsView","text":"ObsView(data, indices)\n\nConstruct an iterator over the elements specified by indices in data.\n\n\n\n\n\n","category":"type"},{"location":"#DataViews.ZippedView","page":"Home","title":"DataViews.ZippedView","text":"ZippedView(data...)\n\nConstruct an iterator that zips each element of the given subiterators into a Tuple.\n\n\n\n\n\n","category":"type"},{"location":"#DataViews.denormalize-Tuple{AbstractArray{<:Integer}, AbstractVector, AbstractVector}","page":"Home","title":"DataViews.denormalize","text":"denormalize(x::AbstractArray, μ::AbstractVector, σ::AbstractVector; dim=1)\n\nDenormalize the input array with respect to the specified dimension. Reverses the effect of normalize.\n\nParameters\n\nμ: A Vector of means for each index in dim.\nσ: A Vector of standard deviations for each index in dim.\ndim: The dimension along which to denormalize the input array.\n\n\n\n\n\n","category":"method"},{"location":"#DataViews.dropobs-Tuple{Any, AbstractVector{Int64}}","page":"Home","title":"DataViews.dropobs","text":"dropobs(data, obs::AbstractVector{Int})\n\nRemove all observations from data whose index corresponds to those in obs.\n\n\n\n\n\n","category":"method"},{"location":"#DataViews.filterobs-Tuple{Any, Any}","page":"Home","title":"DataViews.filterobs","text":"filterobs(f, data)\n\nRemove all observations from data for which f returns false.\n\n\n\n\n\n","category":"method"},{"location":"#DataViews.getobs-Tuple{Any}","page":"Home","title":"DataViews.getobs","text":"getobs(data, [idx])\n\nReturn the observations corresponding to the observation index idx. Note that idx can be any type as long as data has defined getobs for that type. If idx is not provided, then materialize all observations in data.\n\nReturns data[idx] if data does not have getobs defined.\n\nAuthors of custom data containers should implement Base.getindex for their type instead of getobs. getobs should only be implemented for types where there is a difference between getobs and Base.getindex (such as multi-dimensional arrays).\n\nThe returned observation(s) should be in the form intended to be passed as-is to some learning algorithm. There is no strict interface requirement on how this \"actual data\" must look like. Every author behind some custom data container can make this decision themselves. The output should be consistent when idx is a scalar vs vector.\n\n\n\n\n\n","category":"method"},{"location":"#DataViews.kfolds","page":"Home","title":"DataViews.kfolds","text":"kfolds(data, k = 5)\nkfolds(data::Integer, k = 5)\nkfolds(data::AbstractVector{<:Integer}, k = 5)\n\nCompute the train/validation splits for k repartitions of n observations, and return them as a vector of (train, validation) pairs. If data is an Integer, then a vector containing the indices 1:n will be materialized and partitioned. If data is an iterable,  a lazy ObsView will be constructed for each fold.\n\n\n\n\n\n","category":"function"},{"location":"#DataViews.mapobs-Tuple{Any, Any}","page":"Home","title":"DataViews.mapobs","text":"mapobs(f, data)\n\nLazily apply the function f to each element in data.\n\n\n\n\n\n","category":"method"},{"location":"#DataViews.normalize-Tuple{AbstractArray{<:Integer}, AbstractVector, AbstractVector}","page":"Home","title":"DataViews.normalize","text":"normalize(x::AbstractArray, μ::AbstractVector, σ::AbstractVector; dim=1)\n\nNormalize the input array with respect to the specified dimension so that the mean is 0 and the standard deviation is 1.\n\nParameters\n\nμ: A Vector of means for each index in dim.\nσ: A Vector of standard deviations for each index in dim.\ndim: The dimension along which to normalize the input array.\n\n\n\n\n\n","category":"method"},{"location":"#DataViews.numobs-Tuple{Any}","page":"Home","title":"DataViews.numobs","text":"numobs(data)\n\nReturn the total number of observations contained in data.\n\nReturns length(data) if data does not have numobs defined.\n\nArrays are a special case where numobs returns the size of the last dimension.\n\nAuthors of custom data containers should implement Base.length for their type instead of numobs. numobs should only be implemented for types where there is a difference between numobs and Base.length (such as multi-dimensional arrays).\n\n\n\n\n\n","category":"method"},{"location":"#DataViews.obsview-Tuple{Any, AbstractVector{<:Integer}}","page":"Home","title":"DataViews.obsview","text":"obsview(data, indices::AbstractVector{<:Integer})\n\nConstruct a lazy view of data at the specified indices.\n\n\n\n\n\n","category":"method"},{"location":"#DataViews.onehot-Tuple{AbstractVector, Any}","page":"Home","title":"DataViews.onehot","text":"onehot(x::AbstractArray, labels; dim=1)\n\nConverts the input array x into a one-hot encoded representation based on the given labels.  One-hot encoding transforms categorical data into an Array where each category is represented  as a vector with one 1 and all other positions set to 0.\n\nArguments\n\nx: The input array containing categorical values that need to be one-hot encoded.\nlabels: The set of possible labels (or categories) that x can take. This can be a vector of unique class labels or categories.\n\nKeyword Arguments\n\ndim: The dimension along which the one-hot encoding will be applied.\n\nExamples\n\n```julia julia> x = [1, 2, 3, 3, 1];\n\njulia> labels = [1, 2, 3];\n\njulia> onehot(x, labels) 3×5 BitMatrix:  1  0  0  0  1  0  1  0  0  0  0  0  1  1  0\n\njulia> x = rand([0,1], 28, 28, 1, 4);\n\njulia> onehot(x, [0,1], dim=3) |> size (28, 28, 2, 4)\n\n\n\n\n\n","category":"method"},{"location":"#DataViews.ones_like-Union{Tuple{AbstractArray{T}}, Tuple{T}} where T","page":"Home","title":"DataViews.ones_like","text":"ones_like(x::AbstractArray)\n\nConstruct an array of ones with the same size and element type as x.\n\n\n\n\n\n","category":"method"},{"location":"#DataViews.putobs-Tuple{AbstractArray}","page":"Home","title":"DataViews.putobs","text":"putobs(x::AbstractArray)\n\nAdd an N+1 obervation dimension of size 1 to the tensor x.\n\n\n\n\n\n","category":"method"},{"location":"#DataViews.repeatobs-Tuple{Any, Int64}","page":"Home","title":"DataViews.repeatobs","text":"repeatobs(data, n::Int)\n\nCreate a new view which iterates over every element in data n times.\n\n\n\n\n\n","category":"method"},{"location":"#DataViews.rmobs-Union{Tuple{AbstractArray{<:Any, N}}, Tuple{N}} where N","page":"Home","title":"DataViews.rmobs","text":"rmobs(x::AbstractArray)\n\nRemove the observation dimension from the tensor x.\n\n\n\n\n\n","category":"method"},{"location":"#DataViews.sampleobs-Tuple{Any, Int64}","page":"Home","title":"DataViews.sampleobs","text":"sampleobs([rng=default_rng()], data, n)\n\nRandomly sample n elements from data without replacement. rng may be optionally provided for reproducible results.\n\n\n\n\n\n","category":"method"},{"location":"#DataViews.shuffleobs-Tuple{Any}","page":"Home","title":"DataViews.shuffleobs","text":"shuffleobs([rng=default_rng()], data)\n\nRandomly shuffle the elements of data. Provide rng for reproducible results.\n\n\n\n\n\n","category":"method"},{"location":"#DataViews.splitobs-Tuple{Any}","page":"Home","title":"DataViews.splitobs","text":"splitobs([rng=default_rng()], data::Int; kw...)\nsplitobs([rng=default_rng()], data::AbstractVector{Int}; kw...)\nsplitobs([rng=default_rng()], data; at=0.8, shuffle=true)\n\nReturn a set of indices that splits the given observations according to the given break points.\n\nArguments\n\ndata: Any type that implements either Base.length() or numobs. Alternatively, can be\n\neither an AbstractVector of indices or an Int indicating the number of observations.\n\nat: The fractions at which to split data. \nshuffle: If true, shuffles the indices before splitting. \n\nExample\n\njulia> splitobs(1:100, at=(0.7, 0.2), shuffle=false)\n3-element Vector{Vector{Int64}}:\n [1, 2, 3, 4, 5, 6, 7, 8, 9, 10  …  61, 62, 63, 64, 65, 66, 67, 68, 69, 70]\n [71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90]\n [91, 92, 93, 94, 95, 96, 97, 98, 99, 100]\n\n\n\n\n\n","category":"method"},{"location":"#DataViews.stackobs-Tuple{AbstractVector}","page":"Home","title":"DataViews.stackobs","text":"stackobs(x...)\n\nStack the elements in x as if they were observations in a batch. If x is an AbstractArray,  elements will be concatenated along the Nth dimension. Other data types will simply be placed in a Vector in the same order as they were received. Special attention is paid to a collection of Tuples, where each tuple represents a single observation, such as a feature/label pair. In this case, the tuples will be unzipped and have their constituent elements stacked as usual.\n\nExample\n\njulia> stackobs(1, 2, 3, 4, 5)\n5-element Vector{Int64}:\n 1\n 2\n 3\n 4\n 5\n\njulia> stackobs((1, :a), (2, :b), (3, :c))\n([1, 2, 3], [:a, :b, :c])\n\njulia> stackobs([rand(256, 256, 3, 1) for _ in 1:10]...) |> size\n(256, 256, 3, 10)\n\njulia> xs = [rand(256, 256, 3, 1) for _ in 1:10];\n\njulia> ys = [rand(256, 256, 1, 1) for _ in 1:10];\n\njulia> data = collect(zip(xs, ys));\n\njulia> stackobs(data...) .|> size\n((256, 256, 3, 10), (256, 256, 1, 10))\n\n\n\n\n\n","category":"method"},{"location":"#DataViews.takeobs-Tuple{Any, AbstractVector{Int64}}","page":"Home","title":"DataViews.takeobs","text":"takeobs(data, obs::AbstractVector{Int})\n\nTake all observations from data whose index corresponds to obs while removing everything else.\n\n\n\n\n\n","category":"method"},{"location":"#DataViews.unzip-Tuple{Any}","page":"Home","title":"DataViews.unzip","text":"unzip(x::AbstractVector{<:Tuple})\n\nThe inverse of zip.\n\nExample\n\njulia> zip([1, 2, 3], [:a, :b, :c]) |> collect |> unzip\n([1, 2, 3], [:a, :b, :c])\n\n\n\n\n\n","category":"method"},{"location":"#DataViews.zeros_like-Union{Tuple{AbstractArray{T}}, Tuple{T}} where T","page":"Home","title":"DataViews.zeros_like","text":"zeros_like(x::AbstractArray)\n\nConstruct an array of zeros with the same size and element type as x.\n\n\n\n\n\n","category":"method"},{"location":"#DataViews.zipobs-Tuple","page":"Home","title":"DataViews.zipobs","text":"zipobs(data...)\n\nCreate a new iterator where the elements of each iterator in data are returned as a tuple.\n\nExample\n\njulia> z = zipobs(1:5, 41:45, [:a, :b, :c, :d, :e])\n5-element ZippedView\n  with first element:\n  (Int64, Int64, Symbol)\n\njulia> [x for x in z]\n5-element Vector{Tuple{Int64, Int64, Symbol}}:\n (1, 41, :a)\n (2, 42, :b)\n (3, 43, :c)\n (4, 44, :d)\n (5, 45, :e)\n\njulia> z[1:2:end]\n([1, 3, 5], [41, 43, 45], [:a, :c, :e])\n\n\n\n\n\n","category":"method"}]
}
