using DataViews
using Test

using Random
using StableRNGs

const rng = StableRNG(123)

@testset "DataViews.jl" begin
    # Test Data
    x1 = collect(1:10)
    x2 = collect(21:30)
    v1 = ObsView(1:10, 1:10)
    v2 = ObsView(21:30, 1:10)

    # zipobs
    @test all(collect(zipobs(v1, v2)) .== collect(zip(x1, x2)))  # Test zipobs

    # repeatobs
    @test all(repeatobs(v1, 5) .== reduce(vcat, [x1 for _ in 1:5]))  # Test repeatobs
    @test all(repeatobs(zipobs(v1, v2), 2) .== reduce(vcat, [collect(zip(x1, x2)) for _ in 1:2]))  # zipobs + repeatobs

    # splitobs with shuffle
    @test first(splitobs(StableRNGs.StableRNG(123), 1:10, at=0.7)) == [4, 7, 2, 1, 3, 8, 5]
    @test last(splitobs(StableRNGs.StableRNG(123), 1:10, at=0.7)) == [6, 10, 9]
    @test splitobs(StableRNGs.StableRNG(123), 1:10, at=[0.3, 0.5])[1] == [4, 7, 2]
    @test splitobs(StableRNGs.StableRNG(123), 1:10, at=[0.3, 0.5])[2] == [1, 3, 8, 5, 6]
    @test splitobs(StableRNGs.StableRNG(123), 1:10, at=[0.3, 0.5])[3] == [10, 9]
    @test_throws ArgumentError splitobs(1:10, at=[0.3, 0.8])
    @test map(length, splitobs(1:10, at=[0.3, 0.7])) == [3, 7, 0]

    # splitobs without shuffle
    @test first(splitobs(StableRNGs.StableRNG(123), 1:10, at=0.7; shuffle=false)) == [1, 2, 3, 4, 5, 6, 7]
    @test last(splitobs(StableRNGs.StableRNG(123), 1:10, at=0.7; shuffle=false)) == [8, 9, 10]
    @test splitobs(StableRNGs.StableRNG(123), 1:10, at=[0.3, 0.5]; shuffle=false)[1] == [1, 2, 3]
    @test splitobs(StableRNGs.StableRNG(123), 1:10, at=[0.3, 0.5]; shuffle=false)[2] == [4, 5, 6, 7, 8]
    @test splitobs(StableRNGs.StableRNG(123), 1:10, at=[0.3, 0.5]; shuffle=false)[3] == [9, 10]

    # takeobs
    @test all(takeobs(v1, [2, 5, 8, 9]) .== x1[[2, 5, 8, 9]])  # takeobs
    @test_throws ArgumentError takeobs(v1, [0, 1, 2])

    # dropobs
    @test all(dropobs(v1, [1,2,3,5,6,8,9,10]) .== x1[[4,7]])  # dropobs

    # filterobs
    @test all(filterobs(iseven,  1:10) .== [2, 4, 6, 8, 10])
    @test all(filterobs(iseven,  v1) .== [2, 4, 6, 8, 10])

    # mapobs
    @test all(mapobs(x -> x * 2 + 1, v1) .== [3, 5, 7, 9, 11, 13, 15, 17, 19, 21])

    # sampleobs
    @test all(sampleobs(StableRNGs.StableRNG(126), v2, 4) .== [25, 24, 30, 26])
    @test length(sampleobs(v1, 0)) == 0
    @test_throws ArgumentError sampleobs(v2, -1)
    @test_throws ArgumentError sampleobs(v2, length(v2) + 1)

    # shuffleobs
    @test all(shuffleobs(StableRNGs.StableRNG(123), v1) .== [4, 7, 2, 1, 3, 8, 5, 6, 10, 9])
end
