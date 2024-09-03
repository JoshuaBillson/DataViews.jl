using DataViews
using Documenter

DocMeta.setdocmeta!(DataViews, :DocTestSetup, :(using DataViews); recursive=true)

makedocs(;
    modules=[DataViews],
    authors="Joshua Billson",
    sitename="DataViews.jl",
    format=Documenter.HTML(;
        canonical="https://JoshuaBillson.github.io/DataViews.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/JoshuaBillson/DataViews.jl",
    devbranch="main",
)
