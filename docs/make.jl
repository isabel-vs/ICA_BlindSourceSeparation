using ICA_BlindSourceSeparation
using Documenter

DocMeta.setdocmeta!(ICA_BlindSourceSeparation, :DocTestSetup, :(using ICA_BlindSourceSeparation); recursive=true)

makedocs(;
    modules=[ICA_BlindSourceSeparation],
    authors="Erik Felgendreher felgendreher@campus.tu-berlin.de",
    sitename="ICA_BlindSourceSeparation.jl",
    format=Documenter.HTML(;
        canonical="https://github.com/isabel-vs/ICA_BlindSourceSeparation.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Algorithms" => ["jade.md", "shibbs.md", "picard.md"],
        "Test Results" => "test_results.md"
    ],
)

deploydocs(;
    repo="github.com/isabel-vs/ICA_BlindSourceSeparation.jl",
    devbranch="main",
)
