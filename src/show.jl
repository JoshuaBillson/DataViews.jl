function Base.show(io::IO, ::MIME"text/plain", x::AbstractIterator)
    print(io, length(x), "-element ")
    _name(io, x)
    print(io, "\n  with first element:")
    print(io, "\n  ", _expanded_summary(first(x)))
end

function Base.show(io::IO, ::MIME"text/plain", e::DataLoader)
    if Base.haslength(e)
        print(io, length(e), "-element ")
    else
        print(io, "Unknown-length ")
    end
    print(io, "DataLoader")
    print(io, "\n  with first element:")
    print(io, "\n  ", _expanded_summary(first(e)))
end

_expanded_summary(x) = summary(x)
function _expanded_summary(xs::Tuple)
  parts = [_expanded_summary(x) for x in xs]
  "(" * join(parts, ", ") * (length(parts) == 1 ? ",)" : ")")
end
function _expanded_summary(xs::NamedTuple)
  parts = ["$k = "*_expanded_summary(x) for (k,x) in zip(keys(xs), xs)]
  "(; " * join(parts, ", ") * ")"
end

_name(io, ::ZippedView) = print(io, "ZippedView")
_name(io, ::ObsView) = print(io, "ObsView")
_name(io, ::MappedView) = print(io, "MappedView")
_name(io, ::BatchedView) = print(io, "BatchedView")
_name(io, ::JoinedView) = print(io, "JoinedView")
_name(io, ::CachedView) = print(io, "CachedView")
_name(io, x::Any) = Base.showarg(io, x, false)