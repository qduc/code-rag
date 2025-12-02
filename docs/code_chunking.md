Do I need to keep overlap when I chunk code by syntax for RAG?

Short answer: **no**. We removed chunk overlap from Code-RAG because it added complexity, triggered infinite-loop edge cases, and duplicated a lot of text without reliably improving recall. Instead, we lean on three mechanisms that preserve context without copying bytes: syntax-aware chunking, file headers, and adjacency metadata.

### Why we dropped overlap

- Overlap made `_split_large_segment` prone to "position not advancing" failures whenever the remainder of a chunk was shorter than the overlap size.
- Duplicated text increased storage costs and made embedding vectors less discriminative (multiple nearly identical chunks) while still failing on long-range references.
- We already track `prev_id`/`next_id`, line numbers, and file headers, which are cheaper and easier to reason about than hidden overlap heuristics.

### How we preserve context now

1. **Semantic chunks first.** Tree-sitter splits by functions, classes, and other definitions so most references stay within a single chunk.
2. **Include headers when needed.** Imports, constants, and file-level comments can be stored once as a dedicated "file header" chunk and fetched alongside results.
3. **Use adjacency metadata.** When a query returns chunk _i_, fetch `i-1` and `i+1` via `prev_id`/`next_id` rather than relying on overlapped text.
4. **Preserve signatures on split functions.** `_split_large_segment` automatically prepends the function signature/docstring to continuation chunks so follow-up chunks still have the necessary context.
5. **Hybrid retrieval.** Combine chunk-level vectors with optional whole-file vectors or reranking to reassemble broader answers.

### Retrieval tip

When building a response, request the top‑k chunks (k≈3–8) and include their adjacent chunks if the answer spans boundaries. This gives you continuity without storing duplicate content.

### Example

```py
def compute(x):
    """Compute something useful."""
    return helper(x) * 2

def helper(x):
    return x + 1
```

If `compute` exceeded the chunk size, the continuation chunk would start with `# [continued from above]` plus the `def compute` signature stub so the downstream model still knows which symbol it is looking at. No explicit overlap is required.

Bottom line: Code-RAG keeps chunks lean, avoids duplicated overlap, and relies on structural metadata to stitch context back together.
