jax.lax.scan compiles a loop. The unroll parameter tells XLA: "instead of looping N times, paste the loop body K times per iteration and loop N/K times."

Concrete example with your GAE scan (NUM_STEPS = 20):

unroll=1: XLA emits 1 copy of the GAE body, loops 20 times. Minimal compile IR, but each iteration pays loop overhead (check condition, update counter, branch).

unroll=4: XLA emits 4 copies of the GAE body sequentially, loops 5 times. The 4 copies are straight-line code with no branching between them — the hardware can pipeline/fuse them. Still only 5 branch points.

unroll=20 (full): XLA emits 20 copies — no loop at all, just straight-line code. Zero loop overhead, but the HLO IR is 20× larger, which makes the XLA compiler's optimization passes (CSE, fusion, scheduling) work harder and take longer.

The tradeoff:

Compile time	Runtime speed	IR size
unroll=1	Fastest	Slightly slower (loop overhead)	Smallest
unroll=4	Fast	Near-optimal	Small
unroll=20	Slowest	Optimal (no branches)	20× body
For NUM_STEPS=20, unroll=4 hits the sweet spot — compile stays fast, runtime is nearly identical to full unroll, because 5 loop iterations is very cheap.

The reason unroll=16 was problematic with your old NUM_STEPS=200 is that the GAE body got pasted 16 times, and the compiler still had to optimize that large IR across 13 loop iterations — worst of both worlds.