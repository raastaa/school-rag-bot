"""Central configuration constants for retrieval and generation."""

# Retrieval parameters
TOP_K_INITIAL = 12
TOP_K_MAX = 24
MIN_UNIQUE_SOURCES = 4
MMR_LAMBDA = 0.6
RERANK_ALPHA = 0.65
HYDE_N = 3
NEIGHBOR_RADIUS = 2
PERCENTILE_CUT = 0.70
MIN_RESULTS_FLOOR = 10

# Generation budgets
BRIEF_TOKEN_BUDGET = 850
FULL_TOKEN_BUDGET = 2000

# Concurrency and caching
MAX_CONCURRENCY = 8
LOCAL_SEARCH_CACHE_TTL_SEC = 60
