"""Generate docs/examples.md with synthetic conversation samples."""

from __future__ import annotations

import json

from devsynth_generator.models import Conversation

from .doc_context import DocContext

# Predefined synthetic examples covering required scenarios.
# These are used when no real examples are available from the dataset.
SYNTHETIC_EXAMPLES: list[dict] = [
    {
        "title": "Beginner — Python Function Help",
        "difficulty": "easy",
        "category": "feature_request",
        "language": "python",
        "messages": [
            {"role": "user", "content": "I'm new to Python. How do I write a function that reverses a string?"},
            {"role": "assistant", "content": "Here's a simple approach using Python slicing:\n\n```python\ndef reverse_string(s: str) -> str:\n    return s[::-1]\n```\n\nThe `[::-1]` slice notation creates a reversed copy of the string. You can also use `reversed()` with `''.join()` if you prefer a more explicit approach."},
            {"role": "user", "content": "What if I want to reverse it in-place?"},
            {"role": "assistant", "content": "Strings in Python are immutable, so you can't reverse them in-place. However, you can convert to a list first:\n\n```python\ndef reverse_in_place(s: str) -> str:\n    chars = list(s)\n    chars.reverse()\n    return ''.join(chars)\n```\n\nFor most cases, the slice approach `s[::-1]` is more Pythonic and performant."},
        ],
    },
    {
        "title": "Intermediate — TypeScript Error Handling",
        "difficulty": "medium",
        "category": "debugging_session",
        "language": "typescript",
        "messages": [
            {"role": "user", "content": "My TypeScript API keeps returning 500 errors. The error says 'Cannot read properties of undefined'. How do I debug this?"},
            {"role": "assistant", "content": "This is a classic null reference error. Here's a systematic approach:\n\n1. **Check the stack trace** to identify the exact line.\n2. **Add optional chaining** where you access nested properties:\n\n```typescript\n// Before (unsafe)\nconst name = response.data.user.name;\n\n// After (safe)\nconst name = response?.data?.user?.name ?? 'Unknown';\n```\n\n3. **Add input validation** at your API boundary:\n\n```typescript\nif (!request.body?.userId) {\n  return res.status(400).json({ error: 'userId is required' });\n}\n```\n\nCan you share the specific stack trace?"},
            {"role": "user", "content": "It's happening in my middleware when parsing JWT tokens."},
            {"role": "assistant", "content": "That narrows it down. The issue is likely that the authorization header is missing or malformed. Add defensive checks:\n\n```typescript\nfunction extractToken(req: Request): string | null {\n  const header = req.headers.authorization;\n  if (!header?.startsWith('Bearer ')) {\n    return null;\n  }\n  return header.slice(7);\n}\n```\n\nAlso wrap your JWT verification in a try-catch to handle expired or invalid tokens gracefully instead of letting them crash the middleware."},
        ],
    },
    {
        "title": "Advanced — Rust Concurrency Optimization",
        "difficulty": "hard",
        "category": "refactor",
        "language": "rust",
        "messages": [
            {"role": "user", "content": "I have a Rust service processing 10K requests/sec but it's bottlenecked on a Mutex<HashMap>. How can I improve throughput?"},
            {"role": "assistant", "content": "A single `Mutex<HashMap>` is a classic bottleneck. Here are three approaches in order of complexity:\n\n**1. Use `DashMap` (easiest)**\n```rust\nuse dashmap::DashMap;\nlet map: DashMap<String, Value> = DashMap::new();\n// Lock-free concurrent reads, sharded writes\n```\n\n**2. Sharded locking**\n```rust\nconst SHARDS: usize = 64;\nlet shards: Vec<RwLock<HashMap<K, V>>> = (0..SHARDS)\n    .map(|_| RwLock::new(HashMap::new()))\n    .collect();\n\nfn get_shard(key: &K) -> usize {\n    hash(key) % SHARDS\n}\n```\n\n**3. Use `RwLock` instead of `Mutex`** if reads dominate writes.\n\nAt 10K req/s, `DashMap` should give you 3-5x throughput improvement with minimal code changes."},
            {"role": "user", "content": "We have about 80% reads, 20% writes. Would RwLock be enough?"},
            {"role": "assistant", "content": "With 80/20 read-write ratio, `RwLock` alone won't scale well because write locks still block all readers. `DashMap` is the better choice here — it uses sharded `RwLock`s internally, so concurrent readers on different shards never block each other.\n\nBenchmark both. If you need even more performance, consider:\n- **`evmap`** for near-zero-cost reads (eventually consistent)\n- **`flurry`** (concurrent hash map inspired by Java's `ConcurrentHashMap`)\n\nFor your workload, I'd start with `DashMap` and measure before adding complexity."},
        ],
    },
    {
        "title": "Debugging — Java NullPointerException",
        "difficulty": "medium",
        "category": "debugging_session",
        "language": "java",
        "messages": [
            {"role": "user", "content": "I'm getting a NullPointerException in my Spring Boot controller but the object should never be null. The repository.findById() call works in tests."},
            {"role": "assistant", "content": "Common causes when `findById()` works in tests but fails in production:\n\n1. **Transaction scope**: Your test might auto-commit, but the controller runs in a different transaction.\n2. **Lazy loading**: If you're accessing a lazy-loaded relation outside the transaction.\n3. **Optional handling**: `findById()` returns `Optional`, not the entity directly.\n\n```java\n// Wrong\nUser user = userRepository.findById(id).get(); // NPE if not found\n\n// Correct\nUser user = userRepository.findById(id)\n    .orElseThrow(() -> new ResponseStatusException(\n        HttpStatus.NOT_FOUND, \"User not found\"));\n```\n\nCheck if you're calling `.get()` on the Optional without checking presence first."},
            {"role": "user", "content": "That was it — I was using .get() directly. Thanks!"},
            {"role": "assistant", "content": "Glad that fixed it. As a best practice, consider adding a helper in your service layer:\n\n```java\npublic User getUserOrThrow(Long id) {\n    return userRepository.findById(id)\n        .orElseThrow(() -> new EntityNotFoundException(\n            \"User not found with id: \" + id));\n}\n```\n\nThis centralizes the error handling and gives you consistent 404 responses across all endpoints."},
        ],
    },
    {
        "title": "Code Review — Go HTTP Handler",
        "difficulty": "medium",
        "category": "code_review",
        "language": "go",
        "messages": [
            {"role": "user", "content": "Can you review this Go HTTP handler?\n\n```go\nfunc getUser(w http.ResponseWriter, r *http.Request) {\n    id := r.URL.Query().Get(\"id\")\n    user, err := db.GetUser(id)\n    if err != nil {\n        log.Println(err)\n        return\n    }\n    json.NewEncoder(w).Encode(user)\n}\n```"},
            {"role": "assistant", "content": "Several issues to address:\n\n1. **No error response**: When `err != nil`, you return without sending an HTTP error.\n2. **No input validation**: `id` could be empty.\n3. **No Content-Type header**: The client won't know it's JSON.\n4. **Encode error ignored**: `json.Encode` can fail.\n\nImproved version:\n\n```go\nfunc getUser(w http.ResponseWriter, r *http.Request) {\n    id := r.URL.Query().Get(\"id\")\n    if id == \"\" {\n        http.Error(w, \"missing id parameter\", http.StatusBadRequest)\n        return\n    }\n\n    user, err := db.GetUser(id)\n    if err != nil {\n        log.Printf(\"failed to get user %s: %v\", id, err)\n        http.Error(w, \"user not found\", http.StatusNotFound)\n        return\n    }\n\n    w.Header().Set(\"Content-Type\", \"application/json\")\n    if err := json.NewEncoder(w).Encode(user); err != nil {\n        log.Printf(\"failed to encode user %s: %v\", id, err)\n    }\n}\n```"},
        ],
    },
    {
        "title": "Optimization — Python Database Query",
        "difficulty": "hard",
        "category": "refactor",
        "language": "python",
        "messages": [
            {"role": "user", "content": "My Django view takes 8 seconds to load. It shows a list of 100 orders with customer names and product counts. The ORM query looks simple but the page is very slow."},
            {"role": "assistant", "content": "This is almost certainly the **N+1 query problem**. Your template probably accesses `order.customer.name` and `order.products.count()` for each order, triggering 200+ individual database queries.\n\n**Fix with `select_related` and `annotate`:**\n\n```python\nfrom django.db.models import Count\n\norders = (\n    Order.objects\n    .select_related('customer')  # JOIN instead of N queries\n    .annotate(product_count=Count('products'))  # Single aggregate\n    .order_by('-created_at')[:100]\n)\n```\n\nThis reduces 200+ queries to **1 query**. Enable Django Debug Toolbar to verify the query count before and after."},
            {"role": "user", "content": "That brought it down to 200ms. But some customers have thousands of orders — would pagination help?"},
            {"role": "assistant", "content": "Absolutely. Cursor-based pagination is ideal for large datasets:\n\n```python\nfrom django.core.paginator import Paginator\n\ndef order_list(request):\n    queryset = (\n        Order.objects\n        .select_related('customer')\n        .annotate(product_count=Count('products'))\n        .order_by('-created_at')\n    )\n    paginator = Paginator(queryset, 25)  # 25 per page\n    page = paginator.get_page(request.GET.get('page'))\n    return render(request, 'orders/list.html', {'page': page})\n```\n\nAlso add a database index on `created_at` if you haven't:\n\n```python\nclass Meta:\n    indexes = [models.Index(fields=['-created_at'])]\n```\n\nThis keeps response times under 50ms even with millions of orders."},
        ],
    },
    {
        "title": "System Design — Microservice Architecture",
        "difficulty": "hard",
        "category": "architecture_discussion",
        "language": "python",
        "messages": [
            {"role": "user", "content": "We're splitting our monolith into microservices. The user service and order service both need user data. How should they communicate?"},
            {"role": "assistant", "content": "There are three main patterns, each with tradeoffs:\n\n**1. Synchronous (HTTP/gRPC)**\n- Order service calls User service API\n- Simple, strong consistency\n- Creates runtime coupling — if User service is down, Order service fails\n\n**2. Event-driven (recommended)**\n- User service publishes events (UserCreated, UserUpdated)\n- Order service maintains a local read model\n- Eventual consistency, but no runtime coupling\n\n**3. Shared database (avoid)**\n- Both services read the same users table\n- Defeats the purpose of microservices\n\nFor your case, I'd recommend **event-driven** with a message broker like Kafka or RabbitMQ. The Order service keeps a slim `users` cache table that's updated via events."},
            {"role": "user", "content": "What about data consistency? What if a user is deleted but orders still reference them?"},
            {"role": "assistant", "content": "This is the key challenge of distributed systems. Options:\n\n1. **Soft delete** (recommended): Never physically delete users. Mark them as `inactive`. Orders always have a valid reference.\n\n2. **Saga pattern**: The delete triggers a multi-step process:\n   - User service marks user as `deleting`\n   - Order service anonymizes related orders\n   - User service completes deletion\n   - If any step fails, compensating actions roll back\n\n3. **Tombstone events**: Publish a `UserDeleted` event. Order service replaces user data with `[Deleted User]` in its local cache.\n\nSoft delete + event-driven updates gives you the best balance of simplicity and consistency for most cases."},
        ],
    },
]


def generate_examples(ctx: DocContext) -> str:
    """Produce docs/examples.md with conversation examples."""
    lines = [
        "# Example Conversations",
        "",
        f"Sample conversations from the **{ctx.dataset_name}** dataset (v{ctx.version}).",
        "",
        "> All examples below are **synthetically generated** and do not contain real user data.",
        "",
    ]

    # Use dataset examples if available, supplemented with synthetic ones.
    dataset_examples = _select_dataset_examples(ctx)
    if dataset_examples:
        lines.append("## From the Dataset")
        lines.append("")
        for i, conv in enumerate(dataset_examples, 1):
            lines.append(_render_conversation(
                f"Example {i} — {(conv.category or conv.task_type).replace('_', ' ').title()}",
                conv.difficulty,
                conv.category or conv.task_type,
                conv.language,
                [{"role": m.role, "content": m.content} for m in conv.messages],
            ))
        lines.append("")

    lines.append("## Curated Examples")
    lines.append("")
    for example in SYNTHETIC_EXAMPLES:
        lines.append(_render_conversation(
            example["title"],
            example["difficulty"],
            example["category"],
            example["language"],
            example["messages"],
        ))

    return "\n".join(lines) + "\n"


def _select_dataset_examples(ctx: DocContext) -> list[Conversation]:
    """Select diverse examples from the loaded dataset conversations."""
    if not ctx.examples:
        return []

    # Pick one example per difficulty level if possible.
    selected: list[Conversation] = []
    seen_difficulties: set[str] = set()
    for conv in ctx.examples:
        if conv.difficulty not in seen_difficulties and len(selected) < 3:
            selected.append(conv)
            seen_difficulties.add(conv.difficulty)
    return selected


def _render_conversation(
    title: str,
    difficulty: str,
    category: str,
    language: str,
    messages: list[dict],
) -> str:
    """Render a single conversation example as markdown."""
    lines = [
        f"### {title}",
        "",
        f"> **Category**: `{category}` · **Difficulty**: `{difficulty}` · **Language**: `{language}`",
        "",
    ]
    for msg in messages:
        role = msg["role"].capitalize()
        content = msg["content"]
        lines.append(f"**{role}**:")
        lines.append("")
        lines.append(content)
        lines.append("")
    return "\n".join(lines)
