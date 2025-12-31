"""
LLM prompts for different decision types.
"""

# System prompt for item evaluation
ITEM_EVALUATOR_SYSTEM = """You are an expert Diablo 2 item evaluator for Project Diablo 2 (PD2).
You have deep knowledge of item values, builds, and the current meta.

Your task is to evaluate items and determine if they should be kept or dropped.
Consider:
- Usefulness for the player's build (Blizzard Sorceress)
- Trade value in the current economy
- Rarity and roll quality
- Socket potential for runewords

Always respond with valid JSON."""

ITEM_EVALUATION_PROMPT = """Evaluate this Diablo 2 item for a Blizzard Sorceress:

Item Name: {item_name}
Item Type: {item_type}
Rarity: {rarity}
Stats:
{stats}

Respond with JSON:
{{
  "keep": true/false,
  "reason": "Brief explanation (1-2 sentences)",
  "value": "trash" | "charsi" | "self-use" | "trade-low" | "trade-mid" | "trade-high" | "gg",
  "priority": 1-10 (10 = highest priority to pick up)
}}"""

# System prompt for error recovery
ERROR_RECOVERY_SYSTEM = """You are a Diablo 2 bot assistant helping recover from unexpected situations.
The bot is running autonomously and has encountered a problem.

Your job is to analyze the situation and provide a simple recovery plan.
The bot can:
- Teleport in any direction
- Cast town portal
- Use potions (slots 1-4)
- Click on screen coordinates
- Press keyboard keys

Provide clear, actionable steps."""

ERROR_RECOVERY_PROMPT = """The bot is in an unexpected state and needs help recovering.

Current Situation:
- Health: {health_percent}%
- Mana: {mana_percent}%
- Location: {location}
- Last Actions: {last_actions}
- Current State: {current_state}
- Error/Issue: {error_description}

Screen observations:
{screen_description}

What should the bot do to recover? Provide a simple action plan.

Respond with JSON:
{{
  "diagnosis": "What you think went wrong",
  "severity": "low" | "medium" | "high" | "critical",
  "actions": [
    {{"action": "action_name", "params": {{}}, "reason": "why"}}
  ],
  "should_abort_run": true/false
}}

Available actions:
- teleport_direction: params={{direction: "up/down/left/right"}}
- cast_town_portal: params={{}}
- use_potion: params={{slot: 1-4}}
- click: params={{x: int, y: int}}
- press_key: params={{key: "string"}}
- wait: params={{seconds: float}}
- abort_run: params={{}}"""

# System prompt for strategy advisor
STRATEGY_ADVISOR_SYSTEM = """You are a Diablo 2 farming strategy advisor.
You help optimize farming routes and suggest improvements based on session data.

Consider:
- Run success rates
- Run completion times
- Character gear/build
- Item drops and their value
- Death frequency"""

STRATEGY_PROMPT = """Based on the current session data, suggest farming optimizations:

Session Statistics:
- Total Runs: {total_runs}
- Success Rate: {success_rate}
- Average Run Time: {avg_run_time}
- Deaths: {total_deaths}
- Items Found: {items_summary}

Current Configuration:
- Build: Blizzard Sorceress
- Enabled Runs: {enabled_runs}
- Health Threshold: {health_threshold}%
- Mana Threshold: {mana_threshold}%

Recent Issues:
{recent_issues}

Provide recommendations:
{{
  "assessment": "Overall assessment of performance",
  "recommendations": [
    {{"area": "area_name", "suggestion": "what to change", "priority": "high/medium/low"}}
  ],
  "suggested_run_order": ["run1", "run2"],
  "config_changes": {{
    "setting_name": "new_value"
  }}
}}"""

# System prompt for inventory management
INVENTORY_SYSTEM = """You are a Diablo 2 inventory management assistant.
Help decide what items to keep, stash, or drop when inventory is full.

Consider:
- Item value for trade
- Usefulness for current build (Blizzard Sorceress)
- Stash space availability
- Item rarity and uniqueness"""

INVENTORY_PROMPT = """Inventory is full. Help decide what to do with these items:

Current Inventory:
{inventory_items}

Stash Status:
- Personal Stash: {stash_slots_free} free slots
- Shared Stash: {shared_stash_free} free slots

Items on Ground (not picked up):
{ground_items}

For each inventory item, decide: keep, stash, or drop.
For ground items, decide: pickup (and drop something) or leave.

Respond with JSON:
{{
  "inventory_actions": [
    {{"item": "item_name", "action": "keep/stash/drop", "reason": "why"}}
  ],
  "ground_pickups": [
    {{"item": "item_name", "pickup": true/false, "drop_item": "item_to_drop_if_any"}}
  ],
  "total_value_assessment": "description of overall inventory value"
}}"""

# Quick item check (for common items that don't need full evaluation)
QUICK_ITEM_CHECK = {
    # Always keep
    "high_runes": ["ber", "jah", "lo", "ohm", "vex", "gul", "ist", "mal", "um", "pul"],
    "mid_runes": ["lem", "fal", "ko", "lum", "io", "hel", "dol", "shael", "sol", "amn"],

    # Always valuable uniques for trade
    "valuable_uniques": [
        "shako", "harlequin", "oculus", "skin of the vipermagi", "vipermagi",
        "war traveler", "chance guards", "goldwrap", "magefist", "frostburn",
        "tal rasha", "immortal king", "griswold", "mavina", "natalya",
        "arachnid mesh", "verdungo", "string of ears", "thundergod",
    ],

    # Good bases for runewords
    "good_bases": [
        "monarch", "archon plate", "dusk shroud", "mage plate", "wire fleece",
        "eth", "superior",  # prefixes
        "4 socket", "3 socket",  # for runewords
    ],

    # Skip these
    "always_skip": [
        "cracked", "damaged", "crude", "low quality",
        "stamina", "antidote", "thawing",  # weak potions
    ],
}


def should_quick_pickup(item_text: str) -> bool:
    """Quick check if item should definitely be picked up."""
    text_lower = item_text.lower()

    # Check high-value items
    for rune in QUICK_ITEM_CHECK["high_runes"]:
        if rune in text_lower and "rune" in text_lower:
            return True

    for unique in QUICK_ITEM_CHECK["valuable_uniques"]:
        if unique in text_lower:
            return True

    return False


def should_quick_skip(item_text: str) -> bool:
    """Quick check if item should definitely be skipped."""
    text_lower = item_text.lower()

    for skip in QUICK_ITEM_CHECK["always_skip"]:
        if skip in text_lower:
            return True

    return False
