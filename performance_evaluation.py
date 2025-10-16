"""
Performance Evaluation Script for your closet

This script evaluates the latency and quality metrics for both:
- Tab 2: Build Outfit (form-based outfit generation)
- Tab 3: Chat with Stylist (conversational styling advice)

Usage:
    python performance_evaluation.py [--tab2] [--tab3] [--output-dir results]

Examples:
    python performance_evaluation.py                    # Run both evaluations
    python performance_evaluation.py --tab2             # Only Tab 2
    python performance_evaluation.py --tab3             # Only Tab 3
    python performance_evaluation.py --output-dir logs  # Save to logs/
"""

import time
import pandas as pd
import numpy as np
import argparse
import sys
import os
from datetime import datetime
from pathlib import Path


project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Verify we can import from src/
if not (project_root / "src").exists():
    print(f"ERROR: Cannot find src/ directory at {project_root}")
    print("Please ensure performance_evaluation.py is in the repo root")
    sys.exit(1)

from src.app.wardrobe_app import generate_outfit, chat_response, classify_query_intent
from src.retrieval.gemini_rag import make_client


# ============================================================================
# TEST DATA
# ============================================================================

def create_test_wardrobe():
    """Create test wardrobe for evaluation"""
    return pd.DataFrame({
        "Item": ["T-shirt", "Jeans", "Blazer", "Sneakers", "Dress", "Cardigan",
                 "Trousers", "Shirt", "Loafers", "Coat", "Sweater", "Skirt",
                 "Shorts", "Sandals", "Boots", "Scarf"],
        "Color": ["White", "Blue", "Navy", "White", "Black", "Gray",
                  "Charcoal", "Light Blue", "Brown", "Camel", "Burgundy", "Beige",
                  "Khaki", "Tan", "Brown", "Gray"],
        "Pattern": ["Solid"]*16,
        "Material": ["Cotton", "Denim", "Wool", "Leather", "Silk", "Cotton",
                     "Wool", "Cotton", "Leather", "Wool", "Cashmere", "Linen",
                     "Cotton", "Leather", "Leather", "Wool"]
    })


TAB2_SCENARIOS = [
    {"name": "Business Casual Winter", "occasion": "Business Casual", "season": "Winter", "city": "New York", "weather_data": ""},
    {"name": "Casual Spring", "occasion": "Casual", "season": "Spring", "city": "", "weather_data": ""},
    {"name": "Date Night Clear Weather", "occasion": "Date Night", "season": "(None - Use Weather Only)", "city": "Los Angeles", "weather_data": "Location: Los Angeles (Current: 22C, Clear)"},
    {"name": "Job Interview Rain", "occasion": "Job Interview", "season": "Fall", "city": "Seattle", "weather_data": "Location: Seattle (Current: 13C, Rainy)"},
    {"name": "Wedding Guest Summer", "occasion": "Wedding Guest", "season": "Summer", "city": "Miami", "weather_data": ""},
    {"name": "Gym Workout", "occasion": "Gym", "season": "Spring", "city": "", "weather_data": ""},
    {"name": "Casual Cold Snow", "occasion": "Casual", "season": "Winter", "city": "Chicago", "weather_data": "Location: Chicago (Current: -5C, Snow)"},
    {"name": "Business Formal", "occasion": "Business Formal", "season": "Fall", "city": "Boston", "weather_data": ""},
    {"name": "Brunch Spring", "occasion": "Brunch", "season": "Spring", "city": "", "weather_data": ""},
    {"name": "Date Night Summer Hot", "occasion": "Date Night", "season": "Summer", "city": "Austin", "weather_data": "Location: Austin (Current: 35C, Sunny)"},
    {"name": "Smart Casual Fall", "occasion": "Smart Casual", "season": "Fall", "city": "", "weather_data": ""},
    {"name": "Cocktail Party Winter", "occasion": "Cocktail Party", "season": "Winter", "city": "New York", "weather_data": ""},
    {"name": "Beach Vacation", "occasion": "Beach", "season": "Summer", "city": "San Diego", "weather_data": "Location: San Diego (Current: 28C, Clear)"},
    {"name": "Coffee Meeting Casual", "occasion": "Coffee Meeting", "season": "Spring", "city": "", "weather_data": ""},
    {"name": "Dinner Party Fall", "occasion": "Dinner Party", "season": "Fall", "city": "Portland", "weather_data": "Location: Portland (Current: 15C, Rainy)"},
    {"name": "Job Interview Summer", "occasion": "Job Interview", "season": "Summer", "city": "", "weather_data": ""},
    {"name": "Casual Winter Cold", "occasion": "Casual", "season": "Winter", "city": "Minneapolis", "weather_data": "Location: Minneapolis (Current: -10C, Snow)"},
    {"name": "Business Casual Spring", "occasion": "Business Casual", "season": "Spring", "city": "", "weather_data": ""},
    {"name": "Wedding Guest Fall", "occasion": "Wedding Guest", "season": "Fall", "city": "Nashville", "weather_data": ""},
    {"name": "Date Night Fall Cool", "occasion": "Date Night", "season": "Fall", "city": "San Francisco", "weather_data": "Location: San Francisco (Current: 16C, Cloudy)"},
]

TAB3_SCENARIOS = [
    {"name": "Outfit Request Brunch", "message": "I need a casual outfit for brunch this weekend", "expected_intent": "styling"},
    {"name": "Color Question Navy", "message": "What colors go well with navy?", "expected_intent": "styling"},
    {"name": "Specific Item Blazer Jeans", "message": "Can I wear my navy blazer with jeans for a dinner date?", "expected_intent": "styling"},
    {"name": "Weather Rain Meeting", "message": "It's raining today and I have a meeting. What should I wear?", "expected_intent": "styling"},
    {"name": "Styling Advice Boring", "message": "How do I make my white t-shirt and blue jeans look less boring?", "expected_intent": "styling"},
    {"name": "What to Wear Interview", "message": "What should I wear to a job interview at a tech startup?", "expected_intent": "styling"},
    {"name": "What to Wear Wedding", "message": "What should I wear to a summer wedding?", "expected_intent": "styling"},
    {"name": "Fashion History LBD", "message": "What is the history of the little black dress?", "expected_intent": "knowledge"},
    {"name": "Fashion Theory Chanel", "message": "How did Coco Chanel influence modern fashion?", "expected_intent": "knowledge"},
    {"name": "Garment Construction Fabric", "message": "What is the difference between woven and knit fabrics?", "expected_intent": "knowledge"},
    {"name": "Layering Fall", "message": "How should I layer clothes for fall weather?", "expected_intent": "styling"},
    {"name": "First Date Outfit", "message": "I have a first date tonight at a nice restaurant. Help me pick an outfit", "expected_intent": "styling"},
    {"name": "Color Pairing Burgundy Gray", "message": "Does burgundy go with gray?", "expected_intent": "styling"},
    {"name": "Material Cold Weather", "message": "Is wool or cotton better for cold weather?", "expected_intent": "styling"},
    {"name": "Follow-up Shoes", "message": "Can you suggest shoes to go with that?", "expected_intent": "styling"},
    {"name": "Business Casual Help", "message": "I need help putting together a business casual look", "expected_intent": "styling"},
    {"name": "Dress Code Cocktail", "message": "What does cocktail attire mean?", "expected_intent": "styling"},
    {"name": "Pattern Mixing", "message": "Can I mix stripes and florals?", "expected_intent": "styling"},
    {"name": "Accessory Advice", "message": "What accessories should I wear with a black dress?", "expected_intent": "styling"},
    {"name": "Fashion Movement History", "message": "What was the impact of punk fashion on mainstream style?", "expected_intent": "knowledge"},
]


# ============================================================================
# QUALITY METRICS
# ============================================================================

def calculate_wardrobe_adherence(output_text, wardrobe_df):
    """Check if specific items mentioned in output exist in wardrobe."""
    output_lower = output_text.lower()

    wardrobe_items = []
    for _, row in wardrobe_df.iterrows():
        color = row["Color"].lower()
        item = row["Item"].lower()
        wardrobe_items.append(f"{color} {item}")
        wardrobe_items.append(item)

    wardrobe_items = list(set(wardrobe_items))

    mentioned_items = []
    for wardrobe_item in wardrobe_items:
        if wardrobe_item in output_lower:
            mentioned_items.append(wardrobe_item)

    mentioned_items = list(set(mentioned_items))

    clothing_types = ["shirt", "t-shirt", "tee", "jeans", "trouser", "pants",
                     "blazer", "jacket", "dress", "coat", "sweater", "cardigan",
                     "shoe", "sneaker", "loafer", "boot", "skirt", "short",
                     "sandal", "scarf"]

    total_mentioned = sum(1 for item_type in clothing_types if item_type in output_lower)
    has_exception = "suggestion (missing category)" in output_lower or "suggestion (gap)" in output_lower

    if total_mentioned == 0:
        return 0, 0, 0.0, has_exception

    adherence = len(mentioned_items) / total_mentioned
    return len(mentioned_items), total_mentioned, adherence, has_exception


def check_outfit_completeness(output_text):
    """Check if outfit includes top, bottom, and shoes."""
    output_lower = output_text.lower()
    has_dress = "dress" in output_lower

    has_top = has_dress or any(word in output_lower for word in [
        "shirt", "blouse", "t-shirt", "tee", "top", "sweater",
        "cardigan", "blazer", "jacket", "coat"
    ])

    has_bottom = has_dress or any(word in output_lower for word in [
        "jeans", "trouser", "pants", "skirt", "short"
    ])

    has_shoes = any(word in output_lower for word in [
        "shoe", "sneaker", "loafer", "boot", "sandal", "heel"
    ])

    is_complete = has_top and has_bottom and has_shoes
    return has_top, has_bottom, has_shoes, is_complete


def check_weather_mention(output_text, weather_data):
    """Check if output mentions weather when weather data was provided."""
    if not weather_data or weather_data.strip() == "":
        return False, False

    output_lower = output_text.lower()
    weather_keywords = ["weather", "temperature", "rain", "cold", "hot", "warm",
                       "sunny", "cloudy", "layer", "waterproof", "breathable",
                       "degrees", "°", "rainy", "snow"]

    mentions_weather = any(keyword in output_lower for keyword in weather_keywords)
    return mentions_weather, True


# ============================================================================
# RATE LIMITING
# ============================================================================

class RateLimitSafeEvaluator:
    """Wrapper that ensures we don't hit API rate limits"""

    def __init__(self, requests_per_minute=8):
        """
        Args:
            requests_per_minute: Conservative limit (Gemini free tier is 10/min)
        """
        self.requests_per_minute = requests_per_minute
        self.min_delay = 60.0 / requests_per_minute
        self.last_request_time = 0

    def wait_if_needed(self):
        """Wait if we're going too fast"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_delay:
            sleep_time = self.min_delay - time_since_last
            print(f"    ⏱️  Rate limit protection: waiting {sleep_time:.1f}s...")
            time.sleep(sleep_time)

        self.last_request_time = time.time()


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_tab2(wardrobe, scenarios, rate_limiter):
    """Evaluate Tab 2: Build Outfit"""

    results = []

    print(f"\n{'='*80}")
    print("TAB 2: BUILD OUTFIT EVALUATION")
    print(f"{'='*80}")
    print(f"Testing {len(scenarios)} scenarios...")
    print(f"Estimated time: ~{len(scenarios) * 7.5 / 60:.1f} minutes\n")

    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] {scenario['name']}")

        rate_limiter.wait_if_needed()

        try:
            start_time = time.time()
            first_token_time = None
            full_output = ""

            generator = generate_outfit(
                wardrobe_df=wardrobe,
                occasion=scenario["occasion"],
                season=scenario["season"],
                city=scenario["city"],
                selected_items=[],
                custom_occasion="",
                weather_data=scenario["weather_data"],
                previous_outfits=[]
            )

            for chunk in generator:
                if first_token_time is None:
                    first_token_time = time.time()
                full_output = chunk

            end_time = time.time()

            # Calculate metrics
            ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
            total_time = (end_time - start_time) * 1000

            items_from_wardrobe, total_items, adherence, has_exception = \
                calculate_wardrobe_adherence(full_output, wardrobe)
            has_top, has_bottom, has_shoes, is_complete = \
                check_outfit_completeness(full_output)
            mentions_weather, weather_provided = \
                check_weather_mention(full_output, scenario["weather_data"])

            print(f"    ✓ TTFT: {ttft:.0f}ms | Total: {total_time:.0f}ms | "
                  f"Adherence: {adherence:.0%} | Complete: {is_complete}")

            results.append({
                "scenario": scenario["name"],
                "occasion": scenario["occasion"],
                "season": scenario["season"],
                "has_weather": bool(scenario["weather_data"]),
                "ttft_ms": ttft,
                "total_ms": total_time,
                "output_chars": len(full_output),
                "wardrobe_adherence": adherence,
                "items_from_wardrobe": items_from_wardrobe,
                "total_items_mentioned": total_items,
                "has_exception": has_exception,
                "is_complete": is_complete,
                "has_top": has_top,
                "has_bottom": has_bottom,
                "has_shoes": has_shoes,
                "weather_provided": weather_provided,
                "mentions_weather": mentions_weather if weather_provided else None,
                "error": None
            })

        except Exception as e:
            print(f"    ✗ ERROR: {str(e)[:100]}")
            results.append({
                "scenario": scenario["name"],
                "error": str(e)
            })

    return pd.DataFrame(results)


def evaluate_tab3(wardrobe, scenarios, rate_limiter, client):
    """Evaluate Tab 3: Chat with Stylist"""

    results = []

    print(f"\n{'='*80}")
    print("TAB 3: CHAT WITH STYLIST EVALUATION")
    print(f"{'='*80}")
    print(f"Testing {len(scenarios)} scenarios...")
    print(f"Estimated time: ~{len(scenarios) * 15 / 60:.1f} minutes\n")

    for i, scenario in enumerate(scenarios, 1):
        print(f"[{i}/{len(scenarios)}] {scenario['name']}")

        rate_limiter.wait_if_needed()

        try:
            # Classification
            intent_start = time.time()
            detected_intent = classify_query_intent(client, scenario["message"])
            intent_time = (time.time() - intent_start) * 1000

            intent_correct = (detected_intent == scenario["expected_intent"])

            # Wait before chat response
            rate_limiter.wait_if_needed()

            # Chat response
            start_time = time.time()
            first_token_time = None
            full_output = ""

            generator = chat_response(
                message=scenario["message"],
                history=[],
                wardrobe_df=wardrobe
            )

            for chunk in generator:
                if first_token_time is None:
                    first_token_time = time.time()
                full_output = chunk

            end_time = time.time()

            # Calculate metrics
            ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
            total_time = (end_time - start_time) * 1000

            status = "✓" if intent_correct else "✗"
            print(f"    {status} Intent: {detected_intent} (expected {scenario['expected_intent']}) | "
                  f"Classification: {intent_time:.0f}ms | Total: {total_time:.0f}ms")

            results.append({
                "scenario": scenario["name"],
                "message": scenario["message"],
                "expected_intent": scenario["expected_intent"],
                "detected_intent": detected_intent,
                "intent_correct": intent_correct,
                "classification_ms": intent_time,
                "ttft_ms": ttft,
                "total_ms": total_time,
                "output_chars": len(full_output),
                "error": None
            })

        except Exception as e:
            print(f"    ✗ ERROR: {str(e)[:100]}")
            results.append({
                "scenario": scenario["name"],
                "error": str(e)
            })

    return pd.DataFrame(results)


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def print_summary(tab2_df, tab3_df):
    """Print summary statistics"""

    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}\n")

    if tab2_df is not None:
        print("TAB 2: BUILD OUTFIT")
        print("-" * 80)
        successful = tab2_df[tab2_df['error'].isna()]

        if len(successful) > 0:
            print(f"Successful scenarios: {len(successful)}/{len(tab2_df)}")
            print(f"\nLatency Metrics:")
            print(f"  TTFT: {successful['ttft_ms'].mean():.0f} ± {successful['ttft_ms'].std():.0f} ms")
            print(f"  Total Time: {successful['total_ms'].mean():.0f} ± {successful['total_ms'].std():.0f} ms "
                  f"({successful['total_ms'].mean()/1000:.2f} sec)")

            print(f"\nQuality Metrics:")
            print(f"  Wardrobe Adherence: {successful['wardrobe_adherence'].mean():.1%}")
            print(f"  Outfit Completeness: {successful['is_complete'].sum() / len(successful):.1%} "
                  f"({successful['is_complete'].sum()}/{len(successful)})")
            print(f"    - Has Top: {successful['has_top'].sum() / len(successful):.1%}")
            print(f"    - Has Bottom: {successful['has_bottom'].sum() / len(successful):.1%}")
            print(f"    - Has Shoes: {successful['has_shoes'].sum() / len(successful):.1%}")

            weather_cases = successful[successful['weather_provided'] == True]
            if len(weather_cases) > 0:
                print(f"  Weather Mention Rate: {weather_cases['mentions_weather'].sum() / len(weather_cases):.1%} "
                      f"({weather_cases['mentions_weather'].sum()}/{len(weather_cases)})")

            exception_rate = successful['has_exception'].sum() / len(successful)
            print(f"  Exception Rate: {exception_rate:.1%} "
                  f"({successful['has_exception'].sum()}/{len(successful)})")
        else:
            print("No successful scenarios")

    if tab3_df is not None:
        print(f"\n{'='*80}")
        print("TAB 3: CHAT WITH STYLIST")
        print("-" * 80)
        successful = tab3_df[tab3_df['error'].isna()]

        if len(successful) > 0:
            print(f"Successful scenarios: {len(successful)}/{len(tab3_df)}")
            print(f"\nLatency Metrics:")
            print(f"  Classification: {successful['classification_ms'].mean():.0f} ± {successful['classification_ms'].std():.0f} ms")
            print(f"  TTFT: {successful['ttft_ms'].mean():.0f} ± {successful['ttft_ms'].std():.0f} ms")
            print(f"  Total Time: {successful['total_ms'].mean():.0f} ± {successful['total_ms'].std():.0f} ms "
                  f"({successful['total_ms'].mean()/1000:.2f} sec)")

            print(f"\nQuality Metrics:")
            print(f"  Intent Classification Accuracy: {successful['intent_correct'].sum() / len(successful):.1%} "
                  f"({successful['intent_correct'].sum()}/{len(successful)})")

            styling_queries = successful[successful['expected_intent'] == 'styling']
            knowledge_queries = successful[successful['expected_intent'] == 'knowledge']

            if len(styling_queries) > 0:
                styling_accuracy = styling_queries['intent_correct'].sum() / len(styling_queries)
                print(f"  Styling Query Accuracy: {styling_accuracy:.1%} "
                      f"({styling_queries['intent_correct'].sum()}/{len(styling_queries)})")

            if len(knowledge_queries) > 0:
                knowledge_accuracy = knowledge_queries['intent_correct'].sum() / len(knowledge_queries)
                print(f"  Knowledge Query Accuracy: {knowledge_accuracy:.1%} "
                      f"({knowledge_queries['intent_correct'].sum()}/{len(knowledge_queries)})")
        else:
            print("No successful scenarios")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate AI Styling Assistant Performance")
    parser.add_argument('--tab2', action='store_true', help='Run Tab 2 evaluation only')
    parser.add_argument('--tab3', action='store_true', help='Run Tab 3 evaluation only')
    parser.add_argument('--output-dir', default='results', help='Output directory for results')
    parser.add_argument('--requests-per-minute', type=int, default=8,
                       help='API requests per minute (default: 8 for safety)')

    args = parser.parse_args()

    # If neither specified, run both
    run_tab2 = args.tab2 or not args.tab3
    run_tab3 = args.tab3 or not args.tab2

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Initialize
    print(f"\n{'='*80}")
    print("PERFORMANCE EVALUATION - AI STYLING ASSISTANT")
    print(f"{'='*80}")
    print(f"Output directory: {output_dir}")
    print(f"Rate limit: {args.requests_per_minute} requests/minute")

    wardrobe = create_test_wardrobe()
    client = make_client()
    rate_limiter = RateLimitSafeEvaluator(requests_per_minute=args.requests_per_minute)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    tab2_df = None
    tab3_df = None

    # Run evaluations
    try:
        if run_tab2:
            tab2_df = evaluate_tab2(wardrobe, TAB2_SCENARIOS, rate_limiter)
            output_file = output_dir / f"tab2_results_{timestamp}.csv"
            tab2_df.to_csv(output_file, index=False)
            print(f"\n✓ Tab 2 results saved: {output_file}")

        if run_tab3:
            tab3_df = evaluate_tab3(wardrobe, TAB3_SCENARIOS, rate_limiter, client)
            output_file = output_dir / f"tab3_results_{timestamp}.csv"
            tab3_df.to_csv(output_file, index=False)
            print(f"\n✓ Tab 3 results saved: {output_file}")

        # Print summary
        print_summary(tab2_df, tab3_df)

        print(f"\n{'='*80}")
        print("EVALUATION COMPLETE")
        print(f"{'='*80}\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        if tab2_df is not None:
            output_file = output_dir / f"tab2_results_partial_{timestamp}.csv"
            tab2_df.to_csv(output_file, index=False)
            print(f"Partial Tab 2 results saved: {output_file}")
        if tab3_df is not None:
            output_file = output_dir / f"tab3_results_partial_{timestamp}.csv"
            tab3_df.to_csv(output_file, index=False)
            print(f"Partial Tab 3 results saved: {output_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()