from flask import Flask, request, jsonify
from pymongo import MongoClient
from bson import ObjectId
import torch
import torch.nn as nn
from IPython.display import Image, display
from collections import defaultdict
import numpy as np
from flask_cors import CORS  
import os


# MongoDB Configuration
MONGODB_URI = "mongodb+srv://AhmadJabbar:0uU29STyRwhoxV0X@shopsavvy.xaqy1.mongodb.net/"
DATABASE_NAME = "test"
RECENT_COLLECTION = "filters"
AGGREGATED_COLLECTION = "aggregatedfilters"
PRODUCTS_COLLECTION = "products"

# Connect to MongoDB
client = MongoClient(MONGODB_URI)
db = client[DATABASE_NAME]

BRAND_VOCAB = ['LAMA', 'Outfitters']
COLOR_VOCAB = ['Beige', 'Black', 'Blue', 'Brown', 'Green', 'Grey', 'Multi-color', 'Orange', 'Other', 'Pink', 'Purple', 'Red', 'White', 'Yellow']
SIZE_VOCAB = ['XS', 'S', 'M', 'L', 'XL', '2XL', '3XL', '4XL', '24', '26', '28', '30', '32', '33', '34', '36', '40']

# Function to compute price ranges based on conditions
def compute_price_range(min_prices, max_prices, default_min=0, default_max=50000):
    clean_min_prices = [p for p in min_prices if p > 0]
    clean_max_prices = [p for p in max_prices if p < default_max]

    if len(clean_min_prices) >= 3:
        final_min_price = int(np.percentile(clean_min_prices, 10))
    else:
        final_min_price = default_min

    if len(clean_max_prices) >= 3:
        final_max_price = int(np.percentile(clean_max_prices, 90))
    else:
        final_max_price = default_max

    return final_min_price, final_max_price

def merge_filters(recent, aggregated, key, recent_count, aggregated_count, recent_weight=0.75, aggregated_weight=0.25):
    merged = {}
    all_keys = set(recent[key].keys()) | set(aggregated[key].keys())

    for k in all_keys:
        recent_val = recent[key].get(k, 0) / recent_count
        aggregated_val = aggregated[key].get(k, 0) / aggregated_count

        final_score = round(recent_val * recent_weight + aggregated_val * aggregated_weight, 4)
        if final_score > 0:
            merged[k] = final_score

    return merged

def combine_price_ranges(aggregated_min, aggregated_max, recent_min, recent_max):
    if aggregated_min < recent_min:
        final_min = int(recent_min * 0.9)
    else:
        final_min = int(recent_min * 1.05)

    if aggregated_max > recent_max:
        final_max = int(recent_max * 1.1)
    else:
        final_max = int(recent_max * 0.95)

    return final_min, final_max

# Function to build the filter vector
def build_filter_vector(filter_dict):
    vector = []
    
    # --- Brands ---
    for brand in BRAND_VOCAB:
        vector.append(filter_dict.get('brandFilters', {}).get(brand, 0.0))
    
    # --- Colors ---
    for color in COLOR_VOCAB:
        vector.append(filter_dict.get('colorFilters', {}).get(color, 0.0))
    
    # --- Sizes ---
    raw_size_scores = filter_dict.get('sizeFilters', {})
    for size in SIZE_VOCAB:
        vector.append(raw_size_scores.get(size, 0.0))
    
    # --- Price ---
    vector.append(normalize_price(filter_dict.get('minPrice', 0)))
    vector.append(normalize_price(filter_dict.get('maxPrice', 20000)))

    return vector

def normalize_price(value, min_possible=0, max_possible=20000):
    return (value - min_possible) / (max_possible - min_possible)

# Function to fetch filters and process based on user ID
def get_user_filters(user_id):
    # Fetch recent filters
    recent_filters = db[RECENT_COLLECTION].find_one({"userId": user_id})
    
    if not recent_filters:
        print(f"No recent filters found for user_id: {user_id}")
        return {}

    # Fetch aggregated filters
    aggregated_filters = db[AGGREGATED_COLLECTION].find_one({"userId": user_id})

    if not aggregated_filters:
        print(f"No aggregated filters found for user_id: {user_id}")
        return {}

    # Initialize structures to match aggregated format
    recent_color_counts = defaultdict(int)
    recent_size_counts = defaultdict(int)
    recent_brand_counts = defaultdict(int)
    recent_min_prices = []
    recent_max_prices = []

    # Parse recent filters
    for filter_set in recent_filters["filters"]:
        for color in filter_set.get("colorFilters", []):
            recent_color_counts[color] += 1
        for size in filter_set.get("sizeFilters", []):
            recent_size_counts[size] += 1
        for brand in filter_set.get("brandFilters", []):
            recent_brand_counts[brand] += 1

        min_price = filter_set.get("minPrice")
        max_price = filter_set.get("maxPrice")

        if isinstance(min_price, (int, float)):
            recent_min_prices.append(min_price)
        if isinstance(max_price, (int, float)):
            recent_max_prices.append(max_price)

    # Convert defaultdicts to regular dicts
    recent_aggregated = {
        "colorFilters": dict(recent_color_counts),
        "sizeFilters": dict(recent_size_counts),
        "brandFilters": dict(recent_brand_counts),
        "minPriceHistory": recent_min_prices,
        "maxPriceHistory": recent_max_prices,
        "filterAppliedCount": len(recent_filters["filters"])
    }

    # Compute price ranges
    recent_min, recent_max = compute_price_range(recent_aggregated["minPriceHistory"], recent_aggregated["maxPriceHistory"])
    recent_aggregated["minPrice"] = recent_min
    recent_aggregated["maxPrice"] = recent_max

    recent_aggregated.pop("minPriceHistory", None)
    recent_aggregated.pop("maxPriceHistory", None)

    # Apply same logic for aggregated filters
    agg_min, agg_max = compute_price_range(aggregated_filters["minPriceHistory"], aggregated_filters["maxPriceHistory"])
    aggregated_filters["minPrice"] = agg_min
    aggregated_filters["maxPrice"] = agg_max

    aggregated_filters.pop("minPriceHistory", None)
    aggregated_filters.pop("maxPriceHistory", None)

    # Merge filters
    final_data = {
        "brandFilters": merge_filters(
            recent_aggregated, aggregated_filters, "brandFilters",
            recent_aggregated["filterAppliedCount"], aggregated_filters["filterAppliedCount"]
        ),
        "colorFilters": merge_filters(
            recent_aggregated, aggregated_filters, "colorFilters",
            recent_aggregated["filterAppliedCount"], aggregated_filters["filterAppliedCount"]
        ),
        "sizeFilters": merge_filters(
            recent_aggregated, aggregated_filters, "sizeFilters",
            recent_aggregated["filterAppliedCount"], aggregated_filters["filterAppliedCount"]
        )
    }

    final_data["minPrice"], final_data["maxPrice"] = combine_price_ranges(
        aggregated_filters["minPrice"], aggregated_filters["maxPrice"],
        recent_aggregated["minPrice"], recent_aggregated["maxPrice"]
    )

    return final_data


# Function to build product vector based on filters
def build_product_vector(product, filter_vector):
    vector = []

    # --- Brands ---
    for brand in BRAND_VOCAB:
        vector.append(1.0 if product['brand'] == brand else 0.0)

    # --- Colors ---
    for color in COLOR_VOCAB:
        vector.append(1.0 if product.get('filtercolor') == color else 0.0)

    # --- Sizes ---
    sizes_present = set(product.get('sizes', []))
    total_sizes = len(sizes_present)
    for size in SIZE_VOCAB:
        if size in sizes_present:
            vector.append(1.0 / total_sizes)  # Normalized multi-hot
        else:
            vector.append(0.0)

    # --- Price ---
    price = product.get('price', 0)
    vector.append(normalize_price(price))  # normalized min price
    vector.append(normalize_price(price))  # normalized max price (same as min for product)

    return vector

# Function to score products using cosine similarity
def score_all_products_cosine(all_products, filter_vector):
    results = []

    for i, product in enumerate(all_products):
        product_vector = build_product_vector(product, filter_vector)

        temp_filter_vector = filter_vector.copy()
        temp_product_vector = product_vector.copy()

        # Adjust vectors based on price range
        filter_min_price = temp_filter_vector[-2]
        filter_max_price = temp_filter_vector[-1]
        product_price = temp_product_vector[-1]

        if filter_min_price <= product_price <= filter_max_price:
            temp_filter_vector[-2] = 0.4
            temp_filter_vector[-1] = 0.4
            temp_product_vector[-2] = 0.4
            temp_product_vector[-1] = 0.4

        # Convert to torch tensors
        f_vec = torch.tensor(temp_filter_vector, dtype=torch.float32)
        p_vec = torch.tensor(temp_product_vector, dtype=torch.float32)

        # Cosine similarity
        cosine_sim = nn.functional.cosine_similarity(f_vec, p_vec, dim=0).item()

        results.append({
            "index": i,
            "product_id": product.get("id", f"product_{i}"),
            "cosine_score": cosine_sim,
            "product": product
        })

    return results


# Flask app initialization
app = Flask(__name__)
CORS(app)

@app.route('/get-top-100', methods=['GET'])
def get_top_100_picks():
    # Extract the user_id from the request arguments
    user_id = request.args.get('user_id')
    user_id = ObjectId(user_id)
    
    # Fetch user filters and build filter vector
    user_filters = get_user_filters(user_id)

    # Build filter vector from user filters
    filter_vector = build_filter_vector(user_filters)

    # Fetch all products
    all_products = list(db[PRODUCTS_COLLECTION].find())

    # Score products based on filter vector
    all_cosine_scores = score_all_products_cosine(all_products, filter_vector)

    # Get top 100 picks based on cosine similarity
    top_100_by_cosine = sorted(all_cosine_scores, key=lambda x: x["cosine_score"], reverse=True)[:100]

    # Return the top 100 products as a JSON response
    top_100_products = [{
        "product_id": str(item["product"]["_id"])
    } for item in top_100_by_cosine]

    return jsonify(top_100_products)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render sets PORT env variable
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)

