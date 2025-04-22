from pymongo import MongoClient
from bson import ObjectId
from pprint import pprint
from collections import defaultdict
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

MONGODB_URI = "mongodb+srv://AhmadJabbar:0uU29STyRwhoxV0X@shopsavvy.xaqy1.mongodb.net/"
DATABASE_NAME = "test"

BRAND_VOCAB = ['LAMA', 'Outfitters']
COLOR_VOCAB = ['Beige', 'Black', 'Blue', 'Brown', 'Green', 'Grey', 'Multi-color', 'Orange', 'Other', 'Pink', 'Purple', 'Red', 'White', 'Yellow']
SIZE_VOCAB = ['XS', 'S', 'M', 'L', 'XL', '2XL', '3XL', '4XL', '24', '26', '28', '30', '32', '33', '34', '36', '40']
GENDER_VOCAB = ["Men", "Women"]
CATEGORY_VOCAB = [
    "T-Shirt", "Polo", "Shirts", "Sweaters & Cardigans", "Hoodies & Sweatshirts", "Jackets & Coats",
    "Blazers", "Activewear", "Jeans", "Trousers", "Shorts", "Camisole & Bandeaus", "Tops & Blouses", 
    "Bodysuits", "Co-ords", "Dresses & Skirts", "Fur & Fleece", "Studio", "TrueBody"
]

# Setup MongoDB client
client = MongoClient(MONGODB_URI)
db = client[DATABASE_NAME]

def fetch_all_data(user_id, user_id_str):
    # Perform all queries sequentially using pymongo (synchronously)
    recent_filters = db["filters"].find_one({ "userId": user_id })
    aggregated_filters = db["aggregatedfilters"].find_one({ "userId": user_id })
    recent_type_interests = db["typeinterests"].find_one({ "userId": user_id })
    aggregated_type_interests = db["aggregatedtypeinterests"].find_one({ "userId": user_id })
    user_doc = db["accounts"].find_one({ "_id": user_id }, { "wishlist": 1, "_id": 0 })
    recent_clicks_doc = db["userrecentproductclicks"].find_one({ "userId": user_id_str })
    aggregated_clicks_doc = db["userproductclickaggregates"].find_one({ "userId": user_id_str })
    all_products = list(db["products"].find())

    # Return structured data
    return {
        "recent_filters": recent_filters,
        "aggregated_filters": aggregated_filters,
        "recent_type_interests": recent_type_interests,
        "aggregated_type_interests": aggregated_type_interests,
        "user_doc": user_doc,
        "recent_clicks_doc": recent_clicks_doc,
        "aggregated_clicks_doc": aggregated_clicks_doc,
        "all_products": all_products
    }

def process_applied_filters(data):
    try:
        recent_filters = data["recent_filters"]
        aggregated_filters = data["aggregated_filters"]

        # --- Price normalization ---
        def normalize_price(val, min_=0, max_=20000):
            try: return (val - min_) / (max_ - min_)
            except: return 0.0

        # --- Compute price range ---
        def compute_price_range(mins, maxs, default_min=0, default_max=50000):
            clean_min = [p for p in mins if p > 0]
            clean_max = [p for p in maxs if p < default_max]
            min_p = int(np.percentile(clean_min, 10)) if len(clean_min) >= 3 else default_min
            max_p = int(np.percentile(clean_max, 90)) if len(clean_max) >= 3 else default_max
            return min_p, max_p

        # --- Merge filters ---
        def merge_filters(recent, aggregated, key, r_count, a_count, rw=0.75, aw=0.25):
            merged = {}
            all_keys = set(recent.get(key, {})) | set(aggregated.get(key, {}) if aggregated else {})
            for k in all_keys:
                r_val = recent.get(key, {}).get(k, 0) / r_count if r_count else 0
                a_val = aggregated.get(key, {}).get(k, 0) / a_count if aggregated and a_count else 0
                score = round(r_val * rw + a_val * aw, 4)
                if score > 0:
                    merged[k] = score
            return merged

        # --- Combine price ranges ---
        def combine_price_ranges(agg_min, agg_max, rec_min, rec_max):
            if agg_min is not None and agg_max is not None:
                min_final = int(rec_min * 0.9 if agg_min < rec_min else rec_min * 1.05)
                max_final = int(rec_max * 1.1 if agg_max > rec_max else rec_max * 0.95)
            else:
                min_final, max_final = rec_min, rec_max
            return min_final, max_final

        # --- Build filter vector ---
        def build_filter_vector(f):
            if not f: raise ValueError("❌ No filter data provided.")
            v = []
            v += [f.get('brandFilters', {}).get(b, 0.0) for b in BRAND_VOCAB]
            v += [f.get('colorFilters', {}).get(c, 0.0) for c in COLOR_VOCAB]
            v += [f.get('sizeFilters', {}).get(s, 0.0) for s in SIZE_VOCAB]
            v += [normalize_price(f.get('minPrice', 0)), normalize_price(f.get('maxPrice', 20000))]
            return v

        # --- Build product vector ---
        def build_product_vector(p):
            sizes = set(p.get('sizes', []))
            size_norm = 1.0 / len(sizes) if sizes else 0
            return (
                [1.0 if p.get('brand') == b else 0.0 for b in BRAND_VOCAB] +
                [1.0 if p.get('filtercolor') == c else 0.0 for c in COLOR_VOCAB] +
                [size_norm if s in sizes else 0.0 for s in SIZE_VOCAB] +
                [normalize_price(p.get('price', 0))] * 2
            )

        # --- Score all products ---
        def score_all_products_cosine(products, filter_vector):
            if not filter_vector:
                return None
            results = []
            for i, p in enumerate(products):
                pv = build_product_vector(p)
                f_vec, p_vec = filter_vector[:], pv[:]
                price = p_vec[-1]
                if f_vec[-2] <= price <= f_vec[-1]:
                    f_vec[-2] = f_vec[-1] = p_vec[-2] = p_vec[-1] = 0.4
                cosine = nn.functional.cosine_similarity(
                    torch.tensor(f_vec, dtype=torch.float32),
                    torch.tensor(p_vec, dtype=torch.float32),
                    dim=0
                ).item()
                results.append({
                    "index": i,
                    "product_id": p.get("id", f"product_{i}"),
                    "cosine_score": cosine,
                    "product": p
                })
            return results

        # --- Process recent filters ---
        recent_aggregated = None
        if recent_filters:
            color, size, brand, min_p, max_p = defaultdict(int), defaultdict(int), defaultdict(int), [], []
            for f in recent_filters.get("filters", []):
                for c in f.get("colorFilters", []): color[c] += 1
                for s in f.get("sizeFilters", []): size[s] += 1
                for b in f.get("brandFilters", []): brand[b] += 1
                if isinstance(f.get("minPrice"), (int, float)): min_p.append(f["minPrice"])
                if isinstance(f.get("maxPrice"), (int, float)): max_p.append(f["maxPrice"])
            recent_aggregated = {
                "colorFilters": dict(color),
                "sizeFilters": dict(size),
                "brandFilters": dict(brand),
                "minPriceHistory": min_p,
                "maxPriceHistory": max_p,
                "filterAppliedCount": len(recent_filters.get("filters", []))
            }

        if recent_aggregated:
            r_min, r_max = compute_price_range(
                recent_aggregated["minPriceHistory"], recent_aggregated["maxPriceHistory"]
            )
            recent_aggregated.update({"minPrice": r_min, "maxPrice": r_max})
            recent_aggregated.pop("minPriceHistory", None)
            recent_aggregated.pop("maxPriceHistory", None)

        if aggregated_filters:
            a_min, a_max = compute_price_range(
                aggregated_filters.get("minPriceHistory", []),
                aggregated_filters.get("maxPriceHistory", [])
            )
            aggregated_filters.update({"minPrice": a_min, "maxPrice": a_max})
            aggregated_filters.pop("minPriceHistory", None)
            aggregated_filters.pop("maxPriceHistory", None)

        if not recent_aggregated:
            return None

        use_agg = aggregated_filters is not None
        rw, aw = (1.0, 0.0) if not use_agg else (0.75, 0.25)

        final_data = {
            "brandFilters": merge_filters(
                recent_aggregated, aggregated_filters, "brandFilters",
                recent_aggregated["filterAppliedCount"], aggregated_filters["filterAppliedCount"] if use_agg else 0,
                rw, aw
            ),
            "colorFilters": merge_filters(
                recent_aggregated, aggregated_filters, "colorFilters",
                recent_aggregated["filterAppliedCount"], aggregated_filters["filterAppliedCount"] if use_agg else 0,
                rw, aw
            ),
            "sizeFilters": merge_filters(
                recent_aggregated, aggregated_filters, "sizeFilters",
                recent_aggregated["filterAppliedCount"], aggregated_filters["filterAppliedCount"] if use_agg else 0,
                rw, aw
            )
        }

        final_data["minPrice"], final_data["maxPrice"] = combine_price_ranges(
            aggregated_filters["minPrice"] if use_agg else None,
            aggregated_filters["maxPrice"] if use_agg else None,
            recent_aggregated["minPrice"],
            recent_aggregated["maxPrice"]
        )

        filter_vector = build_filter_vector(final_data)
        return score_all_products_cosine(data["all_products"], filter_vector)

    except Exception as e:
        print(f"❌ Error in process_applied_filters: {e}")
        return None
    
def process_type_interest(data):
    try:
        # Step 1: Aggregate recent gender/category interest
        gender_interest_map = defaultdict(float)
        category_interest_map = defaultdict(float)

        if data["recent_type_interests"]:
            for block in data["recent_type_interests"].get("typeInterests", []):
                gender = block.get("gender")
                category = block.get("category")
                if not gender or not category or gender.lower() == "all" or category.lower() == "all":
                    continue
                weight = 1 + (0.5 * block.get("pagesClicked", 0)) + (block.get("productsClicked", 0))
                gender_interest_map[gender] += weight
                category_interest_map[category] += weight

        recent_aggregated = {
            "genderInterestMap": dict(gender_interest_map),
            "categoryInterestMap": dict(category_interest_map)
        }

        # Step 2: Merge with aggregated interests
        def merge_interest_maps(recent, aggregated, key, recent_weight=0.75, aggregated_weight=0.25):
            merged = {}
            recent_map = recent.get(key, {})
            aggregated_map = aggregated.get(key, {}) if aggregated else {}
            for k in set(recent_map) | set(aggregated_map):
                recent_val = recent_map.get(k, 0)
                aggregated_val = aggregated_map.get(k, 0)
                merged[k] = round(recent_val * recent_weight + aggregated_val * aggregated_weight, 4) \
                            if recent_val + aggregated_val > 0 else 0
            return merged

        use_agg = data.get("aggregated_type_interests") is not None
        rw, aw = (1.0, 0.0) if not use_agg else (0.75, 0.25)

        final_type_interests = {
            "genderInterestMap": merge_interest_maps(
                recent_aggregated, data.get("aggregated_type_interests"), "genderInterestMap", rw, aw
            ),
            "categoryInterestMap": merge_interest_maps(
                recent_aggregated, data.get("aggregated_type_interests"), "categoryInterestMap", rw, aw
            )
        }

        # Step 3: Build type-interest vector
        def build_type_interest_vector(type_interest_dict):
            vector = []
            for gender in GENDER_VOCAB:
                vector.append(type_interest_dict.get("genderInterestMap", {}).get(gender, 0.0))
            for category in CATEGORY_VOCAB:
                vector.append(type_interest_dict.get("categoryInterestMap", {}).get(category, 0.0))
            return vector

        type_interest_vector = build_type_interest_vector(final_type_interests)

        # Step 4: Build product type vectors and score
        def build_product_type_vector(product):
            vector = []
            for g in GENDER_VOCAB:
                vector.append(1.0 if product.get("gender") == g else 0.0)
            for c in CATEGORY_VOCAB:
                vector.append(1.0 if product.get("type") == c else 0.0)
            return vector

        def score_all_products_type_interest(all_products, type_interest_vector):
            results = []
            for product in all_products:
                product_vector = build_product_type_vector(product)
                cosine_sim = F.cosine_similarity(
                    torch.tensor(type_interest_vector, dtype=torch.float32),
                    torch.tensor(product_vector, dtype=torch.float32),
                    dim=0
                ).item()
                results.append({
                    "product_id": str(product.get("_id", "unknown")),
                    "cosine_score": cosine_sim,
                    "product": product
                })
            return results

        return score_all_products_type_interest(data["all_products"], type_interest_vector)

    except Exception as e:
        print(f"❌ Error in process_type_interest: {e}")
        return None
    
def process_wishlist(data):
    wishlist = data["user_doc"].get("wishlist", [])
    if not wishlist or len(wishlist) < 3:
        print("⚠️ Not enough wishlist data to compute vector.")
        return None

    # Clean relevant fields
    wishlist_cleaned = [
        {
            "gender": item["gender"],
            "type": item["type"],
            "brand": item["brand"],
            "filtercolor": item["filtercolor"],
            "price": item["price"]
        }
        for item in wishlist
        if all(k in item for k in ["gender", "type", "brand", "filtercolor", "price"])
    ]

    # --- Wishlist Vector ---
    def normalize_price(value, min_possible=0, max_possible=20000):
        return (value - min_possible) / (max_possible - min_possible)

    def build_wishlist_vector(wishlist):
        vector = []
        gender_counts = Counter(item["gender"] for item in wishlist)
        type_counts = Counter(item["type"] for item in wishlist)
        brand_counts = Counter(item["brand"] for item in wishlist)
        color_counts = Counter(item["filtercolor"] for item in wishlist)
        prices = [item["price"] for item in wishlist]
        total_items = len(wishlist)

        for gender in GENDER_VOCAB:
            vector.append(gender_counts.get(gender, 0) / total_items)
        for cat in CATEGORY_VOCAB:
            vector.append(type_counts.get(cat, 0) / total_items)
        for brand in BRAND_VOCAB:
            vector.append(brand_counts.get(brand, 0) / total_items)
        for color in COLOR_VOCAB:
            vector.append(color_counts.get(color, 0) / total_items)

        price_array = np.array(sorted(prices))
        p10, p90 = np.percentile(price_array, [10, 90])
        vector.append(normalize_price((p10 + p90) / 2))

        return vector

    def build_product_vector_for_wishlist_match(product):
        vector = []
        for gender in GENDER_VOCAB:
            vector.append(1.0 if product.get("gender") == gender else 0.0)
        for type_ in CATEGORY_VOCAB:
            vector.append(1.0 if product.get("type") == type_ else 0.0)
        for brand in BRAND_VOCAB:
            vector.append(1.0 if product.get("brand") == brand else 0.0)
        for color in COLOR_VOCAB:
            vector.append(1.0 if product.get("filtercolor") == color else 0.0)
        vector.append(normalize_price(product.get("price", 0)))
        return vector

    def score_all_products_wishlist(all_products, wishlist_vector):
        results = []
        for i, product in enumerate(all_products):
            product_vector = build_product_vector_for_wishlist_match(product)
            cosine_sim = F.cosine_similarity(
                torch.tensor(wishlist_vector, dtype=torch.float32),
                torch.tensor(product_vector, dtype=torch.float32),
                dim=0
            ).item()
            results.append({
                "index": i,
                "product_id": str(product.get("_id", f"product_{i}")),
                "cosine_score": cosine_sim,
                "product": product
            })
        return results

    try:
        wishlist_vector = build_wishlist_vector(wishlist_cleaned)
        if wishlist_vector is None:
            return None
        return score_all_products_wishlist(data["all_products"], wishlist_vector)
    except Exception as e:
        print(f"\n❌ Failed to compute wishlist scores: {e}\n")
        return None

    
def process_user_click_data(data):
    recent_clicks = data["recent_clicks_doc"].get("clickedProducts", [])
    aggregated_data = data["aggregated_clicks_doc"].get("aggregateData", {})

    # Process existing aggregated priceMean
    if aggregated_data.get("priceHistory"):
        prices = np.sort(aggregated_data["priceHistory"])
        trimmed = [p for p in prices if np.percentile(prices, 10) <= p <= np.percentile(prices, 90)]
        aggregated_data["priceMean"] = round(np.mean(trimmed), 2)
        del aggregated_data["priceHistory"]
    else:
        aggregated_data["priceMean"] = None

    def compute_weighted_score(product):
        return math.pow(product["visitCount"], 0.9) + (3 if product["redirected"] else 0)

    # Aggregates
    brand_counts = defaultdict(float)
    filtercolor_counts = defaultdict(float)
    gender_counts = defaultdict(float)
    type_counts = defaultdict(float)
    size_counts = defaultdict(float)
    price_history = []

    # Process recent clicks
    for product in recent_clicks:
        score = compute_weighted_score(product)
        rounded_score = math.floor(score)

        for field, counts in zip(
            ["brand", "filtercolor", "gender", "type"],
            [brand_counts, filtercolor_counts, gender_counts, type_counts]
        ):
            counts[product[field]] += score

        for size in product["sizes"]:
            size_counts[size] += score

        price_history.extend([product["price"]] * rounded_score)

    # Compute price mean from recent clicks
    if price_history:
        sorted_prices = np.sort(price_history)
        trimmed_prices = [p for p in sorted_prices if np.percentile(sorted_prices, 10) <= p <= np.percentile(sorted_prices, 90)]
        price_mean = round(np.mean(trimmed_prices), 2)
    else:
        price_mean = None

    recent_aggregated_data = {
        "brandCounts": dict(brand_counts),
        "filtercolorCounts": dict(filtercolor_counts),
        "genderCounts": dict(gender_counts),
        "typeCounts": dict(type_counts),
        "sizeCounts": dict(size_counts),
        "priceMean": price_mean
    }

    def weighted_merge(recent, aggregated, weight_recent=0.75, weight_agg=0.25):
        if not recent and not aggregated:
            return None
        if recent and not aggregated:
            return recent
        if not recent and aggregated:
            return aggregated

        def merge_dicts(r_dict, a_dict):
            keys = set(r_dict) | set(a_dict)
            return {
                k: r_dict.get(k, 0) * weight_recent + a_dict.get(k, 0) * weight_agg
                for k in keys
            }

        return {
            "brandCounts": merge_dicts(recent.get("brandCounts", {}), aggregated.get("brandCounts", {})),
            "filtercolorCounts": merge_dicts(recent.get("filtercolorCounts", {}), aggregated.get("filtercolorCounts", {})),
            "genderCounts": merge_dicts(recent.get("genderCounts", {}), aggregated.get("genderCounts", {})),
            "typeCounts": merge_dicts(recent.get("typeCounts", {}), aggregated.get("typeCounts", {})),
            "sizeCounts": merge_dicts(recent.get("sizeCounts", {}), aggregated.get("sizeCounts", {})),
            "priceMean": round(
                recent.get("priceMean", 0) * weight_recent + aggregated.get("priceMean", 0) * weight_agg, 2
            )
        }

    final_product_click_data = weighted_merge(recent_aggregated_data, aggregated_data)

    def normalize_section(section):
        total = sum(section.values())
        return {k: round(v / total, 4) if total else 0 for k, v in section.items()}

    def normalize_product_click_data(data):
        if not data:
            return None
        return {
            k: normalize_section(v) if isinstance(v, dict) else round(v, 2)
            for k, v in data.items()
        }

    normalized_data = normalize_product_click_data(final_product_click_data)

    def normalize_price(price, min_price=1000, max_price=15000):
        return round((price - min_price) / (max_price - min_price), 4)

    def build_click_vector(data):
        vector = []
        for key, vocab in {
            "genderCounts": GENDER_VOCAB,
            "typeCounts": CATEGORY_VOCAB,
            "brandCounts": BRAND_VOCAB,
            "filtercolorCounts": COLOR_VOCAB,
            "sizeCounts": SIZE_VOCAB,
        }.items():
            for token in vocab:
                vector.append(data.get(key, {}).get(token, 0))
        vector.append(normalize_price(data.get("priceMean", 0)))
        return vector

    def build_product_vector(product):
        vector = []
        for key, vocab in {
            "gender": GENDER_VOCAB,
            "type": CATEGORY_VOCAB,
            "brand": BRAND_VOCAB,
            "filtercolor": COLOR_VOCAB,
            "sizes": SIZE_VOCAB,
        }.items():
            for token in vocab:
                if key == "sizes":
                    vector.append(1 if token in product.get(key, []) else 0)
                else:
                    vector.append(1 if product.get(key) == token else 0)
        vector.append(normalize_price(product.get("price", 0)))
        return vector

    def score_all_products_by_click_vector(all_products, click_vector):
        if not click_vector:
            return None
        results = []
        try:
            u_vec = torch.tensor(click_vector, dtype=torch.float32)
            for i, product in enumerate(all_products):
                p_vec = torch.tensor(build_product_vector(product), dtype=torch.float32)
                cosine_sim = F.cosine_similarity(u_vec, p_vec, dim=0).item()
                results.append({
                    "index": i,
                    "product_id": str(product.get("_id", f"product_{i}")),
                    "cosine_score": cosine_sim,
                    "product": product
                })
        except Exception as e:
            print(f"❌ Error during product scoring: {e}")
            return None
        return results

    final_click_vector = build_click_vector(normalized_data)
    product_scores = score_all_products_by_click_vector(data["all_products"], final_click_vector)

    return product_scores

def calculate_final_recommendations(data, all_product_scores, all_wishlist_scores, all_type_scores, all_cosine_scores_filter):
    def get_dynamic_weights(scores):
        present = {
            k: scores.get(k) is not None for k in [
                'products_clicked', 'wishlist', 'type_interest', 'applied_filters'
            ]
        }

        if all(present.values()):
            return {"products_clicked": 0.4, "wishlist": 0.3, "type_interest": 0.2, "applied_filters": 0.1}
        if not present['products_clicked']:
            return {"wishlist": 0.4, "type_interest": 0.35, "applied_filters": 0.25}
        if not present['wishlist']:
            return {"products_clicked": 0.5, "type_interest": 0.3, "applied_filters": 0.2}
        if not present['type_interest']:
            return {"products_clicked": 0.5, "wishlist": 0.35, "applied_filters": 0.15}
        if not present['applied_filters']:
            return {"products_clicked": 0.5, "wishlist": 0.35, "type_interest": 0.15}
        if not present['products_clicked'] and not present['applied_filters']:
            return {"wishlist": 0.5, "type_interest": 0.5}
        if not present['products_clicked'] and not present['type_interest']:
            return {"wishlist": 0.5, "applied_filters": 0.5}
        if not present['wishlist'] and not present['type_interest']:
            return {"products_clicked": 0.65, "applied_filters": 0.35}
        if not present['wishlist'] and not present['applied_filters']:
            return {"products_clicked": 0.65, "type_interest": 0.35}
        if not present['type_interest'] and not present['applied_filters']:
            return {"products_clicked": 0.6, "wishlist": 0.4}
        if all(not present[k] for k in ['products_clicked', 'wishlist', 'type_interest']):
            return {"applied_filters": 1.0}
        if all(not present[k] for k in ['products_clicked', 'wishlist', 'applied_filters']):
            return {"type_interest": 1.0}
        if all(not present[k] for k in ['wishlist', 'type_interest', 'applied_filters']):
            return {"products_clicked": 1.0}
        return {}

    def calculate_final_cosine_score(scores):
        weights = get_dynamic_weights(scores)
        if not weights:
            return None
        return sum(scores[k]["cosine_score"] * w for k, w in weights.items() if scores.get(k))

    # Compute and sort final product scores
    final_scores = []
    for i, product in enumerate(data["all_products"]):
        scores = {
            "products_clicked": all_product_scores[i] if all_product_scores else None,
            "wishlist": all_wishlist_scores[i] if all_wishlist_scores else None,
            "type_interest": all_type_scores[i] if all_type_scores else None,
            "applied_filters": all_cosine_scores_filter[i] if all_cosine_scores_filter else None
        }

        score = calculate_final_cosine_score(scores)
        if score is not None:
            final_scores.append({
                "product_id": str(product["_id"]),
                "product": product["product"],
                "final_cosine_score": round(score, 4)
            })

    sorted_top_100 = sorted(final_scores, key=lambda x: x["final_cosine_score"], reverse=True)[:100]
    return sorted_top_100

app = Flask(__name__)
CORS(app)

@app.route('/get-top-100', methods=['GET'])
def get_top_100_picks():
    try:
        user_id_str = request.args.get('user_id')
        print(f"User ID (raw): {user_id_str}")
        
        user_id = ObjectId(user_id_str)
        print(f"Parsed ObjectId: {user_id}")

        data = fetch_all_data(user_id, user_id_str)
        print("✅ Data fetched successfully")

        all_cosine_scores_filter = process_applied_filters(data)
        print("✅ Filters processed")

        all_type_scores = process_type_interest(data)
        print("✅ Type interests processed")

        all_wishlist_scores = process_wishlist(data)
        print("✅ Wishlist processed")

        all_product_scores = process_user_click_data(data)
        print("✅ Click data processed")

        top_100_products = calculate_final_recommendations(
            data,
            all_product_scores,
            all_wishlist_scores,
            all_type_scores,
            all_cosine_scores_filter
        )
        print("✅ Final recommendations calculated")

        response = [{
            "product_id": item["product_id"],
            "product": item["product"],
            "final_cosine_score": item["final_cosine_score"]
        } for item in top_100_products]

        return jsonify(response)

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5050))  # Render sets PORT env variable
    app.run(host='0.0.0.0', port=port, debug=True, threaded=True)