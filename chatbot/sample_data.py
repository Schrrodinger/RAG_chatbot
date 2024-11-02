# sample_data.py
import json
import random
from datetime import datetime, timedelta


def generate_sample_products():
    # Sample specifications for different phone tiers
    specs_templates = {
        'high_end': {
            'Màn hình': ['AMOLED 6.7 inch', 'OLED 6.8 inch', 'Dynamic AMOLED 6.6 inch'],
            'CPU': ['Snapdragon 8 Gen 2', 'A16 Bionic', 'Dimensity 9200'],
            'RAM': ['12GB', '8GB'],
            'Bộ nhớ': ['256GB', '512GB'],
            'Pin': ['5000 mAh', '4800 mAh'],
            'Camera': ['108MP + 12MP + 10MP', '50MP + 48MP + 12MP']
        },
        'mid_range': {
            'Màn hình': ['AMOLED 6.4 inch', 'LCD 6.5 inch', 'IPS 6.6 inch'],
            'CPU': ['Snapdragon 7 Gen 1', 'Dimensity 8200', 'Helio G96'],
            'RAM': ['8GB', '6GB'],
            'Bộ nhớ': ['128GB', '256GB'],
            'Pin': ['4500 mAh', '5000 mAh'],
            'Camera': ['64MP + 8MP + 2MP', '50MP + 8MP + 2MP']
        },
        'budget': {
            'Màn hình': ['IPS 6.5 inch', 'LCD 6.4 inch', 'TFT 6.5 inch'],
            'CPU': ['Snapdragon 680', 'Helio G85', 'Dimensity 700'],
            'RAM': ['4GB', '6GB'],
            'Bộ nhớ': ['64GB', '128GB'],
            'Pin': ['5000 mAh', '4500 mAh'],
            'Camera': ['50MP + 2MP + 2MP', '48MP + 2MP + 2MP']
        }
    }

    # Sample phone brands and models
    phones = [
        # High-end phones
        {'name': 'Samsung Galaxy S23 Ultra', 'tier': 'high_end', 'brand': 'Samsung', 'base_price': 25990000},
        {'name': 'iPhone 14 Pro Max', 'tier': 'high_end', 'brand': 'Apple', 'base_price': 27990000},
        {'name': 'Xiaomi 13 Pro', 'tier': 'high_end', 'brand': 'Xiaomi', 'base_price': 23990000},

        # Mid-range phones
        {'name': 'Samsung Galaxy A54', 'tier': 'mid_range', 'brand': 'Samsung', 'base_price': 9990000},
        {'name': 'Redmi Note 12 Pro', 'tier': 'mid_range', 'brand': 'Xiaomi', 'base_price': 7990000},
        {'name': 'OPPO Reno8 T', 'tier': 'mid_range', 'brand': 'OPPO', 'base_price': 8490000},

        # Budget phones
        {'name': 'Samsung Galaxy A24', 'tier': 'budget', 'brand': 'Samsung', 'base_price': 5990000},
        {'name': 'Redmi 12C', 'tier': 'budget', 'brand': 'Xiaomi', 'base_price': 3190000},
        {'name': 'OPPO A57', 'tier': 'budget', 'brand': 'OPPO', 'base_price': 4290000}
    ]

    # Generate reviews templates
    review_templates = [
        "Sản phẩm {quality}. {feature} {opinion}. {recommend}",
        "{opinion_start} {feature} {quality}. {recommend}",
        "Đã dùng {duration}, {quality}. {feature} {opinion}."
    ]

    quality_phrases = {
        'high_end': ['rất tốt', 'xuất sắc', 'đáng tiền', 'cao cấp'],
        'mid_range': ['ổn định', 'tốt', 'đáng mua', 'khá tốt'],
        'budget': ['tạm được', 'ổn', 'phù hợp giá tiền', 'basic']
    }

    products = []

    for phone in phones:
        # Generate product specs
        specs = {}
        for key, values in specs_templates[phone['tier']].items():
            specs[key] = random.choice(values)

        # Generate random price variation
        price_variation = random.uniform(-0.05, 0.05)  # ±5%
        price = int(phone['base_price'] * (1 + price_variation))

        # Generate promotions
        promotions = []
        if random.random() < 0.7:  # 70% chance of having promotion
            discount = random.randint(5, 20)
            promotions.append(f"Giảm {discount}% khi thanh toán qua ví điện tử")
        if random.random() < 0.5:  # 50% chance of having additional promotion
            promotions.append("Tặng kèm ốp lưng chính hãng")

        # Generate reviews
        num_reviews = random.randint(5, 15)
        reviews = []
        for _ in range(num_reviews):
            template = random.choice(review_templates)
            quality = random.choice(quality_phrases[phone['tier']])
            feature = random.choice([
                f"Camera {['rất tốt', 'ổn', 'chụp đẹp', 'chụp nét'][random.randint(0, 3)]}",
                f"Pin {['trâu', 'dùng được lâu', 'tốt', 'đủ dùng'][random.randint(0, 3)]}",
                f"Màn hình {['đẹp', 'sắc nét', 'hiển thị tốt', 'có độ tương phản cao'][random.randint(0, 3)]}"
            ])
            opinion = random.choice([
                "rất hài lòng", "đáng mua", "giá cả hợp lý", "phù hợp nhu cầu"
            ])
            recommend = random.choice([
                "Recommend cho mọi người", "Sẽ giới thiệu cho bạn bè", "",
                "Nên mua về sử dụng", ""
            ])
            duration = random.choice([
                "1 tháng", "2 tháng", "3 tháng", "nửa năm"
            ])
            opinion_start = random.choice([
                "Theo đánh giá của tôi", "Nhìn chung", "Tổng thể", "Đánh giá sau khi dùng"
            ])

            review = template.format(
                quality=quality,
                feature=feature,
                opinion=opinion,
                recommend=recommend,
                duration=duration,
                opinion_start=opinion_start
            )
            reviews.append({
                'content': review,
                'rating': random.randint(4, 5) if phone['tier'] == 'high_end' else random.randint(3, 5),
                'date': (datetime.now() - timedelta(days=random.randint(1, 90))).strftime('%Y-%m-%d')
            })

        product = {
            'id': len(products) + 1,
            'name': phone['name'],
            'brand': phone['brand'],
            'price': price,
            'description': f"Điện thoại {phone['name']} - Phiên bản mới nhất từ {phone['brand']}",
            'specifications': specs,
            'promotions': promotions,
            'reviews': reviews,
            'warranty_info': "Bảo hành chính hãng 12 tháng tại trung tâm bảo hành toàn quốc",
            'category': 'Điện thoại',
            'in_stock': random.choice([True, True, True, False]),  # 75% chance in stock
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        products.append(product)

    return products


if __name__ == "__main__":
    products = generate_sample_products()

    # Save to JSON file
    with open('nlp-model/sample_products.json', 'w', encoding='utf-8') as f:
        json.dump(products, f, ensure_ascii=False, indent=2)

    print(f"Generated {len(products)} sample products")