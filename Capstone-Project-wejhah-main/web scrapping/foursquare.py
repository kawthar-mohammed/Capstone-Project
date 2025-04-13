import requests
import csv
import time
import arabic_reshaper
from bidi.algorithm import get_display

# Function to fix Arabic text for proper RTL display
def fix_arabic_text(text):
    if isinstance(text, str) and any("\u0600" <= char <= "\u06FF" for char in text):
        try:
            return get_display(arabic_reshaper.reshape(text))
        except Exception:
            return text
    return text

# Safe getter to traverse nested dictionaries and lists
def safe_get(d, keys, default="N/A"):
    for key in keys:
        try:
            d = d[key]
        except (KeyError, TypeError, IndexError):
            return default
    return d if d is not None else default

# Your Foursquare API key and endpoint
API_KEY = "foursquare_api_key"  # Replace with your API key
URL = "https://api.foursquare.com/v3/places/search"

HEADERS = {
    "Accept": "application/json",
    "Authorization": API_KEY
}

# Define neighborhoods for each district separately
# riyadh = [
#     ("Al Sahafah", 24.8067, 46.6285),
#     ("Al Rabi", 24.8000, 46.6500),
#     ("Al Nada", 24.8300, 46.6600),
#     ("Al Narjis", 24.8868, 46.6453),
#     ("Al ‘Arid", 24.9410, 46.5580),
#     ("Al Nafil", 24.8130, 46.6250),
#     ("Al ‘Aqiq", 24.7700, 46.5850),
#     ("Al Wadi", 24.7900, 46.6170),
#     ("Al Ghadir", 24.7560, 46.6530),
#     ("Al Yasmin", 24.8260, 46.6130),
#     ("Al Falah", 24.8340, 46.6550),
#     ("Banban", 25.1000, 46.5000),
#     ("Al Qirawan", 24.8620, 46.5580),
#     ("Hittin", 24.7730, 46.5710),
#     ("Al Malqa", 24.8008, 46.5977),
#     ("Al Khair", 25.0409, 46.4677),
#     ("Al Rawdah", 24.7540, 46.7980),
#     ("Al Rimal", 24.8600, 46.8570),
#     ("Al Munsiyah", 24.8320, 46.8250),
#     ("Qurtubah", 24.8060, 46.7820),
#     ("Al Janadriyah", 24.9040, 46.8570),
#     ("Al Qadisiyah", 24.7890, 46.8840),
#     ("Al Yarmuk", 24.7680, 46.8770),
#     ("Ghirnatah (Granada)", 24.7800, 46.7610),
#     ("Ishbiliyah", 24.7660, 46.8090),
#     ("Al Hamra", 24.7520, 46.8090),
#     ("Al Mu'aysim (Al Mu’aizilah)", 24.7840, 46.8460),
#     ("Al Khaleej", 24.7450, 46.8280),
#     ("King Faisal", 24.7340, 46.8210),
#     ("Al Quds", 24.7340, 46.7970),
#     ("An Nahdah", 24.7230, 46.8330),
#     ("Al Andalus", 24.7190, 46.8100),
#     ("East Al Naseem", 24.7450, 46.8250),
#     ("West Al Naseem", 24.7450, 46.7940),
#     ("As Salam", 24.7260, 46.7710),
#     ("Ar Rayyan", 24.7190, 46.8060),
#     ("Umm Al Hammam (East)", 24.6680, 46.6580),
#     ("Umm Al Hammam (West)", 24.6690, 46.6370),
#     ("Al Aziziyah", 24.5700, 46.7620),
#     ("Taybah (Al Taibah)", 24.5010, 46.8090),
#     ("Al Misfalah (Al Masfah)", 24.4800, 46.9130),
#     ("Ad Dar Al Bayda", 24.5010, 46.8370),
#     ("Al Shifa", 24.5655, 46.6947),
#     ("Badr", 24.5400, 46.6690),
#     ("Al Marwah", 24.5200, 46.7100),
#     ("Al Fawaz", 24.5060, 46.6540),
#     ("Al Hazm", 24.5340, 46.5890),
#     ("Al Mansuriyah", 24.4740, 46.5530),
#     ("Dirab", 24.5170, 46.5230),
#     ("Al Ha’ir", 24.3740, 46.8000),
#     ("Al Uraija", 24.5930, 46.5590),
#     ("Al Uraija Al Wusta", 24.5800, 46.5320),
#     ("Al Uraija Al Gharbiyyah", 24.5830, 46.5080),
#     ("Hijrat Wadi Laban", 24.5680, 46.4730),
#     ("Dhahrat Laban", 24.6000, 46.4820),
#     ("Shubra", 24.5810, 46.6250),
#     ("As Suwaidi", 24.5820, 46.6250),
#     ("As Suwaidi Al Gharbi", 24.5660, 46.5900),
#     ("Dhahrat Al Badiah", 24.5820, 46.6460),
#     ("Al Badiah", 24.5700, 46.6670),
#     ("Sultana", 24.5680, 46.6260),
#     ("Az Zahra", 24.5520, 46.6170),
#     ("Namar", 24.5500, 46.5400),
#     ("Dhahrat Namar", 24.5620, 46.5620),
#     ("Tuwaiq", 24.5639, 46.5733),
#     ("Al Hazm", 24.5340, 46.5890),
#     ("Al Malaz", 24.6643, 46.7354),
#     ("Al Rabwah", 24.6700, 46.7090),
#     ("Jarir", 24.6550, 46.7320),
#     ("Al Zahra", 24.6480, 46.7170),
#     ("As Safa", 24.6450, 46.7050),
#     ("Adh Dhubbat", 24.6510, 46.7160),
#     ("Al Wizarat", 24.6600, 46.7050),
#     ("Al Faruq", 24.6420, 46.7130),
#     ("Al Amal", 24.6370, 46.7150),
#     ("Thulaim", 24.6310, 46.7130),
#     ("Al Murabba", 24.6490, 46.7080),
#     ("Al Futah", 24.6310, 46.7030),
#     ("Al Morooj", 24.7510, 46.6570),
#     ("Al Mursalat", 24.7490, 46.6860),
#     ("An Nuzhah", 24.7600, 46.6970),
#     ("Al Mughrizat", 24.7640, 46.7250),
#     ("Al Wurud", 24.7160, 46.6550),
#     ("Salah ad-Din", 24.7420, 46.7080),
#     ("King Salman Neighborhood", 24.7403, 46.7144),
#     ("Al Olaya", 24.6957, 46.6811),
#     ("As Sulaymaniyah", 24.6940, 46.6750),
#     ("King Abdul Aziz District", 24.7210, 46.7197)
# ]
events_coordinates = [('Al ‘Aqiq', 24.75560630409196, 46.58741836085954), ('Hittin', 24.7678522, 46.5765421), ('Hittin', 24.74553, 46.53523), ('Qurtubah', 24.80225, 46.79553), ('Umm Al Hammam (West)', 24.6674632, 46.6334669), ('Al Nada', 24.818236, 46.6870917), ('Al Wurud', 24.7248589, 46.6763911), ('Hittin', 24.7479649, 46.5363683), ('Al Janadriyah', 24.98251, 46.77867), ('Hittin', 24.759493, 46.432959), ('Al Ghadir', 24.767865, 46.6443752), ('Al Futah', 24.634885932161467, 46.67277854232931), ('Al ‘Aqiq', 24.733717098196934, 46.57388469999999), ('Al ‘Aqiq', 24.7692187, 46.6048397), ('Al ‘Aqiq', 24.76929793989671, 46.60621687301289), ('An Nuzhah', 24.781665, 46.697334), ('Umm Al Hammam (West)', 24.66709173200477, 46.628629643388685), ('Al Ghadir', 24.774561, 46.639855), ('Al Janadriyah', 24.991706824443003, 46.78504424267779), ('Hittin', 24.78492010423248, 46.56904167145531), ('Al Ghadir', 24.759684, 46.66369), ('Hijrat Wadi Laban', 24.377, 45.92174), ('King Salman Neighborhood', 24.7470703, 46.72400659999999), ('Al ‘Aqiq', 24.7677222, 46.6043889), ('Al Wizarat', 24.666637, 46.6967202), ('Al ‘Aqiq', 24.74179281879065, 46.57149092604672), ('Umm Al Hammam (West)', 24.665783399478, 46.630823884655), ('Al Janadriyah', 24.94631, 46.75176), ('Al ‘Aqiq', 24.74184391717291, 46.57154276958509), ('Al ‘Aqiq', 24.7738154, 46.6054912), ('Al Wizarat', 24.6666546, 46.6921413), ('Qurtubah', 24.8413476, 46.7332522), ('Al Mughrizat', 24.7530562, 46.7267723), ('Hijrat Wadi Laban', 21.365809, 40.280285), ('Umm Al Hammam (West)', 24.675873, 46.561861), ('Al ‘Aqiq', 24.76433, 46.607), ('Al Ghadir', 24.767525, 46.641954), ('Al Ghadir', 24.759811, 46.663645), ('Al Nada', 24.8284444, 46.6602578), ('Al Ghadir', 24.75961, 46.66373), ('Al Qirawan', 24.872431, 46.574255), ('Al ‘Arid', 25.028510320090447, 46.57973996970665), ('Qurtubah', 24.835667695113, 46.729393550928), ('Al ‘Aqiq', 24.767419, 46.6049014), ('Al Mughrizat', 24.8081769, 46.7183483), ('Al Morooj', 24.7357905, 46.6373989), ('Al ‘Aqiq', 24.77273277699925, 46.59946921349281), ('Al ‘Aqiq', 24.74976, 46.614551), ('Al Wurud', 24.7135517, 46.6752957), ('Al ‘Aqiq', 24.7418217, 46.57145), ('Hittin', 24.74556, 46.53518), ('Al Rabi', 24.794371, 46.65040009999999), ('Al Murabba', 24.64805747978537, 46.71134526715597), ('Al Nafil', 24.809330241322687, 46.61908246942067), ('Al Sahafah', 24.803417847989472, 46.62873617116404), ('Hittin', 24.774876, 46.369563), ('Al Khair', 24.998697, 46.492561), ('Al Wurud', 24.7169067, 46.6391112), ('Al Malaz', 24.66272, 46.74242)]

# Choose district to process
selected_district = events_coordinates  # Change this to the district you want to run
csv_filename = "riyadh_events_venues.csv"  # Modify filename for each district

# Set to track seen venue IDs to avoid duplicates
seen_ids = set()
total_records = 0

with open(csv_filename, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "Neighborhood", "FSQ_ID", "Name", "Address", "Locality", "Region", "Country",
        "Latitude", "Longitude", "Category", "Rating", "Total Ratings",
        "Total Photos", "Total Tips", "Popularity Score", "Verified",
        "Website", "Telephone", "Price Tier", "Hours", "Popular Hours",
        "Menu", "Social Media", "Description", "Photo URL", "User Tip", "Timezone"
    ])
    
    for neighborhood, lat, lon in selected_district:
        print(f"Fetching venues in {neighborhood}...")
        
        params = {
            "query": "mall, cinema", 
            "ll": f"{lat},{lon}",
            "radius": 5000,
            "limit": 50,
            "sort": "POPULARITY",
            "fields": (
                "fsq_id,name,location,categories,distance,geocodes,"
                "rating,stats,popularity,verified,website,tel,price,"
                "hours,menu,photos,social_media,description,tips,timezone,"
                "hours_popular"
            )
        }

        response = requests.get(URL, headers=HEADERS, params=params)
        data = response.json()

        if "results" not in data or not data["results"]:
            print(f"No results found for {neighborhood}.")
            continue

        for place in data["results"]:
            fsq_id = safe_get(place, ["fsq_id"])
            # Skip if this venue has already been processed
            if fsq_id in seen_ids:
                continue
            seen_ids.add(fsq_id)

            writer.writerow([
                neighborhood, fsq_id, fix_arabic_text(safe_get(place, ["name"])),
                fix_arabic_text(safe_get(place, ["location", "formatted_address"])),
                fix_arabic_text(safe_get(place, ["location", "locality"])),
                fix_arabic_text(safe_get(place, ["location", "region"])),
                fix_arabic_text(safe_get(place, ["location", "country"])),
                safe_get(place, ["geocodes", "main", "latitude"]),
                safe_get(place, ["geocodes", "main", "longitude"]),
                fix_arabic_text(safe_get(place, ["categories", 0, "name"])),
                safe_get(place, ["rating"], default="N/A"),
                safe_get(place, ["stats", "total_ratings"], default="0"),
                safe_get(place, ["stats", "total_photos"], default="0"),
                safe_get(place, ["stats", "total_tips"], default="0"),
                safe_get(place, ["popularity"], default="N/A"),
                safe_get(place, ["verified"], default=False),
                safe_get(place, ["website"]),
                safe_get(place, ["tel"]),
                safe_get(place, ["price"]),
                safe_get(place, ["hours", "display"]),
                "; ".join([f"Day {entry.get('day', 'N/A')}: {entry.get('open', 'N/A')} - {entry.get('close', 'N/A')}" 
                           for entry in safe_get(place, ["hours_popular"], default=[])]),
                safe_get(place, ["menu"]),
                safe_get(place, ["social_media"]),
                fix_arabic_text(safe_get(place, ["description"])),
                "N/A",  # Placeholder for Photo URL
                "N/A",  # Placeholder for User Tip
                safe_get(place, ["timezone"])
            ])
            total_records += 1
        
        time.sleep(2)

print(f"✅ Successfully saved {total_records} venue records to {csv_filename}.")
