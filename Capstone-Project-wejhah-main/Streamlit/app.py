import streamlit as st
import folium
from streamlit_folium import folium_static, st_folium
import pandas as pd
import os
import base64
import numpy as np
from math import radians, cos, sin, asin, sqrt


# Initialize session state keys if not already present
if "selected_activities" not in st.session_state:
    st.session_state.selected_activities = []
if "selected_event" not in st.session_state:
    st.session_state.selected_event = None
if "selected_panels" not in st.session_state:
    st.session_state.selected_panels = set()
# For our location, we store lat, lon, and the currently selected neighborhood.
if "lat" not in st.session_state:
    st.session_state.lat = None
if "lon" not in st.session_state:
    st.session_state.lon = None
if "neighborhood" not in st.session_state:
    st.session_state.neighborhood = None

# Use st.cache_data to cache the creation of the event map so that minor slider changes don't cause a full reload.
@st.cache_data(show_spinner=False)
def create_event_map(event_lat, event_lon, radius_val):
    m = folium.Map(location=[event_lat, event_lon], zoom_start=14)
    folium.Circle(
        location=[event_lat, event_lon],
        radius=radius_val * 1000,
        color="red",
        fill=True,
        fill_opacity=0.2
    ).add_to(m)
    return m

# Dummy update_panels function (replace with your own if needed)
def update_panels(panel_name, checked):
    if checked:
        st.session_state.selected_panels.add(panel_name)
    else:
        st.session_state.selected_panels.discard(panel_name)

def main():
    st.set_page_config(layout="wide")
    
    # Inject CSS for RTL direction, centering the title, and styling checkboxes/expanders
    st.markdown(
    """
    <style>
    body {
        background-color: #ece2d9 !important;  /* Light beige background */
    }
    .stApp {
        background-color: #ece2d9 !important;  /* Ensuring background color for the whole app */
    }
    .stCheckbox > label {
    display: block;
    background-color: ##C67B78;  /* Set the green background */
    color: white;
    text-align: center;
    padding: 10px;
    margin: 10px 0;
    border-radius: 5px;
    cursor: pointer;
    transition: background-color 0.3s ease;
    border: 1px solid ##C67B78;  /* Border in the same green */
}

.stCheckbox > label:hover {
    background-color: ##C67B78;  /* Slightly darker green when hovered */
}

/* Hide the default checkbox but keep the label visible */
.stCheckbox input[type='checkbox'] {
    display: none;
}

/* For the selected checkbox (when checked) */
.stCheckbox input[type="checkbox"]:checked + label {
    background-color: #C67B78  /* Darker green for checked state */
}

/* Ensure the container of the checkbox is green as well */
.stCheckbox {
    background-color: #C67B78;  /* Green background for container */
}

/* Additional general styles */
html { direction: rtl; }
.stTitle { text-align: center; }

/* Styling the expanders */
.stExpander {
    background-color: #C67B78;
    padding: 15px;
    border-radius: 10px;
    margin-top: 10px;
}

.stExpander:hover {
    background-color: #C67B78;
}

/* Styling radio buttons */
.stRadio label {
    padding: 10px;
    border-radius: 8px;
    margin-bottom: 8px;
    transition: background-color 0.3s ease;
}

/* Additional map styling */
.stMap { 
    border-radius: 15px;
    padding: 10px;
}

    </style>
    """,
    unsafe_allow_html=True,
)
    

    st.image("logo.png", caption=" ", use_container_width=True)

    # Create tabs for scheduling by event and by location
    tab1, tab2 = st.tabs(["الجدول حسب الايفت", "الجدول حسب موقعك"])

    with tab1:
        st.header("🎉 وش الفعالية اللي ودك تحضرها؟")
        # Load events data
        try:
            df = pd.read_csv("all_clean_events.csv")
        except FileNotFoundError:
            st.error("Data file not found. Please ensure 'all_clean_events-2.csv' exists.")
            st.stop()
        
        # Mapping: English activity types to Arabic labels
        activity_mapping = {
            "performence": "أداء",                   # performance
            "music": "موسيقى",                       # music
            "experience": "تجربة",                   # experience
            "activite": "نشاط",                      # activite
            "Exhibitions and festivals": "معارض ومهرجانات",  # Exhibitions and festivals
            "sport": "رياضة"                         # sport
        }
        
        # Get unique activity types from the DataFrame.
        unique_activity_types = df["type"].unique().tolist()

        # Number of columns per row for checkboxes.
        columns_per_row = 3

        # Initialize selected activities if not in session state.
        if "selected_activities" not in st.session_state:
            st.session_state.selected_activities = []
        
        # Ensure selected_panels is available for recommendations.
        if "selected_panels" not in st.session_state:
            st.session_state.selected_panels = set()

        # Create columns for activity checkboxes.
        cols = st.columns(columns_per_row)

        # Loop through each activity type; show Arabic label.
        for i, activity_type in enumerate(unique_activity_types):
            arabic_label = activity_mapping.get(activity_type, activity_type.capitalize())
            with cols[i % columns_per_row]:
                is_checked = st.checkbox(
                    arabic_label, 
                    key=activity_type, 
                    value=(activity_type in st.session_state.selected_activities)
                )
                if is_checked:
                    if activity_type not in st.session_state.selected_activities:
                        st.session_state.selected_activities.append(activity_type)
                else:
                    st.session_state.selected_activities = [
                        a for a in st.session_state.selected_activities if a != activity_type
                    ]

        # -----------------------------
        # Event selection and map display
        # -----------------------------
        if st.session_state.selected_activities:
            filtered_df = df[df["type"].isin(st.session_state.selected_activities)]
            event_names = list(filtered_df["names"].unique())
            if event_names:
                selected_event = st.selectbox(" ", ["-- اختار الفعالية --"] + event_names, key="selected_event")
                if selected_event != "-- اختار الفعالية --":
                    selected_event_df = filtered_df[filtered_df["names"] == selected_event]
                    if not selected_event_df.empty:
                        event_details = selected_event_df.iloc[0]
                    else:
                        event_details = {}
                else:
                    event_details = {}
            else:
                st.write("No events available for the selected activity types.")
                event_details = {}
            
            if isinstance(event_details, pd.Series) and not event_details.empty:
                st.markdown(
                    f"""
                    <div style="border: 2px solid #ccc; padding: 10px; border-radius:5px;">
                        <h3>{event_details.get('names', 'N/A')}</h3>
                        <p><strong>Date:</strong> {event_details.get('date', 'N/A')}</p>
                        <p><strong>Price:</strong> {event_details.get('price', 'N/A')}</p>
                        <p><strong>URL:</strong> <a href="{event_details.get('event_link', '#')}" target="_blank">{event_details.get('event_link', 'N/A')}</a></p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if "Latitude" in event_details and "Longitude" in event_details:
                    user_location = {"lat": event_details["Latitude"], "lon": event_details["Longitude"]}
                    # Build the map later using the slider radius stored in st.session_state.
                    event_map = folium.Map(location=[user_location["lat"], user_location["lon"]], zoom_start=14)
                    # Use st.session_state.radius_km once set below.
                    folium.Circle(
                        location=[user_location["lat"], user_location["lon"]],
                        radius=st.session_state.radius_km * 1000,
                        color="blue",
                        fill=True,
                        fill_opacity=0.2
                    ).add_to(event_map)
                    st.write("### موقع الفعالية")
                    folium_static(event_map, width=700, height=500)
                else:
                    st.write("Event location not available.")
            else:
                st.write("⚠️ لازم تختار فعالية في الأول")
        else:
            st.write("⚠️ لازم تختار على الاقل فعالية واحدة")
        
        st.write("##### 🗓️ يلا نسوي جدولك")
        st.markdown("<h1 style='font-size: 24px;'>🚗قد ايش ودك تبعد عن الفعالية؟:</h1>", unsafe_allow_html=True)

        max_radius = 5.0
        min_radius = 1.0
        # Compute slider value and store radius_km in session state.
        reversed_slider_value = st.slider(
            " ",
            0.0,
            max_radius - min_radius,
            max_radius - 1.0,
            0.5,
            key="unique_slider_key"
        )
        radius_km = max_radius - reversed_slider_value
        st.session_state.radius_km = radius_km  # Store for later use
        st.write(f"بندَور لك اماكن داخل هذه النطاق ({radius_km} km )")
        
        st.write("#### وين ودك تروح بعد الفعالية؟:")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            mall_checked = st.checkbox("تبي تروح تتسوق؟", key="rec2_mall_checkbox_unique")
            if mall_checked:
                st.session_state.selected_panels.add("Mall")
            else:
                st.session_state.selected_panels.discard("Mall")
        with col2:
            coffee_checked = st.checkbox("تبي تروح تتقهوى؟", key="rec2_coffee_checkbox_unique")
            if coffee_checked:
                st.session_state.selected_panels.add("Coffee")
            else:
                st.session_state.selected_panels.discard("Coffee")
        with col3:
            restaurant_checked = st.checkbox("تبي تروح تاكل؟", key="rec2_restaurant_checkbox_unique")
            if restaurant_checked:
                st.session_state.selected_panels.add("Restaurant")
            else:
                st.session_state.selected_panels.discard("Restaurant")
        with col4:
            cinema_checked = st.checkbox("تبي تشوف موڤي؟", key="rec2_cinema_checkbox_unique")
            if cinema_checked:
                st.session_state.selected_panels.add("Cinema")
            else:
                st.session_state.selected_panels.discard("Cinema")

        # Venue categories based on selected panels.
        venue_categories = []
        if "Mall" in st.session_state.selected_panels or "Cinema" in st.session_state.selected_panels:
            if "Mall" in st.session_state.selected_panels and "Cinema" not in st.session_state.selected_panels:
                venue_categories = ["Shopping Mall"]
                st.write("المولات")
            elif "Cinema" in st.session_state.selected_panels and "Mall" not in st.session_state.selected_panels:
                st.write("سينما")
            elif "Mall" in st.session_state.selected_panels and "Cinema" in st.session_state.selected_panels:
                venue_categories = ["Shopping Mall", "Movie Theater"]
                st.write("مولات وسينما")
        else:
            venue_categories = []

        # Always define radio button labels for coffee and restaurant preferences.
        if "Coffee" in st.session_state.selected_panels:
            st.write('مقاهي')
            coffee_pref_label = st.radio(" ", ['تقيمه عالي لكن مو ترند', 'يكون ترند'], key="rec2_coffee_pref_unique")
        else:
            coffee_pref_label = 'تقيمه عالي لكن مو ترند'
            
        if "Restaurant" in st.session_state.selected_panels:
            try:
                df_rest = pd.read_csv("cleaned_restaurant.csv")
            except Exception as e:
                st.error(f"Error loading restaurant data: {e}")
                df_rest = pd.DataFrame()
            all_restaurant_categories = sorted(df_rest['Category'].dropna().unique()) if not df_rest.empty else []
            st.write('مطاعم')
            restaurant_categories = st.multiselect(" ", options=all_restaurant_categories, key="rec2_restaurant_categories_unique")
            restaurant_pref_label = st.radio(" ", ['تقيمه عالي لكن مو ترند', 'يكون ترند'], key="rec2_restaurant_pref_unique")
        else:
            restaurant_pref_label = 'تقيمه عالي لكن مو ترند'
        st.write("##### ⭐ ايش اكثر شي يهمك في المكان اللي ودك فية:")
        # Radio button for sorting (using a unique key)
        with col1:
            similarity_score = 'ابهرني'
        with col2:
            rating_button1 = 'الاعلى تقييم'
        with col3:
            distance_button1 = 'الاقل مسافه'

        sort_by = st.radio(
            '',
            options=[similarity_score, rating_button1, distance_button1],
            index=0,
            key="rec2_sort_by_unique",
            label_visibility="collapsed"
        )
        if sort_by == similarity_score:
            sort_by_internal = 'similarity_score'
        elif sort_by == rating_button1:
            sort_by_internal = 'Rating'
        elif sort_by == distance_button1:
            sort_by_internal = 'distance_km'
        
        # Convert radio button selections for preferences into internal codes.
        coffee_preference = "hidden_gem" if coffee_pref_label == 'تقيمه عالي لكن مو ترند' else "trending"
        restaurant_preference = "hidden_gem" if restaurant_pref_label == 'تقيمه عالي لكن مو ترند' else "trending"
        
        # -----------------------------
        # Recommendation logic
        # -----------------------------
        if st.button("اقتراحاتنا لك"):
            def haversine_distance(lat1, lon1, lat2, lon2):
                lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
                c = 2 * asin(sqrt(a))
                return c * 6371

            def min_max_scale(series):
                diff = series.max() - series.min()
                if diff == 0:
                    return pd.Series([1] * len(series), index=series.index)
                return (series - series.min()) / diff

            def weighted_cosine_similarity(u, v, weights, eps=1e-8):
                weighted_u = u * np.sqrt(weights)
                weighted_v = v * np.sqrt(weights)
                dot_product = np.dot(weighted_u, weighted_v)
                norm_u = np.linalg.norm(weighted_u) + eps
                norm_v = np.linalg.norm(weighted_v) + eps
                return dot_product / (norm_u * norm_v)

            def get_coffee_recommendations(df_coffee, user_location, radius, preference, sort_by):
                df = df_coffee.copy()
                df['distance_km'] = df.apply(lambda row: haversine_distance(user_location['lat'], user_location['lon'],
                                                                            row['Latitude'], row['Longitude']), axis=1)
                df = df[df['distance_km'] <= radius].copy()
                if df.empty:
                    st.write(f"No coffee venues within {radius} km.")
                    return None
                for col in ['Rating', 'Total Ratings', 'Popularity Score', 'Total Photos', 'Total Tips']:
                    df[col] = df[col].fillna(0)
                df['norm_rating'] = df['Rating'] / 5.0
                df['engagement'] = df['Total Photos'] + df['Total Tips']
                df['norm_engagement'] = min_max_scale(df['engagement'])
                df['norm_total_ratings'] = min_max_scale(df['Total Ratings'])
                df['norm_popularity'] = min_max_scale(df['Popularity Score'])
                threshold = 0.5
                if preference == "hidden_gem":
                    df_cat = df[df['norm_rating'] >= 0.7].copy()
                    df_cat['feature_total_ratings'] = 1 - df_cat['norm_total_ratings']
                    df_cat['feature_popularity'] = 1 - df_cat['norm_popularity']
                    df_cat['feature_engagement'] = 1 - df_cat['norm_engagement']
                else:
                    df_cat = df[(df['norm_rating'] >= 0.7) & ((df['norm_engagement'] >= threshold) | (df['norm_popularity'] >= threshold))].copy()
                    df_cat['feature_total_ratings'] = df_cat['norm_total_ratings']
                    df_cat['feature_popularity'] = df_cat['norm_popularity']
                    df_cat['feature_engagement'] = df_cat['norm_engagement']
                feature_cols = ['norm_rating', 'feature_total_ratings', 'feature_popularity', 'feature_engagement']
                weights = np.array([1.0, 1.0, 1.0, 1.0])
                ideal_vector = np.ones(len(feature_cols))
                df_cat['base_score'] = df_cat[feature_cols].apply(lambda x: weighted_cosine_similarity(ideal_vector, x.values, weights), axis=1)
                df_cat['base_score'] *= 0.8
                df_cat['similarity_score'] = df_cat['base_score']
                if sort_by in df_cat.columns:
                    ascending_order = True if sort_by == 'distance_km' else False
                    df_cat = df_cat.sort_values(by=sort_by, ascending=ascending_order).head(10)
                else:
                    df_cat = df_cat.sort_values(by='similarity_score', ascending=False).head(10)
                return df_cat

            def get_restaurant_recommendations(df_rest, user_location, radius, chosen_categories, preference, sort_by):
                df = df_rest.copy()
                if chosen_categories:
                    df = df[df['Category'].isin(chosen_categories)].copy()
                if df.empty:
                    st.write("No restaurants found for the chosen category/categories.")
                    return None
                df['distance_km'] = df.apply(lambda row: haversine_distance(user_location['lat'], user_location['lon'],
                                                                            row['Latitude'], row['Longitude']), axis=1)
                df = df[df['distance_km'] <= radius].copy()
                if df.empty:
                    st.write(f"No restaurants within {radius} km.")
                    return None
                df['distance_score'] = df['distance_km'].apply(lambda d: max(0, 1 - ((d if d >= 2 else 2) / radius)))
                for col in ['Rating','Total Ratings','Popularity Score','Total Photos','Total Tips']:
                    df[col] = df[col].fillna(0)
                df['norm_rating'] = df['Rating'] / 5.0
                df['engagement'] = df['Total Photos'] + df['Total Tips']
                df['norm_engagement'] = min_max_scale(df['engagement'])
                df['norm_total_ratings'] = min_max_scale(df['Total Ratings'])
                df['norm_popularity'] = min_max_scale(df['Popularity Score'])
                if preference == "hidden_gem":
                    df['feature_total_ratings'] = 1 - df['norm_total_ratings']
                    df['feature_popularity'] = 1 - df['norm_popularity']
                    df['feature_engagement'] = 1 - df['norm_engagement']
                else:
                    df['feature_total_ratings'] = df['norm_total_ratings']
                    df['feature_popularity'] = df['norm_popularity']
                    df['feature_engagement'] = df['norm_engagement']
                feature_cols = ['norm_rating','feature_total_ratings','feature_popularity','feature_engagement','distance_score']
                weights = np.array([1.0]*5)
                ideal_vec = np.ones(len(feature_cols))
                df['base_score'] = df[feature_cols].apply(lambda x: weighted_cosine_similarity(ideal_vec, x.values, weights), axis=1)
                df['base_score'] *= 0.8
                df['similarity_score'] = df['base_score']
                if sort_by in df.columns:
                    ascending_order = True if sort_by=='distance_km' else False
                    df = df.sort_values(by=sort_by, ascending=ascending_order).head(10)
                else:
                    df = df.sort_values(by='similarity_score', ascending=False).head(10)
                return df

            def get_venue_recommendations(df_venue, user_location, radius, chosen_categories):
                df = df_venue.copy()
                if chosen_categories:
                    df = df[df['Category'].isin(chosen_categories)].copy()
                if df.empty:
                    st.write("No venues found for the chosen categories.")
                    return None
                df['distance_km'] = df.apply(lambda row: haversine_distance(user_location['lat'], user_location['lon'],
                                                                            row['Latitude'], row['Longitude']), axis=1)
                df = df[df['distance_km'] <= radius].copy()
                if df.empty:
                    st.write(f"No venues found within {radius} km.")
                    return None
                df['distance_score'] = df['distance_km'].apply(lambda d: max(0, 1 - (d / radius)))
                df['similarity_score'] = 0.4 * df['distance_score']
                df = df.sort_values(by='similarity_score', ascending=False).head(10)
                return df

            # Build the map using event location and the computed radius.
            event_map = folium.Map(location=[user_location["lat"], user_location["lon"]], zoom_start=14)
            folium.Circle(
                location=[user_location["lat"], user_location["lon"]],
                radius=st.session_state.radius_km * 1000,
                color="blue",
                fill=True,
                fill_opacity=0.2
            ).add_to(event_map)
            folium.Marker(
                location=[user_location["lat"], user_location["lon"]],
                popup=f"Event: {event_details['names']}",
                icon=folium.Icon(color="blue", icon="info-sign")
            ).add_to(event_map)
            
            # Coffee recommendations
            if "Coffee" in st.session_state.selected_panels:
                st.markdown("**توصيات Coffee**")
                try:
                    df_coffee = pd.read_csv("cleaned_coffee.csv")
                except Exception as e:
                    st.error(f"Error loading coffee data: {e}")
                    df_coffee = pd.DataFrame()
                if not df_coffee.empty:
                    rec_coffee = get_coffee_recommendations(df_coffee, user_location, st.session_state.radius_km, 
                                                            coffee_preference, sort_by_internal)
                    if rec_coffee is not None:
                        for _, row in rec_coffee.iterrows():
                            folium.Marker(
                                location=[row['Latitude'], row['Longitude']],
                                popup=f"Coffee: {row['Name']}<br>Rating: {row['Rating']}<br>Distance: {row['distance_km']} km",
                                icon=folium.Icon(color="green", icon="coffee", prefix="fa")
                            ).add_to(event_map)
                            st.markdown(f"""
                            <div style="border:1px solid #ccc; padding:10px; border-radius:5px; margin-bottom:10px;">
                                <h4>{row['Name']}</h4>
                                <p>التصنيف: {row['Category']}</p>
                                <p>التقييم: {row['Rating']} - عدد التقييمات: {row['Total Ratings']}</p>
                                <p>المسافة: {row['distance_km']:.2f} </p>
                            </div>
                            """, unsafe_allow_html=True)

            # Restaurant recommendations
            if "Restaurant" in st.session_state.selected_panels:
                st.markdown("**توصيات Restaurant**")
                try:
                    df_rest = pd.read_csv("cleaned_restaurant.csv")
                except Exception as e:
                    st.error(f"Error loading restaurant data: {e}")
                    df_rest = pd.DataFrame()
                if not df_rest.empty:
                    chosen_rest_categories = st.session_state.get("restaurant_categories", [])
                    rec_rest = get_restaurant_recommendations(df_rest, user_location, st.session_state.radius_km, 
                                                            chosen_rest_categories, restaurant_preference, sort_by_internal)
                    if rec_rest is not None:
                        for _, row in rec_rest.iterrows():
                            folium.Marker(
                                location=[row['Latitude'], row['Longitude']],
                                popup=f"Restaurant: {row['Name']}<br>Category: {row['Category']}<br>Rating: {row['Rating']}",
                                icon=folium.Icon(color="red", icon="cutlery", prefix="fa")
                            ).add_to(event_map)
                            st.markdown(f"""
                            <div style="border:1px solid #ccc; padding:10px; border-radius:5px; margin-bottom:10px;">
                                <h4>{row['Name']}</h4>
                                <p>التصنيف: {row['Category']}</p>
                                <p>التقييم: {row['Rating']} - عدد التقييمات: {row['Total Ratings']}</p>
                                <p>المسافة: {row['distance_km']:.2f} </p>
                            </div>
                            """, unsafe_allow_html=True)

            # Venue recommendations
            if "Mall" in st.session_state.selected_panels or "Cinema" in st.session_state.selected_panels:
                st.markdown("**توصيات Venue**")
                try:
                    df_venue = pd.read_csv("cleaned_venues.csv")
                except Exception as e:
                    st.error(f"Error loading venue data: {e}")
                    df_venue = pd.DataFrame()
                if not df_venue.empty:
                    rec_venue = get_venue_recommendations(df_venue, user_location, st.session_state.radius_km, venue_categories)
                    if rec_venue is not None:
                        for _, row in rec_venue.iterrows():
                            if row['Category'] in ["Shopping Mall"]:
                                icon = folium.Icon(color="blue", icon="shopping-bag", prefix="fa")
                            elif row['Category'] in ["Movie Theater"]:
                                icon = folium.Icon(color="purple", icon="film", prefix="fa")
                            else:
                                icon = folium.Icon(color="blue", icon="building", prefix="fa")
                            folium.Marker(
                                location=[row['Latitude'], row['Longitude']],
                                popup=f"Venue: {row['Name']}<br>Category: {row['Category']}<br>Distance: {row['distance_km']} km",
                                icon=icon
                            ).add_to(event_map)
                            st.markdown(f"""
                            <div style="border:1px solid #ccc; padding:10px; border-radius:5px; margin-bottom:10px;">
                                <h4>{row['Name']}</h4>
                                <p>التصنيف: {row['Category']}</p>
                                <p>المسافة: {row['distance_km']:.2f} </p>
                            </div>
                            """, unsafe_allow_html=True)

            st.write("### كيف زبطناك؟ 😎")
            folium_static(event_map, width=700, height=500)






    with tab2:
        st.markdown("<h1 style='font-size: 24px;'>📍 انت في اي حي؟</h1>", unsafe_allow_html=True)

        # Full list of Riyadh neighborhoods (name, latitude, longitude)
        riyadh_neighborhoods = [
        ("الصحافة", 24.803329, 46.639133),
        ("الربيع", 24.795393, 46.658502),
        ("النخيل", 24.805883, 46.683010),
        ("النرجس", 24.8868, 46.6453),
        ("العارض", 24.9410, 46.5580),
        ("النفل", 24.782061, 46.673138),
        ("العقيق", 24.773696, 46.630278),
        ("الوادي", 24.789435, 46.691111),
        ("الغدير", 24.774338, 46.655725),
        ("الياسمين", 24.825492, 46.645506),
        ("الفلاح", 24.796854, 46.708911),
        ("بنبان", 25.1000, 46.5000),
        ("القيروان", 24.835707, 46.588102),
        ("حي حطين", 24.763594, 46.603193),
        ("الملقا", 24.800978, 46.599884),
        ("الروضة", 24.735197, 46.766606),
        ("الرمال", 24.868298, 46.819176),
        ("المونسية", 24.830625, 46.768180),
        ("قرطبة", 24.816610, 46.732107),
        ("الجنادرية", 24.874620, 46.914538),
        ("القديسية", 24.820355, 46.832171),
        ("اليارموك", 24.806966, 46.780283),
        ("غرناطة", 24.792252, 46.743551),
        ("إشبيلية", 24.795295, 46.797018),
        ("الحمراء", 24.777299, 46.755940),
        ("المعيصم", 24.794113, 46.847337),
        ("الخليج", 24.774367, 46.802162),
        ("الفيصل", 24.762708, 46.775200),
        ("القدس", 24.757108, 46.753978),
        ("النهضة", 24.760764, 46.813228),
        ("الأندلس", 24.743469, 46.790225),
        ("شرق النسيم", 24.741012, 46.842088),
        ("غرب النسيم", 24.725588, 46.826113),
        ("السلام", 24.707980, 46.812565),
        ("الريان", 24.711461, 46.777113),
        ("أم الحمام (الشرق)", 24.684976, 46.658778),
        ("أم الحمام (الغرب)", 24.690595, 46.642350),
        ("الزهراء", 24.588095, 46.777443),
        ("طيبة (الطائبة)", 24.554506, 46.814119),
        ("الدار البيضاء", 24.559590, 46.794021),
        ("الشفا", 24.564974, 46.699419),
        ("بدر", 24.526356, 46.692448),
        ("المروة", 24.543875, 46.678347),
        ("الحزم", 24.541416, 46.649821),
        ("المنصور", 24.515338, 46.790894),
        ("ديراب", 24.506087, 46.621640),
        ("العريجاء", 24.627045, 46.658139),
        ("حجرة وادي لبن", 24.608211, 46.548163),
        ("ظهرة لبن", 24.634237, 46.548403),
        ("شبرا", 24.574487, 46.665164),
        ("السويدي", 24.589750, 46.669655),
        ("السويدي الغربي", 24.573917, 46.622862),
        ("ظهرة البادية", 24.597627, 46.645941),
        ("البادية", 24.617597, 46.677971),
        ("سلطانة", 24.602795, 46.688851),
        ("الزهراء", 24.686516, 46.730357),
        ("نمار", 24.576014, 46.689328),
        ("ظهرة نمار", 24.559673, 46.608500),
        ("تويقة", 24.577483, 46.556666),
        ("الملز", 24.662771, 46.732030),
        ("الربوة", 24.691833, 46.758159),
        ("جارير", 24.677952, 46.751709),
        ("الصفا", 24.666869, 46.763167),
        ("الضباط", 24.679357, 46.721530),
        ("الوزارات", 24.674183, 46.712200),
        ("الفروق", 24.651466, 46.771769),
        ("الأمل", 24.643030, 46.719084),
        ("الثليم", 24.640868, 46.727866),
        ("المربع", 24.662535, 46.707007),
        ("الفطا", 24.641475, 46.708730),
        ("المروج", 24.757602, 46.663114),
        ("المرسلات", 24.746842, 46.690077),
        ("النزهة", 24.755813, 46.707651),
        ("المغرزات", 24.763837, 46.725299),
        ("الورود", 24.724934, 46.679873),
        ("حي الملك سلمان", 24.737337, 46.705895),
        ("العلية", 24.695890, 46.680222),
        ("السليمانية", 24.703162, 46.708742),
        ("حي الملك عبد العزيز", 24.715223, 46.734389)
    ]

        # Sort by neighborhood name
        riyadh_neighborhoods_arabic_sorted = sorted(riyadh_neighborhoods, key=lambda x: x[0])

        neighborhood_names = [n[0] for n in riyadh_neighborhoods_arabic_sorted]
        selected_neighborhood_name = st.selectbox("اسم الحي:", neighborhood_names)
        selected = next((n for n in riyadh_neighborhoods_arabic_sorted if n[0] == selected_neighborhood_name), None)
        if selected:
            orig_lat, orig_lon = selected[1], selected[2]
            if (st.session_state.lat is None) or (st.session_state.neighborhood != selected_neighborhood_name):
                st.session_state.lat = orig_lat
                st.session_state.lon = orig_lon
                st.session_state.neighborhood = selected_neighborhood_name

            # Use one slider for the interactive map circle radius (this same radius is also used for recommendations)
            #radius_km = st.slider("Radius (km):", 1.0, 5.0, 1.0, 0.5)
            st.markdown("<h1 style='font-size: 24px;'>🚗 حدد لي النطاق الي حاب ادور فيه:</h1>", unsafe_allow_html=True)

            max_radius = 5.0
            min_radius = 1.0
            # Reversed slider (internally)
            reversed_slider_value = st.slider(
                " ",
                0.0,  # Start at 0
                max_radius - min_radius,  # Maximum range
                max_radius - 1.0,  # Initial reversed value
                0.5,
            )
            st.markdown(
                """
                <style>
                .stSlider > div > div > div > div[data-baseweb="slider-thumb"] {
                    direction: rtl;
                }
                .stSlider > div > div > div > div[data-baseweb="slider-runway"] > div {
                    direction: rtl;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            
            # Calculate the actual radius
            radius_km = max_radius - reversed_slider_value

            st.write(f"بندَور لك اماكن داخل هذه النطاق ({radius_km} km )")
            def create_map(lat, lon):
                m = folium.Map(location=[lat, lon], zoom_start=14)
                folium.Marker(location=[lat, lon],
                            popup=selected_neighborhood_name,
                            draggable=True).add_to(m)
                folium.Circle(location=[lat, lon],
                            radius=radius_km * 1000,
                            color="blue",
                            fill=True,
                            fill_color="blue",
                            fill_opacity=0.2).add_to(m)
                return m
            st.markdown("<h1 style='font-size: 24px;'>حاب نكمل ندور لك عن اماكن بنفس الحي اللي اخترتة! ولا ودك تعدل موقعك؟</h1>", unsafe_allow_html=True)

            map_placeholder = st.empty()
            m = create_map(st.session_state.lat, st.session_state.lon)
            with map_placeholder:
                data = st_folium(m, key="map", width=700, height=500)
            if data:
                new_coords = data.get("last_object_clicked") or data.get("last_clicked")
                if new_coords:
                    if new_coords["lat"] != st.session_state.lat or new_coords["lng"] != st.session_state.lon:
                        st.session_state.lat = new_coords["lat"]
                        st.session_state.lon = new_coords["lng"]
                        map_placeholder.empty()
                        m = create_map(st.session_state.lat, st.session_state.lon)
                        with map_placeholder:
                            st_folium(m, key="map_updated", width=700, height=500)
            #st.write("Marker Coordinates:")
            #st.write(f"Latitude: {st.session_state.lat}, Longitude: {st.session_state.lon}")
        else:
            st.write("Neighborhood not found.")

        # --- RECOMMENDATION LOGIC USING THE NEW LOCATION ---
        # Use the same radius value as the interactive map
        rec_radius = radius_km  
        user_location = {"lat": st.session_state.lat, "lon": st.session_state.lon}

        
        
        st.write("##### 🗓️ يلا نسوي جدولك")
        
     

        if 'selected_panels' not in st.session_state:
            st.session_state.selected_panels = set()

        st.write("##### 🚗 وين ودك تروح؟")


        col1, col2, col3, col4 = st.columns(4)

        with col1:
            mall_checked = st.checkbox("تبي تروح تتسوق؟", key="rec2_mall_checkbox")
            if mall_checked:
                st.session_state.selected_panels.add("Mall")
            else:
                st.session_state.selected_panels.discard("Mall")

        with col2:
            coffee_checked = st.checkbox("تبي تروح تتقهوى؟", key="rec2_coffee_checkbox")
            if coffee_checked:
                st.session_state.selected_panels.add("Coffee")
            else:
                st.session_state.selected_panels.discard("Coffee")

        with col3:
            restaurant_checked = st.checkbox("تبي تروح تاكل؟", key="rec2_restaurant_checkbox")
            if restaurant_checked:
                st.session_state.selected_panels.add("Restaurant")
            else:
                st.session_state.selected_panels.discard("Restaurant")

        with col4:
            cinema_checked = st.checkbox("تبي تشوف موڤي؟", key="rec2_cinema_checkbox")
            if cinema_checked:
                st.session_state.selected_panels.add("Cinema")
            else:
                st.session_state.selected_panels.discard("Cinema")

        # Continue with your existing logic for displaying recommendations based on selected panels
        venue_categories = []
        if "Mall" in st.session_state.selected_panels or "Cinema" in st.session_state.selected_panels:
            #st.markdown("**🏬 Venue Options**")
            if "Mall" in st.session_state.selected_panels and "Cinema" not in st.session_state.selected_panels:
                venue_categories = ["Shopping Mall"]
                st.write("المولات")
            elif "Cinema" in st.session_state.selected_panels and "Mall" not in st.session_state.selected_panels:
                #venue_categories = ["Movie Theater"]
                st.write("سينما")
            elif "Mall" in st.session_state.selected_panels and "Cinema" in st.session_state.selected_panels:
                venue_categories = ["Shopping Mall", "Movie Theater"]
                st.write("مولات وسينما")

        else:
            venue_categories = []

        if "Coffee" in st.session_state.selected_panels:
            st.markdown("**☕ ايش تفضل في المقاهي؟**")
            coffee_pref = st.radio(" ", ['تقيمه عالي لكن مو ترند', 'يكون ترند'], key="rec2_coffee_pref")

        if "Restaurant" in st.session_state.selected_panels:
            st.markdown("**🍽️ ايش تفضل في المطاعم**")
            try:
                df_rest = pd.read_csv("cleaned_restaurant.csv")
            except Exception as e:
                st.error(f"Error loading restaurant data: {e}")
                df_rest = pd.DataFrame()
            all_restaurant_categories = sorted(df_rest['Category'].dropna().unique()) if not df_rest.empty else []
            restaurant_categories = st.multiselect(" ", options=all_restaurant_categories, key="rec2_restaurant_categories")
            restaurant_pref = st.radio(" ", ['تقيمه عالي لكن مو ترند', 'يكون ترند'], key="rec2_restaurant_pref")


        st.write("##### ⭐ ايش اكثر شي يهمك في المكان اللي ودك فية:")

        col1, col2, col3 = st.columns(3)

        # Radio button choices in the same row
        with col1:
            similarity_score_button = 'ابهرني'
        with col2:
            rating_button = 'الاعلى تقييم'
        with col3:
            distance_button = 'الاقل مسافه'

        # Combine the choices into one single radio button
        sort_by = st.radio(
            '',
            options=[similarity_score_button, rating_button, distance_button],
            index=0,  # Default to the first one
            key="rec1_sort_by",
            label_visibility="collapsed"
        )


# Default sorting to 'similarity_score'
        sort_by = 'similarity_score'

        # Update the sorting based on the button clicked
        if similarity_score_button:
            sort_by = 'similarity_score'
        elif rating_button:
            sort_by = 'Rating'
        elif distance_button:
            sort_by = 'distance_km'


        if st.button("اقتراحاتنا لك", key="rec2_get_recommendations"):

            # Helper functions for distance and scoring
            def haversine_distance(lat1, lon1, lat2, lon2):
                lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
                dlat = lat2 - lat1
                dlon = lon2 - lon1
                a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
                c = 2 * asin(sqrt(a))
                return c * 6371

            def min_max_scale(series):
                diff = series.max() - series.min()
                if diff == 0:
                    return pd.Series([1]*len(series), index=series.index)
                return (series - series.min()) / diff

            def weighted_cosine_similarity(u, v, weights, eps=1e-8):
                weighted_u = u * np.sqrt(weights)
                weighted_v = v * np.sqrt(weights)
                dot_product = np.dot(weighted_u, weighted_v)
                norm_u = np.linalg.norm(weighted_u) + eps
                norm_v = np.linalg.norm(weighted_v) + eps
                return dot_product / (norm_u * norm_v)

            def get_coffee_recommendations(df_coffee, user_location, radius, preference, sort_by):
                df = df_coffee.copy()
                df['distance_km'] = df.apply(lambda row: haversine_distance(user_location['lat'], user_location['lon'],
                                                                            row['Latitude'], row['Longitude']), axis=1)
                df = df[df['distance_km'] <= radius].copy()
                if df.empty:
                    st.write(f"No coffee venues within {radius} km.")
                    return None
                for col in ['Rating', 'Total Ratings', 'Popularity Score', 'Total Photos', 'Total Tips']:
                    df[col] = df[col].fillna(0)
                df['norm_rating'] = df['Rating'] / 5.0
                df['engagement'] = df['Total Photos'] + df['Total Tips']
                df['norm_engagement'] = min_max_scale(df['engagement'])
                df['norm_total_ratings'] = min_max_scale(df['Total Ratings'])
                df['norm_popularity'] = min_max_scale(df['Popularity Score'])
                threshold = 0.5
                if preference == "hidden_gem":
                    df_cat = df[df['norm_rating'] >= 0.7].copy()
                    df_cat['feature_total_ratings'] = 1 - df_cat['norm_total_ratings']
                    df_cat['feature_popularity'] = 1 - df_cat['norm_popularity']
                    df_cat['feature_engagement'] = 1 - df_cat['norm_engagement']
                else:
                    df_cat = df[(df['norm_rating'] >= 0.7) & ((df['norm_engagement'] >= threshold) | (df['norm_popularity'] >= threshold))].copy()
                    df_cat['feature_total_ratings'] = df_cat['norm_total_ratings']
                    df_cat['feature_popularity'] = df_cat['norm_popularity']
                    df_cat['feature_engagement'] = df_cat['norm_engagement']
                feature_cols = ['norm_rating', 'feature_total_ratings', 'feature_popularity', 'feature_engagement']
                weights = np.array([1.0, 1.0, 1.0, 1.0])
                ideal_vector = np.ones(len(feature_cols))
                df_cat['base_score'] = df_cat[feature_cols].apply(lambda x: weighted_cosine_similarity(ideal_vector, x.values, weights), axis=1)
                df_cat['base_score'] *= 0.8
                df_cat['similarity_score'] = df_cat['base_score']
                if sort_by in df_cat.columns:
                    ascending_order = True if sort_by == 'distance_km' else False
                    df_cat = df_cat.sort_values(by=sort_by, ascending=ascending_order).head(10)
                else:
                    df_cat = df_cat.sort_values(by='similarity_score', ascending=False).head(10)
                return df_cat

            def get_restaurant_recommendations(df_rest, user_location, radius, chosen_categories, preference, sort_by):
                df = df_rest.copy()
                if chosen_categories:
                    df = df[df['Category'].isin(chosen_categories)].copy()
                if df.empty:
                    st.write("No restaurants found for the chosen category/categories.")
                    return None
                df['distance_km'] = df.apply(lambda row: haversine_distance(user_location['lat'], user_location['lon'],
                                                                            row['Latitude'], row['Longitude']), axis=1)
                df = df[df['distance_km'] <= radius].copy()
                if df.empty:
                    st.write(f"No restaurants within {radius} km.")
                    return None
                df['distance_score'] = df['distance_km'].apply(lambda d: max(0, 1 - ((d if d >= 2 else 2) / radius)))
                for col in ['Rating', 'Total Ratings', 'Popularity Score', 'Total Photos', 'Total Tips']:
                    df[col] = df[col].fillna(0)
                df['norm_rating'] = df['Rating'] / 5.0
                df['engagement'] = df['Total Photos'] + df['Total Tips']
                df['norm_engagement'] = min_max_scale(df['engagement'])
                df['norm_total_ratings'] = min_max_scale(df['Total Ratings'])
                df['norm_popularity'] = min_max_scale(df['Popularity Score'])
                if preference == "hidden_gem":
                    df['feature_total_ratings'] = 1 - df['norm_total_ratings']
                    df['feature_popularity'] = 1 - df['norm_popularity']
                    df['feature_engagement'] = 1 - df['norm_engagement']
                else:
                    df['feature_total_ratings'] = df['norm_total_ratings']
                    df['feature_popularity'] = df['norm_popularity']
                    df['feature_engagement'] = df['norm_engagement']
                feature_cols = ['norm_rating', 'feature_total_ratings', 'feature_popularity', 'feature_engagement', 'distance_score']
                weights = np.array([1.0] * 5)
                ideal_vec = np.ones(len(feature_cols))
                df['base_score'] = df[feature_cols].apply(lambda x: weighted_cosine_similarity(ideal_vec, x.values, weights), axis=1)
                df['base_score'] *= 0.8
                df['similarity_score'] = df['base_score']
                if sort_by in df.columns:
                    ascending_order = True if sort_by == 'distance_km' else False
                    df = df.sort_values(by=sort_by, ascending=ascending_order).head(10)
                else:
                    df = df.sort_values(by='similarity_score', ascending=False).head(10)
                return df

            def get_venue_recommendations(df_venue, user_location, radius, chosen_categories):
                df = df_venue.copy()
                if chosen_categories:
                    df = df[df['Category'].isin(chosen_categories)].copy()
                if df.empty:
                    st.write("No venues found for the chosen categories.")
                    return None
                df['distance_km'] = df.apply(lambda row: haversine_distance(user_location['lat'], user_location['lon'],
                                                                            row['Latitude'], row['Longitude']), axis=1)
                df = df[df['distance_km'] <= radius].copy()
                if df.empty:
                    st.write(f"No venues found within {radius} km.")
                    return None
                df['distance_score'] = df['distance_km'].apply(lambda d: max(0, 1 - (d / radius)))
                df['similarity_score'] = 0.4 * df['distance_score']
                df = df.sort_values(by='similarity_score', ascending=False).head(10)
                return df

            # Create a single combined map with neighborhood marker, radius, and recommendations
            combined_map = folium.Map(location=[st.session_state.lat, st.session_state.lon], zoom_start=14)

            # Draw the neighborhood's radius and add marker
            folium.Circle(
                location=[st.session_state.lat, st.session_state.lon],
                radius=rec_radius * 1000,
                color="blue",
                fill=True,
                fill_color="blue",
                fill_opacity=0.2
            ).add_to(combined_map)
            folium.Marker(
                location=[st.session_state.lat, st.session_state.lon],
                popup=selected_neighborhood_name,
                icon=folium.Icon(color="orange", icon="map-marker", prefix="fa")
            ).add_to(combined_map)

          

            # Add Coffee recommendations markers with a coffee cup icon
            if "Coffee" in st.session_state.selected_panels:
                try:
                    df_coffee = pd.read_csv("cleaned_coffee.csv")
                except Exception as e:
                    st.error(f"Error loading coffee data: {e}")
                    df_coffee = pd.DataFrame()
                if not df_coffee.empty:
                    rec_coffee = get_coffee_recommendations(
                        df_coffee, user_location, rec_radius,
                        st.session_state.get("rec2_coffee_pref", "hidden_gem"), sort_by
                    )
                    if rec_coffee is not None:
                        for _, row in rec_coffee.iterrows():
                            folium.Marker(
                                location=[row['Latitude'], row['Longitude']],
                                popup=f"Coffee: {row['Name']}<br>Rating: {row['Rating']}<br>Distance: {row['distance_km']:.2f} km",
                                icon=folium.Icon(color="green", icon="coffee", prefix="fa")
                            ).add_to(combined_map)
                            st.markdown(f"""
                            <div style="border:1px solid #ccc; padding:10px; border-radius:5px; margin-bottom:10px;">
                                <h4>{row['Name']}</h4>
                                <p>التصنيف: {row['Category']}</p>
                                <p>التقييم: {row['Rating']} - عدد التقييمات: {row['Total Ratings']}</p>
                                <p>المسافة: {row['distance_km']:.2f} </p>
                            </div>
                            """, unsafe_allow_html=True)

            # Add Restaurant recommendations markers with a cutlery icon
            if "Restaurant" in st.session_state.selected_panels:
                try:
                    df_rest = pd.read_csv("cleaned_restaurant.csv")
                except Exception as e:
                    st.error(f"Error loading restaurant data: {e}")
                    df_rest = pd.DataFrame()
                if not df_rest.empty:
                    chosen_rest_categories = st.session_state.get("rec2_restaurant_categories", [])
                    rec_rest = get_restaurant_recommendations(
                        df_rest, user_location, rec_radius,
                        chosen_rest_categories, st.session_state.get("rec2_restaurant_pref", "hidden_gem"), sort_by
                    )
                    if rec_rest is not None:
                        for _, row in rec_rest.iterrows():
                            folium.Marker(
                                location=[row['Latitude'], row['Longitude']],
                                popup=f"Restaurant: {row['Name']}<br>Category: {row['Category']}<br>Rating: {row['Rating']}",
                                icon=folium.Icon(color="red", icon="cutlery", prefix="fa")
                            ).add_to(combined_map)
                            st.markdown(f"""
                            <div style="border:1px solid #ccc; padding:10px; border-radius:5px; margin-bottom:10px;">
                                <h4>{row['Name']}</h4>
                                <p>التصنيف: {row['Category']}</p>
                                <p>التقييم: {row['Rating']} - عدد التقييمات: {row['Total Ratings']}</p>
                                <p>المسافة: {row['distance_km']:.2f} </p>
                            </div>
                            """, unsafe_allow_html=True)

            # Add Venue recommendations markers with different icons for malls and cinemas
            if ("Mall" in st.session_state.selected_panels or "Cinema" in st.session_state.selected_panels) and venue_categories:
                try:
                    df_venue = pd.read_csv("cleaned_venues.csv")
                except Exception as e:
                    st.error(f"Error loading venue data: {e}")
                    df_venue = pd.DataFrame()
                if not df_venue.empty:
                    rec_venue = get_venue_recommendations(df_venue, user_location, rec_radius, venue_categories)
                    if rec_venue is not None:
                        for _, row in rec_venue.iterrows():
                            # Choose the icon based on category
                            if row['Category'] in ["Shopping Mall"]:
                                icon = folium.Icon(color="blue", icon="shopping-bag", prefix="fa")
                            elif row['Category'] in ["Movie Theater"]:
                                icon = folium.Icon(color="purple", icon="film", prefix="fa")
                            else:
                                icon = folium.Icon(color="blue", icon="building", prefix="fa")
                            folium.Marker(
                                location=[row['Latitude'], row['Longitude']],
                                popup=f"Venue: {row['Name']}<br>Category: {row['Category']}<br>Distance: {row['distance_km']:.2f} km",
                                icon=icon
                            ).add_to(combined_map)
                            st.markdown(f"""
                            <div style="border:1px solid #ccc; padding:10px; border-radius:5px; margin-bottom:10px;">
                                <h4>{row['Name']}</h4>
                                <p>التصنيف: {row['Category']}</p>
                                <p>المسافة: {row['distance_km']:.2f} </p>
                            </div>
                            """, unsafe_allow_html=True)

            st.write("### كيف زبطناك؟ 😎")
            folium_static(combined_map, width=700, height=500)
if __name__ == "__main__":
    main()
