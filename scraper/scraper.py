import sqlite3
import requests
import time
import re
import json
import csv

from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from random import randint

import os

# Database file
DB_FILE = "assets/immo_elliza.db"

sitemap_url = "https://www.immoweb.be/sitemap.xml"
headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'
}


print(f"Database absolute path: {os.path.abspath(DB_FILE)}")

def retry_request(url, headers, retries=3, timeout=10):
    for attempt in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            wait_time = 2 ** attempt + randint(1, 3)  # Exponential backoff with jitter
            print(f"Retry {attempt+1}/{retries} for {url} | Waiting {wait_time}s")
            time.sleep(wait_time)
    raise Exception(f"Failed after {retries} retries: {url}")

def initialize_links_table():
    """
    Initialize the database and create the links table if it doesn't exist.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create the table if it does not exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS links (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE NOT NULL,
            classified_id INTEGER UNIQUE,
            status TEXT DEFAULT 'pending',
            last_checked DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def initialize_properties_table():
    """
    Initialize the 'properties' table to store scraped property details.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Create the properties table if it doesn't exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS properties (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            link_id INTEGER UNIQUE NOT NULL,
            locality_name TEXT,
            postal_code TEXT,
            price TEXT,
            property_type TEXT,
            property_subtype TEXT,
            number_of_bedrooms INTEGER,
            living_area TEXT,
            street TEXT,
            number TEXT,
            latitude TEXT,
            longitude TEXT,
            open_fire TEXT,
            swimming_pool TEXT,
            hasTerrace TEXT,
            terraceSurface TEXT,
            hasGarden TEXT,
            gardenSurface TEXT,
            kitchen_type TEXT,
            number_of_facades INTEGER,
            state_of_building TEXT,
            construction_year INTEGER,
            epc TEXT,
            landSurface TEXT,
            scraped_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (link_id) REFERENCES links(classified_id)
        )
    """)
    conn.commit()
    conn.close()

def get_links_from_sitemap():
    """
    Fetch and parse classified links from sitemap XML files.
    Conditions:
    - hreflang="en-BE"
    - URL contains 'for-sale'
    - URL contains 'apartment' or 'house'
    """
    # Step 1: Fetch the main sitemap and get all XML file links
    response = retry_request(sitemap_url, headers=headers)
    response.raise_for_status()
    
    # Parse the main sitemap
    soup = BeautifulSoup(response.content, "xml")
    xml_files = [loc.text for loc in soup.find_all("loc") if "classified" in loc.text]
    print(f"Found {len(xml_files)} classified sitemap files.")

    # Step 2: Process each XML file and extract filtered links
    filtered_links = set()

    for xml_file in xml_files:
        try:
            # Fetch the content of the XML file
            xml_response = retry_request(xml_file, headers=headers)
            xml_response.raise_for_status()
            
            # Parse the XML content
            xml_soup = BeautifulSoup(xml_response.content, "xml")

            # Loop through each <url> entry
            for url_entry in xml_soup.find_all("url"):
                # Find <xhtml:link> tags with hreflang="en-BE"
                alternate_links = url_entry.find_all("xhtml:link", hreflang="en-BE")
                
                for link in alternate_links:
                    url = link.get("href", "")

                    # Apply the filtering conditions
                    if "for-sale" in url and ("house" in url or "apartment" in url):
                        filtered_links.add(url)

            # with open("assets/list_of_links.txt", "w", encoding="utf-8") as file:
            #     for link in filtered_links:
            #         file.write(link + "\n")
            # print(f"✅ Exported {len(filtered_links)} links to list_of_links.txt")

        except requests.RequestException as e:
            print(f"❌ Failed to process sitemap: {xml_file} | Error: {e}")

    return list(filtered_links)

def extract_classified_id(url):
    """
    Extract the classified ID (last digits) from the URL.
    """
    return url.strip("/").split("/")[-1]  # Extracts the last part of the URL

def update_links(links): #links = filtered_links from Sitemap
    """
    Compare the provided links with the existing database and update the database accordingly.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Log existing classified IDs
    cursor.execute("SELECT classified_id FROM links")
    existing_ids = set(row[0] for row in cursor.fetchall())

    # Get existing URLs from the database
    cursor.execute("SELECT url FROM links")
    existing_links = set(row[0] for row in cursor.fetchall())
    
    # Determine new, existing, and missing links
    links_set = set(links)
    new_links = links_set - existing_links
    still_active_links = links_set & existing_links
    missing_links = existing_links - links_set

    
    
    # Insert new links
   # Insert new links, ignoring duplicates
    for link in new_links:
        classified_id = extract_classified_id(link)
        cursor.execute("""
            INSERT OR IGNORE INTO links (url, classified_id, status, created_at, updated_at)
            VALUES (?, ?, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, (link, classified_id))

    
    # Update timestamps for links still active
    for link in still_active_links:
        cursor.execute("""
            UPDATE links
            SET updated_at = CURRENT_TIMESTAMP
            WHERE url = ?
        """, (link,))
    
    # Optional: Handle missing links (e.g., mark as inactive or delete)
    for link in missing_links:
        cursor.execute("""
            UPDATE links
            SET status = 'inactive', updated_at = CURRENT_TIMESTAMP
            WHERE url = ?
        """, (link,))
    
    conn.commit()
    conn.close()

def safe_get(data, *keys):
    """
    Safely retrieve a nested value from a dictionary.
    If a key does not exist, return None.
    """
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return None
    return data

def process_and_scrape_links_concurrent():
    """
    Process all pending links concurrently: check link status and scrape data in one step.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Fetch all pending links from the database
    cursor.execute("SELECT id, url FROM links WHERE status = 'pending'")
    pending_links = cursor.fetchall()
    conn.close()

    print(f"🔍 Processing {len(pending_links)} links...")

    def process_link(link_id, url):
        """
        Process a single link: check status and scrape data if valid.
        """
        try:
            # Step 1: Perform the GET request
            response = retry_request(url, headers=headers, timeout=10)
            response.raise_for_status()  # Ensure it's a valid link

            # Step 2: Scrape property data (e.g., extract JSON or HTML content)
            soup = BeautifulSoup(response.text, "html.parser")
            script_tag = soup.find("script", string=re.compile("window\\.classified"))
            if not script_tag:
                raise ValueError("No classified script tag found")
            
            match = re.search(r'window\.classified\s*=\s*({.*?});', script_tag.string, re.DOTALL)
            if not match:
                raise ValueError("No classified JSON found in script tag")
            
            classified_data = json.loads(match.group(1))

            # Extract data using a helper function
            kwh = safe_get(classified_data, 'transaction', 'certificates', 'primaryEnergyConsumptionPerSqm')
            epc = safe_get(classified_data, 'transaction', 'certificates', 'epcScore')

            # If EPC is not None or 0 and is not an integer (ensuring it must be a valid EPC string), skip further checks
            if epc not in (None, 0) and not isinstance(epc, int):
                pass  # EPC is already valid, nothing to do

            else:  # EPC is null or 0, so we use kwh to determine it
                if kwh is None:  # If kwh is also null, skip the link
                    raise ValueError("Both epc and kwh are missing, skipping this link")
                
                # Validate kwh: Ensure it's not too low
                if kwh < -100:
                    raise ValueError("kwh too low to be valid, skipping this link")
                
                # Calculate EPC based on kwh
                if -99 < kwh < 0:
                    epc = "A+"
                elif 0 <= kwh < 100:
                    epc = "A"
                elif 100 <= kwh < 200:
                    epc = "B"
                elif 200 <= kwh < 300:
                    epc = "C"
                elif 300 <= kwh < 400:
                    epc = "D"
                elif 400 <= kwh < 500:
                    epc = "E"
                elif kwh >= 500:
                    epc = "F"

            scraped_data = {
                "link_id": link_id,
                "locality_name": safe_get(classified_data, 'property', 'location', 'locality'),
                "postal_code": safe_get(classified_data, 'property', 'location', 'postalCode'),
                "price": safe_get(classified_data, 'transaction', 'sale', 'price'),
                "property_type": safe_get(classified_data, 'property', 'type'),
                "property_subtype": safe_get(classified_data, 'property', 'subtype'),
                "number_of_bedrooms": safe_get(classified_data, 'property', 'bedroomCount'),
                "living_area": safe_get(classified_data, 'property', 'netHabitableSurface'),
                "street": safe_get(classified_data, 'property', 'location', 'street'),
                "number": safe_get(classified_data, 'property', 'location', 'number'),
                "latitude": safe_get(classified_data, 'property', 'location', 'latitude'),
                "longitude": safe_get(classified_data, 'property', 'location', 'longitude'),
                "open_fire": safe_get(classified_data, 'property', 'fireplaceExists'),
                "swimming_pool": safe_get(classified_data, 'property', 'hasSwimmingPool'),
                "hasTerrace": safe_get(classified_data, 'property', 'hasTerrace'),
                "terraceSurface": safe_get(classified_data, 'property', 'terraceSurface'),
                "hasGarden": safe_get(classified_data, 'property', 'hasGarden'),
                "gardenSurface": safe_get(classified_data, 'property', 'gardenSurface'),
                "kitchen_type": safe_get(classified_data, 'property', 'kitchen', 'type'),
                "number_of_facades": safe_get(classified_data, 'property', 'building', 'facadeCount'),
                "state_of_building": safe_get(classified_data, 'property', 'building', 'condition'),
                "construction_year": safe_get(classified_data, 'property', 'building', 'constructionYear'),
                "epc": epc,  # Use the calculated or existing EPC value
                "landSurface": safe_get(classified_data, 'property', 'land', 'surface')
            }

            return link_id, 'scraped', scraped_data, None  # Success

        except requests.RequestException as e:
            return link_id, 'error', None, str(e)  # Network error

        except Exception as e:
            return link_id, 'error', None, str(e)  # Other error


    results = []
    # Use ThreadPoolExecutor for concurrency
    with ThreadPoolExecutor(max_workers=20) as executor:  # Adjust `max_workers` based on your system
        futures = [executor.submit(process_link, link_id, url) for link_id, url in pending_links]
        
        for future in as_completed(futures):
            results.append(future.result())

    # Step 3: Update the database with results
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    for link_id, status, scraped_data, error in results:
        if status == 'scraped':
            # Insert scraped data into the properties table
            cursor.execute("""
                INSERT INTO properties (
                    link_id, locality_name, postal_code, price, property_type, property_subtype,
                    number_of_bedrooms, living_area, street, number, latitude, longitude,
                    open_fire, swimming_pool, hasTerrace, terraceSurface, hasGarden, gardenSurface,
                    kitchen_type, number_of_facades, state_of_building, construction_year,
                    epc, landSurface, scraped_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                scraped_data["link_id"],
                scraped_data["locality_name"],
                scraped_data["postal_code"],
                scraped_data["price"],
                scraped_data["property_type"],
                scraped_data["property_subtype"],
                scraped_data["number_of_bedrooms"],
                scraped_data["living_area"],
                scraped_data["street"],
                scraped_data["number"],
                scraped_data["latitude"],
                scraped_data["longitude"],
                scraped_data["open_fire"],
                scraped_data["swimming_pool"],
                scraped_data["hasTerrace"],
                scraped_data["terraceSurface"],
                scraped_data["hasGarden"],
                scraped_data["gardenSurface"],
                scraped_data["kitchen_type"],
                scraped_data["number_of_facades"],
                scraped_data["state_of_building"],
                scraped_data["construction_year"],
                scraped_data["epc"],
                scraped_data["landSurface"]
            ))
            # Update the link status to 'scraped'
            cursor.execute("""
                UPDATE links
                SET status = 'scraped', last_checked = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (link_id,))
        elif status == 'error':
            # Update the link status to 'error'
            cursor.execute("""
                UPDATE links
                SET status = 'error', last_checked = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (link_id,))
            print(f"❌ Link failed: {link_id} | Error: {error}")

    conn.commit()
    conn.close()

def create_csv_for_preprocessing():
    """
    Export properties table to a CSV so the preporcessing can use it.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    # Define the database columns and desired CSV headers
    db_columns = [
        "id", "link_id", "locality_name", "postal_code", "price",
        "property_type", "property_subtype", "number_of_bedrooms", "living_area",
        "street", "number", "latitude", "longitude", "open_fire",
        "swimming_pool", "hasTerrace", "terraceSurface", "hasGarden",
        "gardenSurface", "kitchen_type", "number_of_facades",
        "state_of_building", "construction_year", "epc",
        "landSurface", "scraped_at"
    ]
    
    csv_headers = [
        "id", "locality_name", "Postal_code", "Price", "Subtype", "Number_of_rooms",
        "Number_of_bedrooms", "Living_area", "sale_annuity", "Type_of_sale",
        "street", "number", "latitude", "longitude", "Open_fire",
        "Swimming_Pool", "hasTerrace", "terraceSurface", "hasGarden",
        "gardenSurface", "Kitchen_type", "Number_of_facades",
        "State_of_building", "Starting_price", "epc", "landSurface"
    ]
    
    # Create a mapping of CSV headers to database columns
    column_mapping = {
        "id": "id",
        "locality_name": "locality_name",
        "Postal_code": "postal_code",
        "Price": "price",
        "Subtype": "property_subtype",
        "Number_of_rooms": None,  # No matching column
        "Number_of_bedrooms": "number_of_bedrooms",
        "Living_area": "living_area",
        "sale_annuity": None,  # No matching column
        "Type_of_sale": None,  # No matching column
        "street": "street",
        "number": "number",
        "latitude": "latitude",
        "longitude": "longitude",
        "Open_fire": "open_fire",
        "Swimming_Pool": "swimming_pool",
        "hasTerrace": "hasTerrace",
        "terraceSurface": "terraceSurface",
        "hasGarden": "hasGarden",
        "gardenSurface": "gardenSurface",
        "Kitchen_type": "kitchen_type",
        "Number_of_facades": "number_of_facades",
        "State_of_building": "state_of_building",
        "Starting_price": None,  # No matching column
        "epc": "epc",
        "landSurface": "landSurface"
    }

    # Fetch data from the database
    cursor.execute(f"SELECT {', '.join(db_columns)} FROM properties")
    rows = cursor.fetchall()

    # Write the data to CSV
    with open("assets/export.csv", mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(csv_headers)  # Write the header row

        for row in rows:
            # Map the database row to the CSV row
            csv_row = [
                row[db_columns.index(column_mapping[csv_header])] if column_mapping[csv_header] in db_columns else "NULL"
                for csv_header in csv_headers
            ]
            writer.writerow(csv_row)


def main():
    """
    Main function to run the script with timers.
    """
    # Step 1: Start the overall timer
    overall_start = time.time()
    
    # Step 2: Initialize the database tables
    print("Initializing database tables...")
    start = time.time()
    initialize_links_table()
    initialize_properties_table()
    print(f"⏲️ Database initialization completed in {time.time() - start:.2f} seconds\n")
    
    # Step 3: Fetch links from sitemap
    print("Fetching links from sitemap...")
    start = time.time()
    links = get_links_from_sitemap()
    print(f"⏲️ Fetched {len(links)} links from sitemap in {time.time() - start:.2f} seconds\n")
    
    # Step 4: Update the database with the new/updated links
    print("Updating the database with new links...")
    start = time.time()
    update_links(links)
    print(f"⏲️ Database update completed in {time.time() - start:.2f} seconds\n")
    
    # Step 5: Process pending links and store scraped data
    print("Processing pending links...")
    start = time.time()
    process_and_scrape_links_concurrent()  # Combine checking and scraping into one step
    print(f"⏲️ Link processing and scraping completed in {time.time() - start:.2f} seconds\n")
    
    # Step 6: create a csv file for the preprocessing to start
    print("CSV Creating process starting...")
    start = time.time()
    create_csv_for_preprocessing()
    print(f"⏲️ CSV created {time.time() - start:.2f} seconds\n")
    
    # Step 7: Total time taken
    print(f"⏲️ Entire script completed in {time.time() - overall_start:.2f} seconds ⏲️")

if __name__ == "__main__":
    main()