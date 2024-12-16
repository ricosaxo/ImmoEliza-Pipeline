import sqlite3
import requests

from datetime import datetime
from bs4 import BeautifulSoup

# Database file
DB_FILE = "assets/sitemap_links.db"

sitemap_url = "https://www.immoweb.be/sitemap.xml"
headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'
}

def get_links_from_sitemap():
    """
    Function to fetch and parse links from your sitemap XML files.
    """
    # Function to get all classified sitemap URLs
    response = requests.get(sitemap_url, headers=headers) # Fetch the sitemap
    response.raise_for_status()  # Raise an error if the request failed

    # Parse the XML content
    soup = BeautifulSoup(response.content, "xml")

    # Find all <loc> tags and filter URLs with "classified"
    classified_urls = [loc.text for loc in soup.find_all("loc") if "classified" in loc.text]

    return classified_urls


def initialize_database():
    """
    Initialize the database and create the links table if it doesn't exist.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Create the table if it does not exist
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS links (
            id INTEGER PRIMARY KEY,
            url TEXT UNIQUE NOT NULL,
            status TEXT DEFAULT 'pending',
            last_checked DATETIME,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

def update_links(links):
    """
    Compare the provided links with the existing database and update the database accordingly.
    """
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Get existing URLs from the database
    cursor.execute("SELECT url FROM links")
    existing_links = set(row[0] for row in cursor.fetchall())
    
    # Determine new, existing, and missing links
    links_set = set(links)
    new_links = links_set - existing_links
    still_active_links = links_set & existing_links
    missing_links = existing_links - links_set
    
    # Insert new links
    for link in new_links:
        cursor.execute("""
            INSERT INTO links (url, status, created_at, updated_at)
            VALUES (?, 'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
        """, (link,))
    
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

def main():
    """
    Main function to run the script.
    """
    # Step 1: Initialize the database
    initialize_database()
    
    # Step 2: Fetch links from sitemap
    links = get_links_from_sitemap()  # Fill this in with your sitemap logic
    
    # Step 3: Update the database with the new/updated links
    update_links(links)

if __name__ == "__main__":
    main()
