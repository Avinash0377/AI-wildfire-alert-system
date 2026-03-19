import requests


def get_location():
    """Get the current location based on IP address.

    Uses the ip-api.com free API to determine city, region, and country.
    Includes a correction for Hyderabad being reported under Andhra Pradesh.

    Returns:
        Location string in 'City, Region, Country' format,
        or 'Location Unavailable' if the API call fails.
    """
    try:
        response = requests.get("http://ip-api.com/json/", timeout=5)
        data = response.json()
        city = data.get('city', 'Unknown')
        region = data.get('regionName', 'Unknown')
        country = data.get('country', 'Unknown')

        # Correct Hyderabad region if misreported
        if city.lower() == "hyderabad" and region.lower() == "andhra pradesh":
            region = "Telangana"

        return f"{city}, {region}, {country}"
    except Exception:
        return "Location Unavailable"
