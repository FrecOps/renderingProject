import json
import zipfile

# Read polygon data from "mkRD.txt".
# Each line in the file is assumed to be a JSON string representing a polygon.
# Note: In your file, each polygon is wrapped in an extra list (e.g., [[{...}, {...}, ...]]).
polygons = []
with open("mkHQ.txt", "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip()
        if line:
            data = json.loads(line)
            # If the data is double nested (i.e., [[ ... ]]), extract the first element.
            if isinstance(data, list) and data and isinstance(data[0], list):
                polygons.append(data[0])
            elif isinstance(data, list):
                polygons.append(data)
            else:
                print("Unexpected data format in line:", line)

def polygon_to_coordinates(polygon):
    """
    Convert a polygon (a list of dictionaries with 'lat' and 'lng') to a
    string of coordinates in KML format: "lng,lat,altitude".
    """
    coords = []
    for point in polygon:
        coords.append(f"{point['lng']},{point['lat']},0")
    # Ensure the polygon is closed by repeating the first coordinate if needed.
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return " ".join(coords)

# Build the KML document.
kml_header = """<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
<Document>
"""
kml_footer = """
</Document>
</kml>
"""

placemarks = ""
for idx, polygon in enumerate(polygons):
    coords_str = polygon_to_coordinates(polygon)
    placemark = f"""
    <Placemark>
        <name>Polygon {idx+1}</name>
        <Style>
            <LineStyle>
                <color>ff0000ff</color>
                <width>2</width>
            </LineStyle>
            <PolyStyle>
                <color>7dff0000</color>
            </PolyStyle>
        </Style>
        <Polygon>
            <outerBoundaryIs>
                <LinearRing>
                    <coordinates>
                        {coords_str}
                    </coordinates>
                </LinearRing>
            </outerBoundaryIs>
        </Polygon>
    </Placemark>
    """
    placemarks += placemark

kml_content = kml_header + placemarks + kml_footer

# Write the KML content to a file named "doc.kml"
with open("doc.kml", "w", encoding="utf-8") as file:
    file.write(kml_content)

# Create the KMZ file (a ZIP archive containing "doc.kml")
kmz_filename = "hq.kmz"
with zipfile.ZipFile(kmz_filename, 'w', zipfile.ZIP_DEFLATED) as kmz:
    kmz.write("doc.kml", arcname="doc.kml")

print(f"KMZ file '{kmz_filename}' created successfully.")